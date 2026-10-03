"""Console output for the CLI: tables, progress, the interactive model picker."""

from __future__ import annotations

import functools
import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from .core.agent import StepEvent
from .core.casefile import CaseError
from .core.llm import LLMError, Usage
from .core.models import STAGES, CaseFile, Role, Stage, display_usd
from .core.parsing import FormatError
from .core.pipeline import PlanError
from .core.thread import ThreadEntry, ThreadError
from .decisionlayer.client import DecisionLayerError, SimulationFailed, SimulationTimeout
from .decisionlayer.live import LiveCaseError
from .decisionlayer.samples import SampleError
from .decisionlayer.simulator import SimulationInputError
from .llm.catalog import Catalog, ModelInfo

console = Console()
err_console = Console(stderr=True)

HANDLED_ERRORS = (
    CaseError,
    PlanError,
    ThreadError,
    FormatError,
    LLMError,
    DecisionLayerError,
    SampleError,
    SimulationInputError,
    SimulationFailed,
    SimulationTimeout,
    LiveCaseError,
    FileNotFoundError,
    ValueError,
)


def handle_errors(command: Callable[..., Any]) -> Callable[..., Any]:
    """Print a known error plainly and exit 1 instead of showing a traceback."""

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except HANDLED_ERRORS as exc:
            if os.environ.get("CASEAGENT_DEBUG"):
                raise
            err_console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from None
        except KeyboardInterrupt:
            err_console.print("\n[yellow]Interrupted.[/yellow]")
            raise typer.Exit(code=130) from None

    return wrapper


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def confirm(question: str, yes: bool) -> bool:
    if yes:
        return True
    if not interactive():
        raise typer.BadParameter(f"{question} Re-run with --yes to confirm non-interactively.")
    return typer.confirm(question, default=False)


def ask(question: str, default: str = "") -> str:
    """Prompt for a line of text; an answer of ``@path`` reads the file at that path."""
    answer = Prompt.ask(question, default=default or None, show_default=bool(default)) or ""
    answer = answer.strip()
    if answer.startswith("@"):
        path = os.path.expanduser(answer[1:].strip())
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                return handle.read().strip()
        except OSError as exc:
            console.print(f"[yellow]Could not read {path}: {exc}[/yellow]")
            return ask(question, default)
    return answer


# ------------------------------------------------------------------------- tables


def money(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.2f}"


def tokens(value: int) -> str:
    return f"{value:,}"


def usage_line(usage: Usage) -> str:
    parts = [
        f"{usage.calls} call{'s' if usage.calls != 1 else ''}",
        f"{tokens(usage.prompt_tokens)} in",
        f"{tokens(usage.completion_tokens)} out",
    ]
    if usage.reasoning_tokens:
        parts.append(f"{tokens(usage.reasoning_tokens)} reasoning")
    if usage.cached_tokens:
        parts.append(f"{tokens(usage.cached_tokens)} cached")
    if usage.cost_usd is not None:
        parts.append(f"{money(usage.cost_usd)}")
    return " · ".join(parts)


def thread_table(
    case: CaseFile,
    entries: list[ThreadEntry],
    to_write: Mapping[Stage, str],
    title: str = "Thread",
) -> Table:
    """Each phase: who filed it, or who is about to write it."""
    table = Table(title=title, show_lines=False)
    table.add_column("Phase")
    table.add_column("Filed by")
    table.add_column("Status")
    table.add_column("Words", justify="right")
    filed = {entry.stage: entry for entry in entries}
    for stage in STAGES:
        party = f"{case.party(stage.role).name} ({stage.role.value})"
        if stage in filed:
            entry = filed[stage]
            table.add_row(
                stage.title, party, f"filed · {entry.written_by}", str(entry.filing.word_count)
            )
        elif stage in to_write:
            table.add_row(
                stage.title, party, f"[bold]to write[/bold] · agent ({to_write[stage]})", ""
            )
        else:
            table.add_row(stage.title, party, "[dim]not filed[/dim]", "")
    return table


def filings_table(case: CaseFile, entries: list[ThreadEntry]) -> Table:
    table = Table(title="Filings", show_lines=False)
    table.add_column("Phase")
    table.add_column("Party")
    table.add_column("Written by")
    table.add_column("Words", justify="right")
    table.add_column("Extras")
    filed = {entry.stage: entry for entry in entries}
    for stage in STAGES:
        entry = filed.get(stage)
        party = case.party(stage.role).name
        if entry is None:
            table.add_row(stage.title, party, "[dim]not filed[/dim]", "-", "")
            continue
        filing = entry.filing
        extras = []
        if filing.counterclaim:
            amount = filing.counterclaim.amount_usd
            extras.append(
                "counterclaim" + (f" {display_usd(amount)}" if amount is not None else "")
            )
        if filing.evidence_demands:
            extras.append("documentation demands")
        if filing.evidence_response:
            extras.append("response to demands")
        table.add_row(
            stage.title, party, entry.written_by, str(filing.word_count), ", ".join(extras)
        )
    return table


def models_table(models: list[ModelInfo], title: str = "Models") -> Table:
    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model")
    table.add_column("Context", justify="right")
    table.add_column("$/M in", justify="right")
    table.add_column("$/M out", justify="right")
    table.add_column("Max out", justify="right")
    table.add_column("Effort levels")
    for number, model in enumerate(models, start=1):
        table.add_row(
            str(number),
            model.id,
            f"{model.context_length:,}" if model.context_length else "-",
            f"{model.prompt_price:.2f}" if model.prompt_price is not None else "-",
            f"{model.completion_price:.2f}" if model.completion_price is not None else "-",
            f"{model.max_output_tokens:,}" if model.max_output_tokens else "-",
            ", ".join(model.efforts) if model.efforts else "-",
        )
    return table


def print_filing(entry: ThreadEntry, party: str) -> None:
    """Show a filing the way the principal would read it."""
    filing = entry.filing
    console.rule(f"{filing.stage.title} · {party} · {entry.written_by}")
    console.print(filing.argument)
    if filing.counterclaim:
        amount = filing.counterclaim.amount_usd
        console.rule("Counterclaim" + (f" · {display_usd(amount)}" if amount is not None else ""))
        console.print(filing.counterclaim.argument)
        if filing.counterclaim.other_relief:
            console.print(f"\nOther relief: {filing.counterclaim.other_relief}")
    if filing.evidence_response:
        console.rule("Response to documentation demands")
        console.print(filing.evidence_response)
    if filing.evidence_demands:
        console.rule("Documentation demanded")
        console.print(filing.evidence_demands)
    if entry.notes_for_principal:
        console.rule("Notes for you (not filed)")
        console.print(entry.notes_for_principal)
    for warning in entry.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.rule()


# ----------------------------------------------------------------------- progress


class StageProgress:
    """Drives a spinner from the agent's step events."""

    def __init__(self, party_names: Mapping[Role, str], models: Mapping[Role, str]) -> None:
        self.party_names = dict(party_names)
        self.models = dict(models)
        self._status: Any = None
        self._started = time.monotonic()
        self._stage_label = ""

    def __enter__(self) -> StageProgress:
        self._status = console.status("Starting...", spinner="dots")
        self._status.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._status.__exit__(*exc)

    def on_stage(self, stage: Stage, phase: str) -> None:
        party = self.party_names[stage.role]
        model = self.models.get(stage.role, "?")
        if phase == "start":
            self._started = time.monotonic()
            self._stage_label = f"{stage.title} · {party}"
            if self._status is not None:
                self._status.update(f"{self._stage_label} · {model} · starting")
            return
        elapsed = time.monotonic() - self._started
        console.print(f"[green]✓[/green] {self._stage_label} written by {model} in {elapsed:.0f}s")

    def on_event(self, event: StepEvent) -> None:
        elapsed = time.monotonic() - self._started
        if event.phase == "end" and event.response is not None:
            console.print(
                f"  [dim]{event.step}: {usage_line(event.response.usage)} · "
                f"{event.response.seconds:.0f}s[/dim]"
            )
            return
        if self._status is None:
            return
        progress = f" · {event.chars:,} chars" if event.chars else ""
        self._status.update(f"{self._stage_label} · {event.step}{progress} · {elapsed:.0f}s")


# ------------------------------------------------------------------------- picker


def pick_model(catalog: Catalog | None, default: str, who: str) -> str:
    """Interactive model choice: a featured list, free-text search, or a typed id."""
    if catalog is None:
        return Prompt.ask(f"Model for the {who}", default=default)
    shown = catalog.featured()
    console.print(models_table(shown, title=f"Pick a model for the {who}"))
    console.print(
        "[dim]Enter a number, a model id, or '/text' to search the catalog "
        f"(Enter keeps {default}).[/dim]"
    )
    while True:
        answer = Prompt.ask("Model", default=default).strip()
        if answer.startswith("/"):
            shown = catalog.search(answer[1:], limit=25)
            if not shown:
                console.print("[yellow]No models match.[/yellow]")
                continue
            console.print(models_table(shown, title=f"Models matching {answer[1:]!r}"))
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(shown):
            return shown[int(answer) - 1].id
        if catalog.get(answer) is not None or ":" in answer:
            return answer
        suggestions = catalog.suggest(answer)
        hint = f" Close matches: {', '.join(suggestions)}." if suggestions else ""
        console.print(f"[yellow]{answer!r} is not in the catalog.{hint}[/yellow]")
