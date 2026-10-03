"""Command-line interface: ``caseagent``."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.table import Table

from . import __version__
from .core.agent import CaseAgent
from .core.casefile import (
    CaseBundle,
    CaseError,
    add_evidence,
    list_cases,
    resolve_case,
    save_case,
    scaffold_case,
)
from .core.documents import extract_text
from .core.models import (
    STAGES,
    Author,
    CaseFile,
    Counterclaim,
    Filing,
    Role,
    Stage,
    Statement,
    display_usd,
)
from .core.options import ComposeOptions, CounterclaimPolicy, Depth
from .core.pipeline import continue_thread, remaining_stages
from .core.runs import (
    finish_run,
    list_run_dirs,
    load_run,
    render_transcript,
    resolve_run,
    start_run,
)
from .core.thread import ThreadEntry, ThreadStore
from .core.view import build_stage_view
from .decisionlayer import live as live_api
from .decisionlayer.client import (
    TEST_KEY_PREFIX,
    Attachment,
    DecisionLayerClient,
    DecisionLayerError,
    Simulation,
)
from .decisionlayer.samples import SAMPLE_ARGUMENTS_FILENAME, SampleStore, fetch_sample, sample_key
from .decisionlayer.simulator import (
    SimulationRecord,
    SimulationRequest,
    await_award,
    build_simulation_request,
    extract_award_section,
    load_simulation,
    save_simulation,
    submit_simulation,
)
from .llm.catalog import Catalog, load_catalog
from .llm.factory import BuiltClient, ModelOptions, build_client, is_fake
from .llm.providers import split_model_spec
from .settings import Settings
from .ui import (
    StageProgress,
    ask,
    confirm,
    console,
    filings_table,
    handle_errors,
    interactive,
    models_table,
    money,
    pick_model,
    print_filing,
    thread_table,
    tokens,
    usage_line,
)

app = typer.Typer(
    help=(
        "Compose arbitration filings with an LLM, one turn at a time, and test them on the "
        "DecisionLayer simulator."
    ),
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
samples_app = typer.Typer(help="DecisionLayer's sample disputes.", no_args_is_help=True)
case_app = typer.Typer(
    help="Local case folders (the dispute and its thread).", no_args_is_help=True
)
sim_app = typer.Typer(help="Simulations on the DecisionLayer API.", no_args_is_help=True)
live_app = typer.Typer(
    help="Real cases on the DecisionLayer API (/api/v1/cases).", no_args_is_help=True
)
app.add_typer(samples_app, name="samples")
app.add_typer(case_app, name="case")
app.add_typer(sim_app, name="sim")
app.add_typer(live_app, name="live")


def _version(value: bool) -> None:
    if value:
        console.print(f"caseagent {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    pass


# ------------------------------------------------------------------------ helpers


class ConfigError(ValueError):
    """Something in the environment or arguments stops the command from running."""


def _model_options(
    settings: Settings,
    effort: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ModelOptions:
    return ModelOptions(
        effort=effort if effort is not None else settings.effort,
        effort_explicit=effort is not None or settings.effort_explicit,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        temperature=temperature,
        timeout=settings.llm_timeout,
    )


def _build(
    spec: str,
    settings: Settings,
    model_options: ModelOptions,
    catalog: Catalog | None,
    session_id: str,
) -> BuiltClient:
    built = build_client(
        spec,
        model_options,
        catalog,
        reload_catalog=lambda: load_catalog(settings.cache_dir, refresh=True),
        session_id=session_id,
    )
    for note in built.notes:
        console.print(f"[yellow]Note:[/yellow] {note}")
    return built


def _party_names(case: CaseFile) -> dict[Role, str]:
    return {role: case.party(role).name or role.title for role in Role}


def _ensure_essentials(bundle: CaseBundle) -> CaseBundle:
    """Ask for the case fields the arbitration cannot do without, and save them."""
    case = bundle.case
    missing = case.missing_essentials()
    if not missing:
        return bundle
    if not interactive():
        raise ConfigError(
            f"{bundle.root / 'case.json'} is missing {', '.join(missing)}. Fill it in, or run "
            "`caseagent next` in a terminal to be asked for it."
        )
    console.print(
        f"[bold]{case.title}[/bold] needs a few details before anything can be written. "
        "Answer in a line, or type @path to read an answer from a file."
    )
    updates: dict = {}
    if not case.claimant.name.strip():
        updates["claimant"] = {"name": ask("Claimant's full name (the party bringing the claim)")}
    if not case.respondent.name.strip():
        updates["respondent"] = {
            "name": ask("Respondent's full name (the party being claimed against)")
        }
    if not case.question_for_arbitration.strip():
        updates["question_for_arbitration"] = ask(
            "The dispute, as it should appear on the claim form: what happened and what the "
            "claimant wants decided"
        )
        updates["financial_demand_usd"] = (
            ask("Amount demanded in USD (0 if none)", "0.00") or "0.00"
        )
        other = ask("Any non-monetary relief demanded (Enter for none)")
        updates["other_relief"] = other or None
    if not (case.governing_contract or "").strip() and not case.contract_file:
        answer = ask(
            "The governing contract: paste its text, @path to a .txt, or the path to a "
            ".pdf/.docx/.rtf/.txt file"
        )
        path = Path(os.path.expanduser(answer))
        if path.is_file() and path.suffix.lower() in {".pdf", ".docx", ".rtf", ".txt"}:
            target = bundle.root / path.name
            if target.resolve() != path.resolve():
                target.write_bytes(path.read_bytes())
            updates["contract_file"] = path.name
            extracted = extract_text(target)
            if extracted.warning:
                console.print(f"[yellow]{extracted.warning}[/yellow]")
        elif answer:
            updates["governing_contract"] = answer
    if not updates:
        return bundle
    new_case = CaseFile.model_validate({**case.model_dump(), **updates})
    if new_case.missing_essentials():
        raise ConfigError("Still missing: " + ", ".join(new_case.missing_essentials()))
    return save_case(new_case, bundle.root)


def _interview(
    agent: CaseAgent,
    bundle: CaseBundle,
    stage: Stage,
    thread: dict[Stage, Filing],
    rounds: int,
    max_questions: int,
) -> CaseBundle:
    """Let the agent ask the party questions; save the answers as that party's statement."""
    role = stage.role
    party = bundle.case.party(role).name
    console.print(
        f"\nThe agent will ask {party} a few questions before writing the {stage.title}. "
        "Answer in a line, type @path to read an answer from a file, or press Enter to skip one."
    )
    for round_number in range(rounds):
        view = build_stage_view(bundle, stage, thread)
        with console.status("Working out what to ask..."):
            questions = agent.questions(
                view, max_questions=max_questions, follow_up=round_number > 0
            )
        if not questions:
            console.print("[dim]The agent has what it needs.[/dim]")
            break
        statements = list(bundle.case.statements.get(role, []))
        for number, question in enumerate(questions, start=1):
            answer = ask(f"[bold]{number}/{len(questions)}[/bold] {question}")
            if answer:
                statements.append(Statement(question=question, answer=answer))
        case = bundle.case.model_copy(
            update={"statements": {**bundle.case.statements, role: statements}}
        )
        bundle = save_case(case, bundle.root)
        max_questions = max(3, max_questions // 2)

    paths = ask(
        f"Documents {party} wants to file with this {stage.title} (paths separated by commas; "
        "Enter for none)"
    )
    if paths:
        files = [
            Path(os.path.expanduser(part.strip())) for part in paths.split(",") if part.strip()
        ]
        bundle = add_evidence(bundle, stage, files)
        console.print(
            f"[green]✓[/green] Added {len(files)} file(s) to evidence/ for the {stage.title}."
        )
    return bundle


def _hint_after(bundle: CaseBundle, store: ThreadStore) -> None:
    upcoming = store.next_stage()
    if upcoming is None:
        console.print(
            f"[dim]The thread is complete. `caseagent simulate {bundle.case.key}` sends it to the "
            "simulator for an award.[/dim]"
        )
        return
    who = bundle.case.party(upcoming.role).name
    console.print(
        f"[dim]Next: `caseagent next {bundle.case.key}` writes the {upcoming.title} for {who}, or "
        f"`caseagent case add-filing {bundle.case.key} {upcoming.value} FILE` records one you "
        f"received. `caseagent simulate {bundle.case.key}` has the agent finish both sides and "
        "submit.[/dim]"
    )


# ------------------------------------------------------------------------ samples


@samples_app.command("list")
@handle_errors
def samples_list() -> None:
    """List the sample disputes on the DecisionLayer simulator page."""
    settings = Settings.load()
    store = SampleStore(settings.decisionlayer_base_url)
    index = store.index()
    local = {bundle.case.key for bundle in list_cases(settings.cases_dir)}
    table = Table(title=f"Sample disputes at {settings.decisionlayer_base_url}")
    table.add_column("Key")
    table.add_column("File")
    table.add_column("Fetched")
    for key, filename in sorted(index.items()):
        table.add_row(key, filename, "yes" if key in local else "")
    console.print(table)
    console.print(
        f"[dim]Fetch one with `caseagent samples fetch <key>` into {settings.cases_dir}/.[/dim]"
    )


@samples_app.command("fetch")
@handle_errors
def samples_fetch(
    keys: Annotated[
        list[str] | None, typer.Argument(help="Sample keys (see `samples list`).")
    ] = None,
    all_samples: Annotated[bool, typer.Option("--all", help="Fetch every sample.")] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite an existing case folder.")
    ] = False,
) -> None:
    """Download sample disputes (claim form, contract, exhibits) into the cases directory.

    Only the dispute is kept as the case. The sample's own written arguments are saved
    to sample-arguments.md for you to read; the agent never sees or sends them.
    """
    settings = Settings.load()
    store = SampleStore(settings.decisionlayer_base_url)
    index = store.index()
    wanted = sorted(index) if all_samples else [sample_key(key) for key in keys or []]
    if not wanted:
        raise typer.BadParameter("Give one or more sample keys, or --all.")
    for key in wanted:
        target = settings.cases_dir / key
        if (target / "case.json").exists() and not force:
            console.print(f"[dim]·[/dim] {key}: already fetched ({target}); use --force to refetch")
            continue
        with console.status(f"Fetching {key}..."):
            bundle = fetch_sample(key, settings.cases_dir, store)
        exhibits = len(bundle.case.evidence_names())
        console.print(
            f"[green]✓[/green] {key}: {bundle.case.title} · {exhibits} exhibit"
            f"{'s' if exhibits != 1 else ''} · {bundle.root}"
        )


# -------------------------------------------------------------------------- cases


@case_app.command("list")
@handle_errors
def case_list() -> None:
    """List local cases and how far each thread has got."""
    settings = Settings.load()
    bundles = list_cases(settings.cases_dir)
    if not bundles:
        console.print(f"No cases in {settings.cases_dir}. Try `caseagent samples fetch --all`.")
        return
    table = Table(title=f"Cases in {settings.cases_dir}")
    table.add_column("Key")
    table.add_column("Title")
    table.add_column("Claimant")
    table.add_column("Respondent")
    table.add_column("Demand", justify="right")
    table.add_column("Exhibits", justify="right")
    table.add_column("Filed")
    for bundle in bundles:
        case = bundle.case
        entries = ThreadStore(bundle.root).entries()
        table.add_row(
            case.key,
            case.title,
            case.claimant.name or "-",
            case.respondent.name or "-",
            display_usd(case.financial_demand_usd),
            str(len(case.evidence_names())),
            ", ".join(entry.stage.title for entry in entries) or "-",
        )
    console.print(table)


@case_app.command("show")
@handle_errors
def case_show(
    case: Annotated[str, typer.Argument(help="Case key or path to a case folder.")],
    full: Annotated[bool, typer.Option("--full", help="Print the filings in full.")] = False,
) -> None:
    """Show a case: parties, dispute, exhibits per phase, and the thread so far."""
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    c = bundle.case
    store = ThreadStore(bundle.root)
    entries = store.entries()
    console.print(f"[bold]{c.title}[/bold]  ({c.key})  {bundle.root}")
    console.print(f"Claimant:   {c.claimant.name or '-'}")
    console.print(f"Respondent: {c.respondent.name or '-'}")
    console.print(f"Demand:     {display_usd(c.financial_demand_usd)}")
    if c.other_relief:
        console.print(f"Other relief: {c.other_relief}")
    contract = bundle.contract()
    console.print(
        f"Contract:   {len(contract.text):,} characters"
        + (f" from {c.contract_file}" if c.contract_file else "")
        + (f" [yellow]({contract.warning})[/yellow]" if contract.warning else "")
    )
    for role in Role:
        count = len(c.statements.get(role, []))
        if count:
            console.print(
                f"Statements: {count} answer{'s' if count != 1 else ''} from the {role.value}"
            )
    if missing := c.missing_essentials():
        console.print(f"[yellow]Missing: {', '.join(missing)}[/yellow]")
    console.print()
    console.print("[bold]Claim form[/bold]")
    console.print(c.question_for_arbitration.strip() or "[dim](empty)[/dim]")
    console.print()
    console.print(thread_table(c, entries, {}, title="Thread"))
    for stage in STAGES:
        exhibits = c.evidence.get(stage, [])
        if exhibits:
            console.print(f"[bold]{stage.title}[/bold] exhibits ({c.party(stage.role).name}):")
            for name in exhibits:
                console.print(f"  - {name}")
    if full:
        for entry in entries:
            console.print()
            print_filing(entry, c.party(entry.stage.role).name)
    if (bundle.root / SAMPLE_ARGUMENTS_FILENAME).is_file():
        console.print(
            f"[dim]The sample's own arguments are in {bundle.root / SAMPLE_ARGUMENTS_FILENAME} "
            "(reference only; never used).[/dim]"
        )


@case_app.command("init")
@handle_errors
def case_init(
    directory: Annotated[Path, typer.Argument(help="Folder to create for the case.")],
    key: Annotated[
        str | None, typer.Option("--key", help="Case key (defaults to the folder name).")
    ] = None,
    title: Annotated[
        str | None, typer.Option("--title", help="A short name for the dispute.")
    ] = None,
) -> None:
    """Start a case folder for a dispute of your own. `caseagent next` then asks for the details."""
    bundle = scaffold_case(directory, key or sample_key(directory.name), title=title)
    console.print(f"[green]✓[/green] Created {bundle.root}/")
    console.print(
        f"Run `caseagent next {bundle.root}` to be asked about the dispute and have the Claim "
        "written, or `caseagent case add-filing` to record a claim you received."
    )


@case_app.command("add-evidence")
@handle_errors
def case_add_evidence(
    case: Annotated[str, typer.Argument(help="Case key or path.")],
    stage: Annotated[
        str, typer.Argument(help="Phase the files are filed with: claim, answer, reply, response.")
    ],
    files: Annotated[list[Path], typer.Argument(help="Files to copy into evidence/.")],
) -> None:
    """Copy exhibit files into the case and list them under the phase that files them."""
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    phase = Stage.parse(stage)
    bundle = add_evidence(bundle, phase, files)
    console.print(
        f"[green]✓[/green] {len(files)} file(s) listed under the {phase.title} "
        f"({bundle.case.party(phase.role).name})."
    )


@case_app.command("add-filing")
@handle_errors
def case_add_filing(
    case: Annotated[str, typer.Argument(help="Case key or path.")],
    stage: Annotated[
        str, typer.Argument(help="Phase of the filing: claim, answer, reply, response.")
    ],
    file: Annotated[Path, typer.Argument(help="Text file with the argument as filed.")],
    counterclaim_file: Annotated[
        Path | None, typer.Option("--counterclaim-file", help="Answer only: the counterclaim text.")
    ] = None,
    counterclaim_amount: Annotated[
        str | None,
        typer.Option(
            "--counterclaim-amount", help="Answer only: amount demanded on the counterclaim."
        ),
    ] = None,
) -> None:
    """Record a filing that already exists (yours, or one you received from the other side)."""
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    phase = Stage.parse(stage)
    store = ThreadStore(bundle.root)
    text = extract_text(file)
    if not text.readable:
        raise CaseError(text.warning or f"Could not read any text from {file}")
    counterclaim = None
    if counterclaim_file is not None:
        counterclaim = Counterclaim(
            argument=extract_text(counterclaim_file).text, amount_usd=counterclaim_amount
        )
    filing = Filing(stage=phase, argument=text.text, counterclaim=counterclaim)
    entry = ThreadEntry(filing=filing, author=Author.HUMAN)
    store.append(entry)
    console.print(
        f"[green]✓[/green] Recorded the {phase.title} for {bundle.case.party(phase.role).name} "
        f"({filing.word_count} words)."
    )
    _hint_after(bundle, store)


@case_app.command("reset")
@handle_errors
def case_reset(
    case: Annotated[str, typer.Argument(help="Case key or path.")],
    last: Annotated[
        bool, typer.Option("--last", help="Remove only the most recent filing.")
    ] = False,
    statements: Annotated[
        bool, typer.Option("--statements", help="Also forget the parties' interview answers.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Remove filings from a case's thread so it can be played again."""
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    store = ThreadStore(bundle.root)
    entries = store.entries()
    if not entries and not statements:
        console.print("The thread is already empty.")
        return
    what = f"the {entries[-1].stage.title}" if last and entries else f"all {len(entries)} filing(s)"
    if statements:
        what += " and the interview answers"
    if not confirm(f"Remove {what} from {bundle.case.key}?", yes):
        raise typer.Exit(code=1)
    if last:
        store.remove_last()
    else:
        store.clear()
    if statements:
        save_case(bundle.case.model_copy(update={"statements": {}}), bundle.root)
    console.print(f"[green]✓[/green] Removed {what}.")


# ------------------------------------------------------------------------- models


@app.command("models")
@handle_errors
def models(
    query: Annotated[
        list[str] | None, typer.Argument(help="Words that must appear in the id or name.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Rows to show.")] = 40,
    refresh: Annotated[bool, typer.Option("--refresh", help="Refetch the catalog.")] = False,
) -> None:
    """Browse OpenRouter's model catalog (newest first)."""
    settings = Settings.load()
    catalog = load_catalog(settings.cache_dir, refresh=refresh)
    if catalog is None:
        raise ConfigError("Could not load the OpenRouter catalog (offline?).")
    if query:
        words = " ".join(query)
        found = catalog.search(words, limit=limit)
        title = f"Models matching {words!r}"
    else:
        found = catalog.featured()
        title = "Featured models (search with `caseagent models <words>`)"
    if not found:
        console.print("No models match.")
        return
    console.print(models_table(found, title=title))
    fetched = datetime.fromtimestamp(catalog.fetched_at).strftime("%Y-%m-%d %H:%M")
    stale = " [yellow](stale: the catalog could not be refreshed)[/yellow]" if catalog.stale else ""
    console.print(f"[dim]{len(catalog.models)} models in catalog, fetched {fetched}{stale}.[/dim]")
    console.print(
        "[dim]Use an id with -m, e.g. `-m anthropic/claude-opus-5.5`; "
        "prefix `openai:` or `custom:` for other providers.[/dim]"
    )


# --------------------------------------------------------------------------- next


def _next(
    case: str,
    model: str | None,
    pick: bool,
    interview: bool | None,
    rounds: int,
    max_questions: int,
    depth: Depth,
    counterclaims: CounterclaimPolicy,
    evidence_demands: bool,
    max_words: int,
    effort: str | None,
    max_tokens: int | None,
    temperature: float | None,
    expect: str | None = None,
) -> None:
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    store = ThreadStore(bundle.root)
    stage = store.next_stage()
    if stage is None:
        raise ConfigError(
            f"{bundle.case.key} already has all four filings. `caseagent simulate "
            f"{bundle.case.key}` sends them to the simulator; `caseagent case reset` starts over."
        )
    if expect == "start" and stage is not Stage.CLAIM:
        raise ConfigError(
            f"{bundle.case.key} already has a {STAGES[stage.index - 1].title}; use `caseagent "
            f"reply` (or `next`) to write the {stage.title}."
        )
    if expect == "reply" and stage is Stage.CLAIM:
        raise ConfigError(
            f"{bundle.case.key} has no filings to reply to yet; use `caseagent start` (or `next`) "
            "to write the Claim."
        )
    bundle = _ensure_essentials(bundle)
    case_file = bundle.case
    party = case_file.party(stage.role).name
    console.print(
        f"[bold]{case_file.title}[/bold]: writing the {stage.title} for {party} "
        f"({stage.role.value})."
    )

    spec = model or settings.model
    catalog = load_catalog(settings.cache_dir)
    if pick:
        spec = pick_model(catalog, spec, f"{stage.role.value} ({party})")
    built = _build(
        spec,
        settings,
        _model_options(settings, effort, max_tokens, temperature),
        catalog,
        f"caseagent-{case_file.key}",
    )
    options = ComposeOptions(
        depth=depth,
        counterclaims=counterclaims,
        evidence_demands=evidence_demands,
        max_words=max_words,
    )
    progress = StageProgress(_party_names(case_file), {stage.role: built.label})
    agent = CaseAgent(built.client, options, on_event=progress.on_event)

    if interview is None:
        interview = stage is Stage.CLAIM and not case_file.is_sample and interactive()
    if interview:
        if not interactive():
            raise ConfigError("--interview needs a terminal to ask questions in.")
        bundle = _interview(agent, bundle, stage, store.thread(), rounds, max_questions)

    view = build_stage_view(bundle, stage, store.thread())
    with progress:
        progress.on_stage(stage, "start")
        result = agent.compose(view)
        progress.on_stage(stage, "end")
    entry = ThreadEntry.from_result(result, built.label)
    path = store.append(entry, result)
    console.print()
    print_filing(entry, party)
    console.print(f"Model usage: {usage_line(result.usage)}")
    console.print(f"Saved → {path.parent}/ (prompts and drafts in {store.work_dir}/)")
    _hint_after(bundle, store)


NEXT_HELP = {
    "case": "Case key under the cases directory, or a path to a case folder.",
    "model": "Model to write with. Default: CASEAGENT_MODEL.",
    "pick": "Choose the model interactively from the catalog.",
    "interview": "Ask the party questions first (default: when starting a case of your own).",
    "rounds": "Interview rounds (a follow-up round asks about the answers).",
    "max_questions": "Questions per interview round.",
    "depth": "quick: draft only; standard: brief then draft; thorough: adds a cite-check revision.",
    "counterclaims": "auto: assert one in the Answer if the record supports it.",
    "evidence_demands": "Let the filing demand documentation from the other side.",
    "max_words": "Target length of the argument.",
    "effort": "Reasoning effort: low, medium, high, xhigh, max, or 'default'.",
    "max_tokens": "Output-token cap per model call.",
    "temperature": "Sampling temperature (sent only when set).",
}


def _register_next(name: str, expect: str | None, hidden: bool, help_text: str) -> None:
    @app.command(name, hidden=hidden, help=help_text)
    @handle_errors
    def command(
        case: Annotated[str, typer.Argument(help=NEXT_HELP["case"])],
        model: Annotated[str | None, typer.Option("--model", "-m", help=NEXT_HELP["model"])] = None,
        pick: Annotated[bool, typer.Option("--pick", "-p", help=NEXT_HELP["pick"])] = False,
        interview: Annotated[
            bool | None, typer.Option("--interview/--no-interview", help=NEXT_HELP["interview"])
        ] = None,
        rounds: Annotated[int, typer.Option("--rounds", help=NEXT_HELP["rounds"])] = 2,
        max_questions: Annotated[
            int, typer.Option("--max-questions", help=NEXT_HELP["max_questions"])
        ] = 8,
        depth: Annotated[Depth, typer.Option("--depth", help=NEXT_HELP["depth"])] = Depth.THOROUGH,
        counterclaims: Annotated[
            CounterclaimPolicy, typer.Option("--counterclaims", help=NEXT_HELP["counterclaims"])
        ] = CounterclaimPolicy.AUTO,
        evidence_demands: Annotated[
            bool,
            typer.Option(
                "--evidence-demands/--no-evidence-demands", help=NEXT_HELP["evidence_demands"]
            ),
        ] = False,
        max_words: Annotated[int, typer.Option("--max-words", help=NEXT_HELP["max_words"])] = 1500,
        effort: Annotated[str | None, typer.Option("--effort", help=NEXT_HELP["effort"])] = None,
        max_tokens: Annotated[
            int | None, typer.Option("--max-tokens", help=NEXT_HELP["max_tokens"])
        ] = None,
        temperature: Annotated[
            float | None, typer.Option("--temperature", help=NEXT_HELP["temperature"])
        ] = None,
    ) -> None:
        _next(
            case,
            model,
            pick,
            interview,
            rounds,
            max_questions,
            depth,
            counterclaims,
            evidence_demands,
            max_words,
            effort,
            max_tokens,
            temperature,
            expect=expect,
        )

    command.__name__ = name


_register_next(
    "next",
    None,
    False,
    "Write the next filing in a case's thread: the Claim if nothing has been filed, otherwise "
    "the reply due from whichever side is up. Starting a case of your own, the agent first asks "
    "you about it.",
)
_register_next("start", "start", True, "Write the Claim for a case with no filings yet.")
_register_next("reply", "reply", True, "Write the reply due to what is already in the thread.")


# ----------------------------------------------------------------------- simulate


def _simulator_client(settings: Settings) -> DecisionLayerClient:
    if not settings.decisionlayer_api_key:
        raise ConfigError(
            "Simulations need DECISIONLAYER_API_KEY (a production key from /settings/api-keys)."
        )
    if settings.decisionlayer_api_key.startswith(TEST_KEY_PREFIX):
        raise ConfigError(
            "DECISIONLAYER_API_KEY is a test key; the simulator requires a production key "
            "(it returns 403 otherwise)."
        )
    return DecisionLayerClient(settings.decisionlayer_api_key, settings.decisionlayer_base_url)


@app.command("simulate")
@handle_errors
def simulate(
    case: Annotated[str, typer.Argument(help="Case key under the cases directory, or a path.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for both sides. Default: CASEAGENT_MODEL."),
    ] = None,
    claimant_model: Annotated[
        str | None, typer.Option("--claimant-model", help="Model writing for the claimant.")
    ] = None,
    respondent_model: Annotated[
        str | None, typer.Option("--respondent-model", help="Model writing for the respondent.")
    ] = None,
    pick: Annotated[
        bool, typer.Option("--pick", "-p", help="Choose the models interactively.")
    ] = False,
    through: Annotated[
        str | None,
        typer.Option(
            "--through", help="Stop after this phase (e.g. 'answer' for a two-phase case)."
        ),
    ] = None,
    depth: Annotated[Depth, typer.Option("--depth", help=NEXT_HELP["depth"])] = Depth.THOROUGH,
    counterclaims: Annotated[
        CounterclaimPolicy, typer.Option("--counterclaims", help=NEXT_HELP["counterclaims"])
    ] = CounterclaimPolicy.AUTO,
    evidence_demands: Annotated[
        bool,
        typer.Option(
            "--evidence-demands/--no-evidence-demands",
            help="Let filings demand documentation (folded into the text for the simulator).",
        ),
    ] = False,
    max_words: Annotated[int, typer.Option("--max-words", help=NEXT_HELP["max_words"])] = 1500,
    effort: Annotated[str | None, typer.Option("--effort", help=NEXT_HELP["effort"])] = None,
    max_tokens: Annotated[
        int | None, typer.Option("--max-tokens", help=NEXT_HELP["max_tokens"])
    ] = None,
    temperature: Annotated[
        float | None, typer.Option("--temperature", help=NEXT_HELP["temperature"])
    ] = None,
    label: Annotated[str | None, typer.Option("--label", help="Suffix for the run id.")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Write the filings but do not submit; `caseagent submit` can later."
        ),
    ] = False,
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the award.")
    ] = 900.0,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmations.")] = False,
) -> None:
    """Have the agent play both sides from where the thread stands, then simulate the award.

    The case folder is left as it is; the run gets its own copy of the thread. Each
    phase is written against the filings before it, by an agent for that side that
    sees nothing of the other side's private work.
    """
    settings = Settings.load()
    bundle = resolve_case(case, settings.cases_dir)
    bundle = _ensure_essentials(bundle)
    case_file = bundle.case
    case_store = ThreadStore(bundle.root)
    stop_at = Stage.parse(through) if through else None
    stages = remaining_stages(case_store, stop_at)

    default_model = model or settings.model
    models = {
        Role.CLAIMANT: claimant_model or default_model,
        Role.RESPONDENT: respondent_model or default_model,
    }
    roles_needed = {stage.role for stage in stages}
    models = {role: spec for role, spec in models.items() if role in roles_needed}
    if not dry_run and any(is_fake(spec) for spec in models.values()):
        raise ConfigError(
            "The fake model's placeholder text is not an argument; use --dry-run with it."
        )

    catalog = load_catalog(settings.cache_dir) if models else None
    if pick:
        for role in list(models):
            models[role] = pick_model(
                catalog, models[role], f"{role.value} ({case_file.party(role).name})"
            )

    console.print(
        thread_table(
            case_file,
            case_store.entries(),
            {stage: models[stage.role] for stage in stages},
            title=f"Simulation plan: {case_file.title}",
        )
    )
    client: DecisionLayerClient | None = None
    if not dry_run:
        client = _simulator_client(settings)
        if not stages and not case_store.entries():
            raise ConfigError("There is nothing to simulate.")
        if not confirm(
            "Write the remaining filings and submit the case to the DecisionLayer simulator? "
            "(This uses one simulation.)",
            yes,
        ):
            raise typer.Exit(code=1)

    options = ComposeOptions(
        depth=depth,
        counterclaims=counterclaims,
        evidence_demands=evidence_demands,
        max_words=max_words,
    )
    record, store = start_run(
        bundle, settings.runs_dir, models, options, stop_at, label or _auto_label(models, stop_at)
    )
    if stages:
        model_options = _model_options(settings, effort, max_tokens, temperature)
        session_id = f"caseagent-{record.run_id}"
        progress = StageProgress(_party_names(case_file), models)
        agents = {}
        labels = {}
        for role, spec in models.items():
            built = _build(spec, settings, model_options, catalog, session_id)
            agents[role] = CaseAgent(built.client, options, on_event=progress.on_event)
            labels[role] = built.label
        progress.models = labels
        with progress:
            continue_thread(bundle, store, agents, labels, stop_at, on_stage=progress.on_stage)
    finish_run(record, bundle, store)

    entries = store.entries()
    console.print()
    console.print(filings_table(case_file, entries))
    for entry in entries:
        for warning in entry.warnings:
            console.print(f"[yellow]Warning ({entry.stage.title}):[/yellow] {warning}")
    if record.usage.calls:
        console.print(f"Model usage: {usage_line(store.usage())}")
    console.print(f"Saved run [bold]{record.run_id}[/bold] → {store.root}/transcript.md")

    if dry_run or client is None:
        console.print(f"[dim]Submit it with `caseagent submit {record.run_id}`.[/dim]")
        return
    _submit_run(client, bundle, store.root, timeout, yes=True)


def _auto_label(models: dict[Role, str], through: Stage | None) -> str:
    parts = []
    for role in Role:
        if role in models:
            parts.append(f"{role.value[0]}-{models[role].rsplit('/', 1)[-1]}")
    if through is not None and through is not STAGES[-1]:
        parts.append(f"through-{through.value}")
    return "-".join(parts) or "as-filed"


@app.command("submit")
@handle_errors
def submit(
    run_ref: Annotated[
        str, typer.Argument(metavar="RUN", help="Run id, run directory, or 'latest'.")
    ],
    timeout: Annotated[
        float, typer.Option("--timeout", help="Seconds to wait for the award.")
    ] = 900.0,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmations.")] = False,
) -> None:
    """Submit a run's filings to the DecisionLayer simulator and save the award."""
    settings = Settings.load()
    run_dir = resolve_run(run_ref, settings.runs_dir)
    record = load_run(run_dir)
    if any(is_fake(spec) for spec in record.models.values()):
        raise ConfigError("This run used the fake model; its text is not an argument.")
    bundle = resolve_case(record.case_dir, settings.cases_dir)
    client = _simulator_client(settings)
    console.print(f"Run {record.run_id}: {record.case_title} ({record.describe_sides()})")
    _submit_run(client, bundle, run_dir, timeout, yes)


def _submit_run(
    client: DecisionLayerClient, bundle: CaseBundle, run_dir: Path, timeout: float, yes: bool
) -> None:
    existing = load_simulation(run_dir)
    if existing is not None and not existing.finished:
        console.print(f"Resuming simulation {existing.simulation_id} (status: {existing.status}).")
        _await_and_save(client, existing, run_dir, timeout)
        return
    if existing is not None:
        console.print(
            f"This run was already simulated ({existing.simulation_id}, {existing.status}); "
            "submitting it again."
        )
    request = build_simulation_request(bundle, ThreadStore(run_dir).thread())
    for warning in request.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    _print_request_summary(request)
    if not confirm("Submit this simulation to DecisionLayer?", yes):
        raise typer.Exit(code=1)
    idempotency_key = str(uuid.uuid4())
    with console.status("Submitting to the simulator..."):
        sim_record = submit_simulation(client, request, idempotency_key)
    save_simulation(run_dir, sim_record)
    console.print(
        f"[green]✓[/green] Queued simulation [bold]{sim_record.simulation_id}[/bold] "
        f"(status: {sim_record.status})"
    )
    if sim_record.page_url:
        console.print(f"  Page (sign in to view): {sim_record.page_url}")
    _await_and_save(client, sim_record, run_dir, timeout)


def _await_and_save(
    client: DecisionLayerClient, sim_record: SimulationRecord, run_dir: Path, timeout: float
) -> None:
    started = datetime.now()

    def on_poll(simulation: Simulation) -> None:
        elapsed = (datetime.now() - started).total_seconds()
        status.update(f"Simulation {simulation.id}: {simulation.status} · {elapsed:.0f}s")

    with console.status("Waiting for the award...") as status:
        try:
            result = await_award(client, sim_record, timeout=timeout, on_poll=on_poll)
        finally:
            save_simulation(run_dir, sim_record)
    save_simulation(run_dir, sim_record, award_text=result.text)
    _print_award(result.text, sim_record)
    console.print(f"Saved award → {run_dir}/award.txt")


def _print_request_summary(request: SimulationRequest) -> None:
    fields = request.fields
    console.print("Simulation request:")
    for name in (
        "plaintiff_argument",
        "respondent_argument",
        "plaintiff_rebuttal",
        "respondent_rebuttal",
    ):
        if name in fields:
            console.print(f"  {name}: {len(fields[name].split()):,} words")
    contract = (
        "text"
        if "governing_contract" in fields
        else "file"
        if any(a.field == "contract_file" for a in request.attachments)
        else "none"
    )
    documents = [a for a in request.attachments if a.field != "contract_file"]
    console.print(f"  contract: {contract} · documents: {len(documents)}")


def _print_award(text: str, sim_record: SimulationRecord) -> None:
    section = sim_record.award_section or extract_award_section(text)
    console.print()
    console.rule("Award")
    if section:
        console.print(section)
        console.print()
        console.print("[dim]Full text in award.txt.[/dim]")
    else:
        console.print(text)
    if sim_record.pdf_url:
        console.print(f"PDF: {sim_record.pdf_url}")
    if sim_record.page_url:
        console.print(f"Page: {sim_record.page_url}")


@sim_app.command("list")
@handle_errors
def sim_list(limit: Annotated[int, typer.Option("--limit", "-n")] = 20) -> None:
    """List simulations this key owns, newest first."""
    settings = Settings.load()
    client = _simulator_client(settings)
    simulations, _cursor = client.list_simulations(limit=limit)
    table = Table(title="Simulations")
    table.add_column("Id")
    table.add_column("Status")
    table.add_column("Created")
    table.add_column("Result")
    for simulation in simulations:
        table.add_row(
            simulation.id,
            simulation.status,
            simulation.created_at.strftime("%Y-%m-%d %H:%M") if simulation.created_at else "",
            simulation.result_url or "",
        )
    console.print(table)


@sim_app.command("status")
@handle_errors
def sim_status(simulation_id: Annotated[str, typer.Argument()]) -> None:
    """Show a simulation's status."""
    settings = Settings.load()
    client = _simulator_client(settings)
    simulation = client.get_simulation(simulation_id)
    console.print_json(simulation.model_dump_json(exclude_none=True))


@sim_app.command("result")
@handle_errors
def sim_result(
    simulation_id: Annotated[str, typer.Argument()],
    run_ref: Annotated[
        str | None, typer.Option("--run", help="Also save the award into this run directory.")
    ] = None,
) -> None:
    """Fetch a finished simulation's award text (and PDF URL)."""
    settings = Settings.load()
    client = _simulator_client(settings)
    simulation = client.get_simulation(simulation_id)
    if simulation.status != "ready":
        console.print(
            f"Simulation {simulation_id} is {simulation.status}; the award is not public yet."
        )
        if simulation.message:
            console.print(simulation.message)
        raise typer.Exit(code=1)
    result = client.get_simulation_result(simulation_id)
    if run_ref:
        run_dir = resolve_run(run_ref, settings.runs_dir)
        sim_record = load_simulation(run_dir) or SimulationRecord(
            simulation_id=simulation_id,
            status=simulation.status,
            base_url=client.base_url,
            idempotency_key="",
            submitted_at=simulation.created_at or datetime.now(),
            page_url=simulation.page_url,
        )
        sim_record.update_from(simulation)
        sim_record.update_result(result)
        save_simulation(run_dir, sim_record, award_text=result.text)
        _print_award(result.text, sim_record)
        console.print(f"Saved award → {run_dir}/award.txt")
        return
    console.print(result.text)
    if result.pdf_url:
        console.print(f"\nPDF: {result.pdf_url}")


# --------------------------------------------------------------------------- runs


@app.command("runs")
@handle_errors
def runs(limit: Annotated[int, typer.Option("--limit", "-n")] = 30) -> None:
    """List simulation runs with their outcomes."""
    settings = Settings.load()
    run_dirs = list_run_dirs(settings.runs_dir)
    if not run_dirs:
        console.print(f"No runs in {settings.runs_dir}.")
        return
    table = Table(title=f"Runs in {settings.runs_dir}")
    table.add_column("Run")
    table.add_column("Case")
    table.add_column("Sides")
    table.add_column("Cost", justify="right")
    table.add_column("Simulation")
    table.add_column("Award")
    for run_dir in run_dirs[-limit:]:
        record = load_run(run_dir)
        sim = load_simulation(run_dir)
        award = (sim.award_section or "").split("\n")[0] if sim else ""
        table.add_row(
            record.run_id,
            record.case_key,
            record.describe_sides(),
            money(record.usage.cost_usd),
            f"{sim.status}" if sim else "-",
            award[:100] + ("…" if len(award) > 100 else ""),
        )
    console.print(table)


@app.command("show")
@handle_errors
def show(
    run_ref: Annotated[
        str, typer.Argument(metavar="RUN", help="Run id, run directory, or 'latest'.")
    ],
    award_only: Annotated[bool, typer.Option("--award", help="Print only the award.")] = False,
) -> None:
    """Print a run's transcript and, if simulated, its award."""
    settings = Settings.load()
    run_dir = resolve_run(run_ref, settings.runs_dir)
    record = load_run(run_dir)
    sim = load_simulation(run_dir)
    if not award_only:
        bundle = resolve_case(record.case_dir, settings.cases_dir)
        console.print(render_transcript(bundle, ThreadStore(run_dir).entries(), record.run_id))
        console.print(
            f"Model usage: {record.usage.calls} calls · {tokens(record.usage.prompt_tokens)} in · "
            f"{tokens(record.usage.completion_tokens)} out · {money(record.usage.cost_usd)}"
        )
    if sim is None:
        console.print("[dim]Not simulated yet.[/dim]")
        return
    award_path = run_dir / "award.txt"
    if award_path.is_file():
        _print_award(award_path.read_text(encoding="utf-8"), sim)
    else:
        console.print(
            f"Simulation {sim.simulation_id}: {sim.status}"
            + (f" ({sim.message})" if sim.message else "")
        )


# --------------------------------------------------------------------------- live


def _live_client(settings: Settings, key_env: str) -> DecisionLayerClient:
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise ConfigError(f"{key_env} is not set.")
    return DecisionLayerClient(api_key, settings.decisionlayer_base_url)


@live_app.command("status")
@handle_errors
def live_status(
    case_id: Annotated[str, typer.Argument(help="Case id (case_...).")],
    key_env: Annotated[
        str, typer.Option("--key-env", help="Environment variable holding this party's API key.")
    ] = "DECISIONLAYER_API_KEY",
) -> None:
    """Show a case's status, whose turn it is, and the response thread."""
    settings = Settings.load()
    client = _live_client(settings, key_env)
    case = client.get_case(case_id)
    console.print(
        f"[bold]{case.id}[/bold] · status {case.status} · you are the {case.role}"
        + (" · [yellow]test case[/yellow]" if case.test else "")
    )
    console.print(
        f"Turn: {case.current_turn or '-'} · next round: {case.next_round or '-'} · "
        f"action required: {case.action_required} ({case.next_action or 'none'})"
    )
    if case.action_url:
        console.print(f"Action URL: {case.action_url}")
    console.print(f"Accepted fields: {', '.join(case.accepted_fields) or '-'}")
    console.print(f"Demand: ${case.financial_demand_usd}")
    console.print()
    console.print("[bold]Claim form[/bold]")
    console.print(case.question_for_arbitration)
    for stage, filing in live_api.thread_from_api(case, client.list_responses(case_id)).items():
        console.print()
        console.print(
            f"[bold]{stage.title}[/bold] ({stage.role.value}) · {filing.word_count} words"
        )
        console.print(filing.argument[:600] + ("…" if len(filing.argument) > 600 else ""))


@live_app.command("respond")
@handle_errors
def live_respond(
    case_id: Annotated[str, typer.Argument(help="Case id (case_...).")],
    materials: Annotated[
        str,
        typer.Option(
            "--materials", help="Local case folder with the contract and this party's exhibits."
        ),
    ],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    key_env: Annotated[
        str, typer.Option("--key-env", help="Environment variable holding this party's API key.")
    ] = "DECISIONLAYER_API_KEY",
    interview: Annotated[
        bool,
        typer.Option("--interview/--no-interview", help="Ask you questions about the case first."),
    ] = False,
    depth: Annotated[Depth, typer.Option("--depth")] = Depth.THOROUGH,
    counterclaims: Annotated[
        CounterclaimPolicy, typer.Option("--counterclaims")
    ] = CounterclaimPolicy.AUTO,
    evidence_demands: Annotated[
        bool, typer.Option("--evidence-demands/--no-evidence-demands")
    ] = True,
    max_words: Annotated[int, typer.Option("--max-words")] = 1500,
    submit: Annotated[
        bool,
        typer.Option(
            "--submit",
            help="Submit the filing after you review and affirm it. Without this, only a draft is saved.",
        ),
    ] = False,
    allow_production: Annotated[
        bool, typer.Option("--allow-production", help="Allow submitting on a real (non-test) case.")
    ] = False,
) -> None:
    """Draft this party's next round on a live case from the API thread plus local materials.

    The case's status, turn, and accepted fields come from the API; the contract and
    this party's exhibits come from the local case folder. Nothing is submitted
    unless --submit is given and you affirm the filing.
    """
    settings = Settings.load()
    client = _live_client(settings, key_env)
    case = client.get_case(case_id)
    stage = live_api.stage_to_file(case)
    thread = live_api.thread_from_api(case, client.list_responses(case_id))

    bundle = resolve_case(materials, settings.cases_dir)
    # The API is authoritative for the claim form; the folder supplies the rest.
    case_file = CaseFile.model_validate(
        {
            **bundle.case.model_dump(),
            "question_for_arbitration": case.question_for_arbitration
            or bundle.case.question_for_arbitration,
            "financial_demand_usd": case.financial_demand_usd,
            "other_relief": case.other_relief,
        }
    )
    bundle = CaseBundle(case=case_file, root=bundle.root)
    console.print(
        f"Case {case.id}: round {case.next_round} ({stage.title}) is open for the {case.role}"
        + (" · [yellow]test case[/yellow]" if case.test else " · [red]live case[/red]")
    )
    if submit and not case.test and not allow_production:
        raise ConfigError("This is a real case. Re-run with --allow-production to submit on it.")

    catalog = load_catalog(settings.cache_dir)
    built = _build(
        model or settings.model, settings, _model_options(settings), catalog, f"caseagent-{case.id}"
    )
    options = ComposeOptions(
        depth=depth,
        counterclaims=counterclaims,
        evidence_demands=evidence_demands,
        max_words=max_words,
    )
    progress = StageProgress(_party_names(case_file), {stage.role: built.label})
    agent = CaseAgent(built.client, options, on_event=progress.on_event)
    if interview:
        if not interactive():
            raise ConfigError("--interview needs a terminal to ask questions in.")
        bundle = _interview(agent, bundle, stage, thread, rounds=2, max_questions=8)
    view = build_stage_view(bundle, stage, thread)
    with progress:
        progress.on_stage(stage, "start")
        result = agent.compose(view)
        progress.on_stage(stage, "end")
    entry = ThreadEntry.from_result(result, built.label)

    draft_dir = (
        settings.runs_dir / f"live-{case.id}-r{case.next_round}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    ThreadStore(draft_dir).filings_dir.mkdir(parents=True)
    (draft_dir / "filing.json").write_text(
        entry.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8"
    )
    (draft_dir / "argument.txt").write_text(result.filing.argument + "\n", encoding="utf-8")
    console.print()
    print_filing(entry, case_file.party(stage.role).name)
    console.print(f"Saved draft → {draft_dir}")
    if not submit:
        console.print("[dim]Review it, then re-run with --submit to file it.[/dim]")
        return

    affirmed = interactive() and typer.confirm(
        f"You are about to file this {stage.title} on case {case.id} as the {case.role}. "
        "Do you affirm that the information provided is accurate and complete?",
        default=False,
    )
    fields = live_api.response_form(result.filing, case.accepted_fields, affirmed=affirmed)
    attachments = [
        Attachment("evidence", path)
        for path in bundle.evidence_paths(stage)
        if "evidence" in case.accepted_fields
    ]
    with console.status("Submitting..."):
        created = client.create_response(case_id, fields, attachments)
    console.print(
        f"[green]✓[/green] Round {created.response.round} submitted at {created.response.submitted_at}."
    )
    updated = created.case
    console.print(
        f"Case status: {updated.status} · turn: {updated.current_turn or '-'} · "
        f"next action: {updated.next_action or 'none'}"
        + (f" → {updated.action_url}" if updated.action_url else "")
    )


# ------------------------------------------------------------------------- doctor


@app.command("doctor")
@handle_errors
def doctor() -> None:
    """Check configuration: keys, API reachability, model catalog, directories."""
    settings = Settings.load()
    ok, warn, bad = "[green]✓[/green]", "[yellow]![/yellow]", "[red]✗[/red]"
    console.print(
        f"{ok if settings.dotenv_path else warn} .env: "
        f"{settings.dotenv_path or 'not found (using environment only)'}"
    )

    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        try:
            response = httpx.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                timeout=15.0,
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                usage = data.get("usage")
                limit = data.get("limit")
                detail = f"label {data.get('label', '?')}"
                if usage is not None:
                    detail += f", used {money(usage)}" + (f" of {money(limit)}" if limit else "")
                console.print(f"{ok} OpenRouter key valid ({detail})")
            else:
                console.print(f"{bad} OpenRouter rejected the key ({response.status_code})")
        except httpx.HTTPError as exc:
            console.print(f"{warn} OpenRouter unreachable: {exc}")
    else:
        console.print(f"{warn} OPENROUTER_API_KEY not set (needed for OpenRouter models)")

    catalog = load_catalog(settings.cache_dir)
    if catalog:
        console.print(
            f"{ok} Model catalog: {len(catalog.models)} models"
            + (" (stale)" if catalog.stale else "")
        )
        provider, model_id = split_model_spec(settings.model)
        if provider == "openrouter" and catalog.get(model_id) is None:
            console.print(f"{bad} Default model {settings.model} is not in the OpenRouter catalog")
        else:
            console.print(f"{ok} Default model: {settings.model}")
    else:
        console.print(f"{warn} Model catalog unavailable (offline?)")

    if settings.decisionlayer_api_key:
        try:
            client = DecisionLayerClient(
                settings.decisionlayer_api_key, settings.decisionlayer_base_url
            )
            me = client.me()
            kind = (
                "test key: simulations will be refused" if client.is_test_key else "production key"
            )
            console.print(
                f"{warn if client.is_test_key else ok} DecisionLayer key for {me.email} ({kind}) "
                f"at {settings.decisionlayer_base_url}"
            )
        except DecisionLayerError as exc:
            console.print(f"{bad} DecisionLayer: {exc.message}")
    else:
        console.print(f"{warn} DECISIONLAYER_API_KEY not set (needed to simulate)")

    cases = list_cases(settings.cases_dir) if settings.cases_dir.is_dir() else []
    console.print(
        f"{ok if cases else warn} Cases: {len(cases)} in {settings.cases_dir}"
        + ("" if cases else " (run `caseagent samples fetch --all`)")
    )
    console.print(f"{ok} Runs: {len(list_run_dirs(settings.runs_dir))} in {settings.runs_dir}")


if __name__ == "__main__":
    app()
