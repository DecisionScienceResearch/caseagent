"""A guided first run: ``caseagent tour``.

Walks through the README's Getting Started section one step at a time. Each step
explains what is about to happen, shows the exact command, and runs it only on a
yes. Steps that need a key the user does not have are skipped with a note, and
nothing that costs money or a simulation runs unasked.

The commands are the real CLI commands, run in-process through ``run``; the tour
never re-implements them, so it cannot drift from what the commands actually do.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.prompt import Prompt

from .core.runs import list_run_dirs, load_run
from .decisionlayer.client import TEST_KEY_PREFIX
from .llm.factory import is_fake
from .settings import Settings
from .ui import console

Runner = Callable[[list[str]], int]
"""Runs ``caseagent ARGS...`` and returns its exit code (130 if interrupted)."""


def say(*objects: object, **kwargs: object) -> None:
    """Print prose without Rich's automatic highlighting of numbers, paths and the like."""
    console.print(*objects, highlight=False, **kwargs)  # type: ignore[arg-type]


OPENROUTER = "openrouter"
DECISIONLAYER = "decisionlayer"
DEFAULT_CASE = "saas-outage"


@dataclass(frozen=True)
class Command:
    args: tuple[str, ...]
    why: str
    after: str = ""
    needs: str | None = None
    spends: str | None = None
    check: Callable[[], str | None] | None = None
    """Returns a reason to skip the command, or None if it can run."""

    @property
    def text(self) -> str:
        return "caseagent " + " ".join(self.args)


@dataclass(frozen=True)
class Step:
    title: str
    intro: str
    commands: tuple[Command, ...] = ()
    outro: str = ""
    examples: tuple[str, ...] = ()
    setup: Callable[[bool], None] | None = None
    """Runs before the commands; the argument says whether the tour is print-only."""


# ------------------------------------------------------------------------ checks


def _missing_key(needs: str | None) -> str | None:
    """Why a command needing this key cannot run now, or None."""
    if needs is None:
        return None
    settings = Settings.load()  # reloaded each time: the user may have just edited .env
    if needs == OPENROUTER:
        if os.environ.get("OPENROUTER_API_KEY", "").strip():
            return None
        return (
            "OPENROUTER_API_KEY is not set. Put a key from https://openrouter.ai/keys in .env "
            "and run `caseagent tour --from 4` to pick up here."
        )
    if needs == DECISIONLAYER:
        key = settings.decisionlayer_api_key
        if not key:
            return (
                "DECISIONLAYER_API_KEY is not set. A production key (dvarb_...) from "
                f"{settings.decisionlayer_base_url}/settings/api-keys goes in .env; then "
                "`caseagent tour --from 6` picks up here."
            )
        if key.startswith(TEST_KEY_PREFIX):
            return (
                "DECISIONLAYER_API_KEY is a test key; the simulator needs a production key "
                "(dvarb_...). Test keys are for `caseagent live` cases."
            )
        return None
    raise ValueError(f"unknown requirement {needs!r}")


def _latest_run_is_fake() -> str | None:
    settings = Settings.load()
    runs = list_run_dirs(settings.runs_dir)
    if not runs:
        return "There is no run to submit yet (step 5 writes one)."
    record = load_run(runs[-1])
    if any(is_fake(spec) for spec in record.models.values()):
        return (
            f"The latest run ({record.run_id}) was written by the fake model, and placeholders "
            "are not an argument. Run step 5 first, then come back with `caseagent tour --from 6`."
        )
    return None


# ------------------------------------------------------------------------- steps


def _setup_env(print_only: bool) -> None:
    """Make sure a .env exists, and give the user a moment to fill it in."""
    env = Path(".env")
    example = Path(".env.example")
    if env.is_file():
        say(f"[green]✓[/green] {env.resolve()} exists.")
    elif print_only:
        say(
            "No .env here yet: `cp .env.example .env`, then put your keys in it."
            if example.is_file()
            else "No .env here yet: create one with OPENROUTER_API_KEY=... in it."
        )
    elif example.is_file():
        if _yes_no(f"No .env here yet. Create it from {example} now?", default=True):
            shutil.copyfile(example, env)
            say(f"[green]✓[/green] Created {env.resolve()}.")
    else:
        say(
            "No .env here yet. Create one with a line `OPENROUTER_API_KEY=...` "
            "(and `DECISIONLAYER_API_KEY=...` when you have a production key)."
        )
    if print_only:
        return
    say(
        "\nOpen .env in an editor and fill in [bold]OPENROUTER_API_KEY[/bold] now if you have "
        "one; the agent cannot write anything real without it. [bold]DECISIONLAYER_API_KEY[/bold] "
        "is only needed at step 6. Steps 2 and 3 need no keys at all, so it is fine to carry on "
        "without either."
    )
    Prompt.ask("Press Enter when ready", default="", show_default=False)


def build_steps(case: str) -> list[Step]:
    """The tour, following the README's Getting Started section."""
    return [
        Step(
            title="Keys and a health check",
            intro=(
                "Models come through OpenRouter by default, so one OPENROUTER_API_KEY covers "
                "Anthropic, OpenAI, Google, xAI, DeepSeek and others; the default model is "
                "anthropic/claude-opus-5.5. The DecisionLayer simulator needs its own key, a "
                "production one (dvarb_...), but not until step 6. Both live in a .env file in "
                "this folder."
            ),
            setup=_setup_env,
            commands=(
                Command(
                    ("doctor",),
                    "Checks the .env, both keys, the model catalog and the case and run folders. "
                    "Yellow warnings are fine for now: they say what is not set up yet.",
                ),
            ),
        ),
        Step(
            title="Fetch the sample disputes",
            intro=(
                "DecisionLayer publishes sample disputes (a claim form, the contract, exhibits). "
                "Fetching them needs no key. Only the dispute is kept as the case. The samples' "
                "own written arguments are saved to sample-arguments.md for you to compare "
                "against; the agent never reads or sends them."
            ),
            commands=(
                Command(
                    ("samples", "fetch", "--all"),
                    "Downloads every sample into cases/<key>/. Already-fetched cases are left "
                    "alone.",
                ),
                Command(("case", "list"), "Lists the local cases and how far each thread has got."),
            ),
            outro=(
                "Each case is a folder: case.json (the dispute), evidence/ (the exhibits) and, "
                "once something is filed, filings/ and work/."
            ),
        ),
        Step(
            title="See the mechanics for free",
            intro=(
                "fake:demo is an offline stand-in model that writes clearly labeled placeholders "
                "instead of arguments. It exercises the whole pipeline without a key: plan the "
                "four phases, write each one for its side, save the run, build the simulator "
                "payload. --dry-run means nothing is submitted (the fake model is refused by the "
                "simulator step anyway)."
            ),
            commands=(
                Command(
                    ("simulate", case, "-m", "fake:demo", "--depth", "quick", "--dry-run"),
                    f"Plays all four phases of {case} with the fake model, in a few seconds.",
                ),
                Command(
                    ("show", "latest"),
                    "Prints the run's transcript: every filing in order, with who wrote it.",
                ),
            ),
            outro=(
                "The run lives under runs/<id>/: transcript.md is the thread and work/ holds, "
                "for each phase, the exact prompt the agent saw. The case folder itself is "
                "untouched; a run gets its own copy of the thread."
            ),
        ),
        Step(
            title="One real filing",
            intro=(
                f"{case} is the smallest sample: a short terms of service, no exhibits, and a "
                "genuinely arguable issue. `next` writes whichever filing is due in the case's "
                "own thread, here the Claim, for the side whose turn it is. With --depth quick "
                "that is a single model call."
            ),
            commands=(
                Command(
                    ("next", case, "--depth", "quick"),
                    "Writes the Claim for the claimant, draft only.",
                    after=(
                        f"Read cases/{case}/filings/1-claim.txt, and the exact prompt the agent "
                        f"saw in cases/{case}/work/1-claim/materials.txt. The notes for the "
                        "principal (what to verify, what the record lacks) are shown above and "
                        "never filed."
                    ),
                    needs=OPENROUTER,
                    spends="one model call, well under a dollar on the default model",
                ),
                Command(
                    ("reply", case),
                    "Has the other side answer the Claim. Default depth is thorough: a private "
                    "brief, a draft, then a cite-check review. Check that the Answer engages the "
                    "Claim's actual points.",
                    needs=OPENROUTER,
                    spends="three model calls, around a dollar on the default model",
                ),
                Command(
                    ("case", "show", case, "--full"),
                    "Prints the thread as it now stands, both filings in full.",
                ),
                Command(
                    ("case", "reset", case, "--yes"),
                    "Rewinds the case to no filings so the next step starts clean. The run from "
                    "step 3 is unaffected.",
                ),
            ),
        ),
        Step(
            title="A full head-to-head, no simulator yet",
            intro=(
                "Now the agent plays all four phases for both sides. Each phase is written by a "
                "separate agent for its side, against the filings before it. It sees the claim "
                "form, the contract, the exhibits on the record and its own exhibits for the "
                "phase, never the other side's private brief, so the Reply really answers the "
                "Answer and the Response really answers the Reply.\n\n"
                "Default depth is thorough, so this is twelve model calls. On Claude Opus 5.5 at "
                "high effort expect a few dollars for this case, and several times that for the "
                "exhibit-heavy ones (invoice-balance, wedding-package, driver-deactivation). "
                "`--depth standard` roughly halves it; `--through answer` writes only the Claim "
                "and Answer. Skip this and run your own variant if you prefer."
            ),
            commands=(
                Command(
                    ("simulate", case, "--dry-run"),
                    "Writes all four filings into a new run and stops short of the simulator.",
                    after=(
                        "Read runs/<id>/transcript.md, and the notes-for-principal.md files under "
                        "its work/ folder: they say what each agent thought the record lacked."
                    ),
                    needs=OPENROUTER,
                    spends="twelve model calls, a few dollars on the default model",
                ),
            ),
        ),
        Step(
            title="Spend a simulation",
            intro=(
                "Submitting sends the dispute and the four filings to the DecisionLayer simulator "
                "and waits for the award. Each simulation counts against the monthly allowance, "
                "so the command shows what it is about to send and confirms first. Next time, "
                "leave off --dry-run and `simulate` does both halves in one go."
            ),
            commands=(
                Command(
                    ("submit", "latest"),
                    "Submits the run from step 5 and prints the AWARD paragraph when it arrives.",
                    after=(
                        "The full text is in runs/<id>/award.txt. If the wait times out, "
                        "`caseagent sim result ID --run RUN` picks the award up later."
                    ),
                    needs=DECISIONLAYER,
                    spends="one simulation from your DecisionLayer allowance",
                    check=_latest_run_is_fake,
                ),
                Command(("runs",), "Lists every run with its models, cost and outcome."),
            ),
        ),
        Step(
            title="Compare models",
            intro=(
                "Any model id from the catalog works with -m, or per side with --claimant-model "
                "and --respondent-model. Swap the sides too, since a case may simply favor one "
                "of them; `caseagent runs` then puts the outcomes side by side. Settings are "
                "fitted to each model from the catalog: reasoning effort is only sent where the "
                "model supports it and the output cap stays within its limits."
            ),
            commands=(
                Command(
                    ("models",),
                    "Browses the catalog with context size and prices. `caseagent models gpt` "
                    "filters by words; -p on next or simulate picks interactively.",
                ),
            ),
            examples=(
                f"caseagent simulate {case} --claimant-model anthropic/claude-opus-5.5 "
                "--respondent-model openai/gpt-6.1-sol",
                f"caseagent simulate {case} --claimant-model openai/gpt-6.1-sol "
                "--respondent-model anthropic/claude-opus-5.5",
                "caseagent runs",
            ),
        ),
        Step(
            title="Your own dispute",
            intro=(
                "`case init` makes an empty case folder and `next` starts it: with nothing filed "
                "it asks for the essentials (parties, the dispute as it should read on the claim "
                "form, the amount, the contract), then the agent interviews you about the facts "
                "and documents, you attach what you have, and it writes the Claim. Your answers "
                "are saved as your side's private statement and reused in later phases.\n\n"
                "If you are the respondent, record the claim you received with `case add-filing`, "
                "attach your exhibits with `case add-evidence`, and `next --interview` answers "
                "it. `next` always writes whichever filing is due, so one case can be driven from "
                "either side: write your Claim, paste in their Answer, write your Reply. "
                "`simulate` finishes both sides and gets an award."
            ),
            examples=(
                "caseagent case init cases/my-dispute",
                "caseagent next cases/my-dispute",
                "",
                'caseagent case init cases/their-claim --title "Deposit dispute"',
                "caseagent case add-filing cases/their-claim claim claim-we-received.txt",
                "caseagent case add-evidence cases/their-claim answer signed-addendum.pdf",
                "caseagent next cases/their-claim --interview",
            ),
            outro=(
                "That is the tour. `caseagent --help` lists every command, and the README covers "
                "how a filing is written and how to pick models. If a command fails, "
                "`CASEAGENT_DEBUG=1` in front of it prints the full traceback."
            ),
        ),
    ]


# ------------------------------------------------------------------------ runner


class TourQuit(Exception):
    """The user chose to stop (or stdin ran out)."""


def _yes_no(question: str, default: bool = True) -> bool:
    try:
        answer = Prompt.ask(question, choices=["y", "n"], default="y" if default else "n")
    except EOFError:
        raise TourQuit from None
    return answer == "y"


def _run_or_skip() -> str:
    """'y' to run, 'n' to skip, 'q' to quit."""
    try:
        return Prompt.ask("Run it?", choices=["y", "n", "q"], default="y")
    except EOFError:
        raise TourQuit from None


def _pause(prompt: str) -> None:
    try:
        answer = Prompt.ask(prompt, default="", show_default=False)
    except EOFError:
        raise TourQuit from None
    if answer.strip().lower() == "q":
        raise TourQuit


def _show_command(command: Command) -> None:
    say(f"\n  [bold cyan]$ {command.text}[/bold cyan]")
    say(f"  {command.why}")
    if command.spends:
        say(f"  [yellow]Spends:[/yellow] {command.spends}.")


def _run_command(command: Command, run: Runner, step_number: int) -> None:
    skip = _missing_key(command.needs) or (command.check() if command.check else None)
    if skip:
        say(f"  [yellow]Skipped:[/yellow] {skip}")
        return
    choice = _run_or_skip()
    if choice == "q":
        raise TourQuit
    if choice == "n":
        return
    say()
    code = run(list(command.args))
    say()
    if code == 130:
        raise TourQuit
    if code:
        say(
            f"[yellow]That did not succeed (exit {code}).[/yellow] Fix the cause and re-run "
            f"`caseagent tour --from {step_number}` to retry this step, or carry on."
        )
        if not _yes_no("Carry on with the tour?"):
            raise TourQuit
        return
    if command.after:
        say(f"[dim]{command.after}[/dim]")


def run_tour(steps: list[Step], run: Runner, start: int = 1, print_only: bool = False) -> None:
    """Walk the user through ``steps``, starting at step ``start`` (1-based)."""
    total = len(steps)
    if not 1 <= start <= total:
        raise ValueError(f"--from must be between 1 and {total}.")
    say(
        "[bold]Welcome to caseagent.[/bold] It writes arbitration filings with a language "
        "model, one turn at a time, and tests them on the DecisionLayer simulator. This tour "
        f"follows the README's Getting Started in {total} steps. Each step shows the exact "
        "command before running it, and nothing that costs money or a simulation runs without "
        "your yes."
    )
    if print_only:
        say(
            "[dim]Printing the walkthrough only; run `caseagent tour` in a terminal "
            "to follow along.[/dim]"
        )
    else:
        say(
            "[dim]Answer y to run a command, n to skip it, q to stop. Ctrl-C stops too. "
            "`caseagent tour --from N` resumes at step N.[/dim]"
        )
    current = start
    try:
        for number, step in enumerate(steps, start=1):
            if number < start:
                continue
            current = number
            console.rule(f"Step {number} of {total}: {step.title}")
            say(step.intro)
            if step.setup is not None:
                step.setup(print_only)
            for command in step.commands:
                _show_command(command)
                if not print_only:
                    _run_command(command, run, number)
            if step.examples:
                say("\n  [dim]To try later:[/dim]")
                for example in step.examples:
                    say(f"  [cyan]$ {example}[/cyan]" if example else "")
            if step.outro:
                say(f"\n{step.outro}")
            if number < total and not print_only:
                _pause("\nPress Enter for the next step (q to stop)")
    except TourQuit:
        say(
            f"\n[dim]Stopped at step {current}. `caseagent tour --from {current}` picks up "
            "there.[/dim]"
        )
        return
    console.rule()
    say("[green]✓[/green] Tour complete.")
