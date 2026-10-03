"""Simulation runs.

A run directory is a thread (``filings/``, ``work/``) plus ``run.json`` with the
case it came from and the models that wrote for each side, and ``transcript.md``
for reading. The simulator adapter adds ``simulation.json`` and ``award.txt``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .casefile import CaseBundle
from .models import STAGES, Author, Role, Stage, display_usd
from .options import ComposeOptions
from .thread import ThreadEntry, ThreadStore, UsageRecord

RUN_FILENAME = "run.json"


class RunRecord(BaseModel):
    run_id: str
    created_at: datetime
    case_key: str
    case_title: str
    case_dir: str
    models: dict[Role, str] = Field(default_factory=dict)
    through: Stage | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    copied_filings: int = 0
    usage: UsageRecord = Field(default_factory=UsageRecord)

    def describe_sides(self) -> str:
        parts = []
        for role in Role:
            model = self.models.get(role)
            parts.append(f"{role.value}={model.rsplit('/', 1)[-1] if model else 'human'}")
        return " ".join(parts)


def new_run_id(case_key: str, label: str = "", now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    slug = _slug(label)
    return f"{stamp}-{case_key}" + (f"-{slug}" if slug else "")


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "-", text).strip("-").lower()[:60]


def start_run(
    bundle: CaseBundle,
    runs_dir: Path,
    models: dict[Role, str],
    options: ComposeOptions,
    through: Stage | None = None,
    label: str = "",
) -> tuple[RunRecord, ThreadStore]:
    """Create a run directory seeded with the case's filings so far."""
    run_id = new_run_id(bundle.case.key, label)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    store = ThreadStore(run_dir)
    copied = ThreadStore(bundle.root).copy_to(store)
    record = RunRecord(
        run_id=run_id,
        created_at=datetime.now(UTC),
        case_key=bundle.case.key,
        case_title=bundle.case.title,
        case_dir=str(bundle.root.resolve()),
        models=dict(models),
        through=through,
        options={
            "depth": options.depth.value,
            "counterclaims": options.counterclaims.value,
            "evidence_demands": options.evidence_demands,
            "max_words": options.max_words,
        },
        copied_filings=len(copied),
    )
    write_run(record, run_dir)
    return record, store


def finish_run(record: RunRecord, bundle: CaseBundle, store: ThreadStore) -> RunRecord:
    """Record usage and write the transcript once the thread is complete."""
    record.usage = UsageRecord.of(store.usage())
    write_run(record, store.root)
    (store.root / "transcript.md").write_text(
        render_transcript(bundle, store.entries(), record.run_id), encoding="utf-8"
    )
    return record


def write_run(record: RunRecord, run_dir: Path) -> None:
    payload = record.model_dump(mode="json", exclude_none=True)
    (run_dir / RUN_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load_run(run_dir: Path) -> RunRecord:
    path = run_dir / RUN_FILENAME if run_dir.is_dir() else run_dir
    if not path.is_file():
        raise FileNotFoundError(f"No {RUN_FILENAME} in {run_dir}")
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def resolve_run(ref: str, runs_dir: Path) -> Path:
    """Find a run directory by path, by id, or ``latest``."""
    if ref == "latest":
        runs = list_run_dirs(runs_dir)
        if not runs:
            raise FileNotFoundError(f"No runs in {runs_dir}")
        return runs[-1]
    for candidate in (Path(ref).expanduser(), runs_dir / ref):
        if (candidate / RUN_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(f"No run named {ref!r} (looked in {runs_dir})")


def list_run_dirs(runs_dir: Path) -> list[Path]:
    """Run directories, oldest first (run ids start with a timestamp)."""
    if not runs_dir.is_dir():
        return []
    return sorted(child for child in runs_dir.iterdir() if (child / RUN_FILENAME).is_file())


def render_transcript(bundle: CaseBundle, entries: list[ThreadEntry], heading: str = "") -> str:
    """A thread as one readable Markdown document."""
    case = bundle.case
    lines = [f"# {case.title}", ""]
    if heading:
        lines.append(f"- Run: `{heading}`")
    lines += [
        f"- Claimant: {case.claimant.name}",
        f"- Respondent: {case.respondent.name}",
        f"- Amount demanded: {display_usd(case.financial_demand_usd)}",
    ]
    if case.other_relief:
        lines.append(f"- Other relief demanded: {case.other_relief}")
    lines += ["", "## Claim form", "", case.question_for_arbitration.strip(), ""]

    by_stage = {entry.stage: entry for entry in entries}
    for stage in STAGES:
        entry = by_stage.get(stage)
        party = case.party(stage.role).name
        lines += [f"## {stage.title}: {party} ({stage.role.value})", ""]
        if entry is None:
            lines += ["_not filed_", ""]
            continue
        author = (
            f"written by the agent ({entry.model})"
            if entry.author is Author.AGENT
            else "provided by a person"
        )
        lines += [f"_{author}_", "", entry.filing.argument, ""]
        filing = entry.filing
        if filing.counterclaim:
            amount = filing.counterclaim.amount_usd
            heading_text = "### Counterclaim" + (f" ({display_usd(amount)})" if amount else "")
            lines += [heading_text, "", filing.counterclaim.argument, ""]
            if filing.counterclaim.other_relief:
                lines += [f"Other relief: {filing.counterclaim.other_relief}", ""]
        if filing.evidence_response:
            lines += ["### Response to documentation demands", "", filing.evidence_response, ""]
        if filing.evidence_demands:
            lines += ["### Documentation demanded", "", filing.evidence_demands, ""]
        if entry.notes_for_principal:
            lines += ["### Notes for the principal (not filed)", "", entry.notes_for_principal, ""]
        for warning in entry.warnings:
            lines += [f"> Warning: {warning}", ""]
    return "\n".join(lines).rstrip() + "\n"
