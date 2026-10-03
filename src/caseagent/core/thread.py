"""The thread: the filings made so far, stored beside the case.

A thread directory holds::

    filings/1-claim.json      ThreadEntry: the filing plus who wrote it and when
    filings/1-claim.txt       the argument text, as it would be filed
    work/1-claim/             the agent's prompts, brief, draft, review, notes

A case folder is a thread (the case as it stands); a simulation run copies that
thread and continues it, so the case folder itself is never changed by a run.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from .agent import ComposeResult
from .llm import Usage
from .models import STAGES, Author, Filing, Stage

FILINGS_DIRNAME = "filings"
WORK_DIRNAME = "work"


class ThreadError(Exception):
    pass


class UsageRecord(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None

    @classmethod
    def of(cls, usage: Usage) -> UsageRecord:
        return cls(
            calls=usage.calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cached_tokens=usage.cached_tokens,
            cost_usd=usage.cost_usd,
        )

    def to_usage(self) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cached_tokens=self.cached_tokens,
            cost_usd=self.cost_usd,
            calls=self.calls,
        )


class ThreadEntry(BaseModel):
    filing: Filing
    author: Author
    model: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes_for_principal: str | None = None
    review_notes: str | None = None
    warnings: list[str] = Field(default_factory=list)
    usage: UsageRecord | None = None
    seconds: float = 0.0

    @property
    def stage(self) -> Stage:
        return self.filing.stage

    @property
    def written_by(self) -> str:
        return f"agent ({self.model})" if self.author is Author.AGENT else "human"

    @classmethod
    def from_result(cls, result: ComposeResult, model: str) -> ThreadEntry:
        return cls(
            filing=result.filing,
            author=Author.AGENT,
            model=model,
            notes_for_principal=result.notes_for_principal,
            review_notes=result.review_notes,
            warnings=list(result.warnings),
            usage=UsageRecord.of(result.usage),
            seconds=result.seconds,
        )


def entry_prefix(stage: Stage) -> str:
    return f"{stage.index + 1}-{stage.value}"


class ThreadStore:
    """Reads and appends the filings in one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.filings_dir = root / FILINGS_DIRNAME
        self.work_dir = root / WORK_DIRNAME

    def entries(self) -> list[ThreadEntry]:
        """Filings in phase order. Phases must be contiguous from the Claim."""
        found: dict[Stage, ThreadEntry] = {}
        for stage in STAGES:
            path = self.filings_dir / f"{entry_prefix(stage)}.json"
            if path.is_file():
                try:
                    entry = ThreadEntry.model_validate_json(path.read_text(encoding="utf-8"))
                except ValueError as exc:
                    raise ThreadError(f"{path} is not a valid filing: {exc}") from exc
                if entry.stage is not stage:
                    raise ThreadError(f"{path} holds a {entry.stage.title}, not a {stage.title}")
                found[stage] = entry
        ordered = []
        for stage in STAGES:
            if stage not in found:
                later = [s.title for s in STAGES[stage.index + 1 :] if s in found]
                if later:
                    raise ThreadError(
                        f"{self.root} has a {', '.join(later)} but no {stage.title}; "
                        "filings must be contiguous from the Claim."
                    )
                break
            ordered.append(found[stage])
        return ordered

    def thread(self) -> dict[Stage, Filing]:
        return {entry.stage: entry.filing for entry in self.entries()}

    def next_stage(self) -> Stage | None:
        """The phase to file next, or None when the Response is in."""
        count = len(self.entries())
        return STAGES[count] if count < len(STAGES) else None

    def append(self, entry: ThreadEntry, result: ComposeResult | None = None) -> Path:
        expected = self.next_stage()
        if entry.stage is not expected:
            due = expected.title if expected else "nothing (the thread is complete)"
            raise ThreadError(f"The next filing due is {due}, not the {entry.stage.title}.")
        self.filings_dir.mkdir(parents=True, exist_ok=True)
        prefix = entry_prefix(entry.stage)
        path = self.filings_dir / f"{prefix}.json"
        path.write_text(entry.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        (self.filings_dir / f"{prefix}.txt").write_text(
            entry.filing.argument.rstrip() + "\n", encoding="utf-8"
        )
        if result is not None:
            self._write_work(self.work_dir / prefix, result)
        return path

    def remove_last(self) -> ThreadEntry | None:
        entries = self.entries()
        if not entries:
            return None
        last = entries[-1]
        prefix = entry_prefix(last.stage)
        for suffix in (".json", ".txt"):
            (self.filings_dir / f"{prefix}{suffix}").unlink(missing_ok=True)
        shutil.rmtree(self.work_dir / prefix, ignore_errors=True)
        return last

    def clear(self) -> int:
        count = len(self.entries())
        shutil.rmtree(self.filings_dir, ignore_errors=True)
        shutil.rmtree(self.work_dir, ignore_errors=True)
        return count

    def copy_to(self, other: ThreadStore) -> list[ThreadEntry]:
        """Start another thread with this one's filings (work files included)."""
        entries = self.entries()
        if entries:
            other.filings_dir.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                prefix = entry_prefix(entry.stage)
                for suffix in (".json", ".txt"):
                    source = self.filings_dir / f"{prefix}{suffix}"
                    if source.is_file():
                        shutil.copy2(source, other.filings_dir / source.name)
                work = self.work_dir / prefix
                if work.is_dir():
                    shutil.copytree(work, other.work_dir / prefix, dirs_exist_ok=True)
        return entries

    def usage(self) -> Usage:
        total = Usage()
        for entry in self.entries():
            if entry.usage is not None:
                total.add(entry.usage.to_usage())
        return total

    @staticmethod
    def _write_work(work_dir: Path, result: ComposeResult) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "system.txt").write_text(result.system + "\n", encoding="utf-8")
        (work_dir / "materials.txt").write_text(result.materials + "\n", encoding="utf-8")
        for number, step in enumerate(result.steps, start=1):
            (work_dir / f"{number}-{step.name}.task.txt").write_text(
                step.task + "\n", encoding="utf-8"
            )
            (work_dir / f"{number}-{step.name}.reply.txt").write_text(
                step.reply + "\n", encoding="utf-8"
            )
        if result.brief:
            (work_dir / "brief.md").write_text(result.brief + "\n", encoding="utf-8")
        if result.notes_for_principal:
            (work_dir / "notes-for-principal.md").write_text(
                result.notes_for_principal + "\n", encoding="utf-8"
            )
