"""Run a played case through the DecisionLayer simulator.

``POST /api/v1/simulations`` takes every filing at once: one text field per
phase plus a repeatable document field for each, and the contract as text or
file. It has no separate counterclaim or documentation-demand fields, so those
parts of a filing are folded into the phase's text under plain headings.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core.casefile import CaseBundle
from ..core.models import STAGES, Filing, Stage, format_usd
from .client import (
    Attachment,
    DecisionLayerClient,
    Simulation,
    SimulationFailed,
    SimulationResult,
)

SIMULATION_FILENAME = "simulation.json"
AWARD_FILENAME = "award.txt"

# Simulator form fields for each phase: (argument text, documents).
STAGE_FIELDS: dict[Stage, tuple[str, str]] = {
    Stage.CLAIM: ("plaintiff_argument", "plaintiff_documents"),
    Stage.ANSWER: ("respondent_argument", "respondent_documents"),
    Stage.REPLY: ("plaintiff_rebuttal", "plaintiff_rebuttal_documents"),
    Stage.RESPONSE: ("respondent_rebuttal", "respondent_rebuttal_documents"),
}

ACCEPTED_SUFFIXES = {".pdf", ".docx", ".rtf", ".txt"}


class SimulationInputError(ValueError):
    """The played case cannot be expressed as a simulation request."""


@dataclass(frozen=True)
class SimulationRequest:
    fields: dict[str, str]
    attachments: tuple[Attachment, ...]
    warnings: tuple[str, ...] = ()

    def summary(self) -> dict[str, Any]:
        """What was sent, without repeating the full texts."""
        return {
            "fields": {
                name: value if len(value) <= 120 else f"<{len(value)} characters>"
                for name, value in self.fields.items()
            },
            "attachments": [
                {"field": attachment.field, "file": attachment.path.name}
                for attachment in self.attachments
            ],
        }


def render_for_simulator(filing: Filing) -> str:
    """One text per phase: the argument plus any structured parts under headings."""
    blocks = [filing.argument.strip()]
    if filing.counterclaim:
        counterclaim = filing.counterclaim
        lines = ["COUNTERCLAIM", "", counterclaim.argument.strip()]
        if counterclaim.amount_usd is not None:
            lines += [
                "",
                f"Amount demanded on the counterclaim: ${format_usd(counterclaim.amount_usd)}",
            ]
        if counterclaim.other_relief:
            lines += [
                "",
                f"Other relief demanded on the counterclaim: {counterclaim.other_relief.strip()}",
            ]
        blocks.append("\n".join(lines))
    if filing.evidence_response:
        blocks.append("RESPONSE TO DOCUMENTATION DEMANDS\n\n" + filing.evidence_response.strip())
    if filing.evidence_demands:
        blocks.append("DOCUMENTATION DEMANDED\n\n" + filing.evidence_demands.strip())
    return "\n\n".join(blocks)


def build_simulation_request(
    bundle: CaseBundle, thread: Mapping[Stage, Filing]
) -> SimulationRequest:
    """Map the case and its filings onto the simulator's form."""
    case = bundle.case
    missing = [stage.title for stage in (Stage.CLAIM, Stage.ANSWER) if stage not in thread]
    if missing:
        raise SimulationInputError(
            f"The simulator needs both a Claim and an Answer; missing: {', '.join(missing)}."
        )

    fields: dict[str, str] = {
        "question_for_arbitration": case.question_for_arbitration.strip(),
        "plaintiff_name": case.claimant.name,
        "respondent_name": case.respondent.name,
        "financial_demand_usd": format_usd(case.financial_demand_usd),
    }
    if case.other_relief and case.other_relief.strip():
        fields["other_relief"] = case.other_relief.strip()

    attachments: list[Attachment] = []
    warnings: list[str] = []
    if case.governing_contract and case.governing_contract.strip():
        fields["governing_contract"] = case.governing_contract.strip()
    elif bundle.contract_path is not None:
        attachments.append(Attachment("contract_file", bundle.contract_path))
    else:
        warnings.append("The case has no contract text or file; the simulator may reject it.")

    for stage in STAGES:
        filing = thread.get(stage)
        if filing is None:
            continue
        text_field, documents_field = STAGE_FIELDS[stage]
        fields[text_field] = render_for_simulator(filing)
        for path in bundle.evidence_paths(stage):
            if path.suffix.lower() not in ACCEPTED_SUFFIXES:
                warnings.append(
                    f"{path.name}: the simulator accepts .pdf, .docx, .rtf and .txt; "
                    "this file may be rejected."
                )
            attachments.append(Attachment(documents_field, path))

    return SimulationRequest(
        fields=fields, attachments=tuple(attachments), warnings=tuple(warnings)
    )


class SimulationRecord(BaseModel):
    """What this run sent to the simulator and what came back; saved as simulation.json."""

    simulation_id: str
    status: str
    base_url: str
    idempotency_key: str
    submitted_at: datetime
    page_url: str | None = None
    result_url: str | None = None
    pdf_url: str | None = None
    message: str | None = None
    published_at: datetime | None = None
    award_section: str | None = None
    request: dict[str, Any] | None = None

    @property
    def finished(self) -> bool:
        return self.status in ("ready", "failed")

    def update_from(self, simulation: Simulation) -> None:
        self.status = simulation.status
        self.page_url = simulation.page_url or self.page_url
        self.result_url = simulation.result_url or self.result_url
        self.message = simulation.message

    def update_result(self, result: SimulationResult) -> None:
        self.status = "ready"
        self.pdf_url = result.pdf_url
        self.page_url = result.page_url or self.page_url
        self.published_at = result.published_at
        self.award_section = extract_award_section(result.text)


def submit_simulation(
    client: DecisionLayerClient,
    request: SimulationRequest,
    idempotency_key: str,
) -> SimulationRecord:
    simulation = client.create_simulation(request.fields, request.attachments, idempotency_key)
    record = SimulationRecord(
        simulation_id=simulation.id,
        status=simulation.status,
        base_url=client.base_url,
        idempotency_key=idempotency_key,
        submitted_at=datetime.now(UTC),
        page_url=simulation.page_url,
        result_url=simulation.result_url,
        message=simulation.message,
        request=request.summary(),
    )
    return record


def await_award(
    client: DecisionLayerClient,
    record: SimulationRecord,
    timeout: float = 900.0,
    interval: float = 10.0,
    on_poll: Callable[[Simulation], None] | None = None,
) -> SimulationResult:
    """Wait for the award and fold the outcome into ``record``, including a failure."""
    try:
        simulation = client.wait_for_simulation(
            record.simulation_id, timeout=timeout, interval=interval, on_poll=on_poll
        )
    except SimulationFailed as exc:
        record.update_from(exc.simulation)
        raise
    record.update_from(simulation)
    result = client.get_simulation_result(record.simulation_id)
    record.update_result(result)
    return result


_AWARD_HEADING_RE = re.compile(r"^[ \t]*AWARD(?::[ \t]*(.*)|[ \t]*)$", re.MULTILINE)
_NEXT_HEADING_RE = re.compile(r"^[ \t]*(?:[A-Z][A-Z &'/()-]{2,}:|={5,}|-{5,})[ \t]*$", re.MULTILINE)


def extract_award_section(text: str) -> str | None:
    """The AWARD paragraph of an award text, if the text has one."""
    match = _AWARD_HEADING_RE.search(text)
    if not match:
        return None
    rest = text[match.end() :]
    stop = _NEXT_HEADING_RE.search(rest)
    body = rest[: stop.start()] if stop else rest
    section = ((match.group(1) or "").strip() + "\n" + body).strip()
    return section or None


def save_simulation(run_dir: Path, record: SimulationRecord, award_text: str | None = None) -> None:
    payload = record.model_dump(mode="json", exclude_none=True)
    (run_dir / SIMULATION_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if award_text:
        (run_dir / AWARD_FILENAME).write_text(award_text.rstrip() + "\n", encoding="utf-8")


def load_simulation(run_dir: Path) -> SimulationRecord | None:
    path = run_dir / SIMULATION_FILENAME
    if not path.is_file():
        return None
    return SimulationRecord.model_validate_json(path.read_text(encoding="utf-8"))
