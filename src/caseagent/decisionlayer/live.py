"""Map the agent's filings onto a real case's turns (``/api/v1/cases``).

The API tells each party what the case is waiting for: ``status``,
``current_turn``, ``next_round`` and ``accepted_fields``, and what the key
owner must do about it: ``next_action``. Rounds map onto the phases as
1 = Answer, 2 = Reply, 3 = Response; the Claim is the case itself. A party
whose ``next_action`` is ``sign_terms`` cannot file until the arbitration
terms are signed, so ``stage_to_file`` stops there before anything is written.
A round can be submitted once, and the party's affirmation that the filing is
accurate and complete is a human act, so ``response_form`` requires it to be
given explicitly.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..core.models import Counterclaim, DocumentKind, FiledDocument, Filing, Stage, format_usd
from .client import ArbitrationCase, CaseResponse, Decision
from .simulator import AWARD_FILENAME, extract_award_section

DECISION_FILENAME = "decision.json"

ROUND_STAGES: dict[int, Stage] = {1: Stage.ANSWER, 2: Stage.REPLY, 3: Stage.RESPONSE}
STAGE_ROUNDS: dict[Stage, int] = {stage: number for number, stage in ROUND_STAGES.items()}

# The case's ``next_action`` values that decide whether a filing can be made now.
NEXT_ACTION_RESPOND = "respond"
NEXT_ACTION_SIGN = "sign_terms"


class LiveCaseError(Exception):
    pass


class ActionRequired(LiveCaseError):
    """A step on the web stands between this party and its filing."""

    def __init__(self, case: ArbitrationCase, message: str) -> None:
        self.case = case
        super().__init__(message)


class SignatureRequired(ActionRequired):
    """The party must sign the arbitration terms first; ``POST /cases/{id}/sign`` has the URL."""


def thread_from_api(
    case: ArbitrationCase, responses: Collection[CaseResponse]
) -> dict[Stage, Filing]:
    """Rebuild the filings so far from the case and its response thread.

    Each round's ``evidence_files`` become the filing's ``documents``: the names
    and kinds of the files submitted with it. The API does not serve the files.
    """
    thread: dict[Stage, Filing] = {}
    if case.argument:
        thread[Stage.CLAIM] = Filing(stage=Stage.CLAIM, argument=case.argument)
    for response in sorted(responses, key=lambda item: item.round):
        stage = ROUND_STAGES.get(response.round)
        if stage is None:
            continue
        spec = stage.spec
        counterclaim = None
        if spec.allows_counterclaim and response.counterclaim_argument:
            counterclaim = Counterclaim(
                argument=response.counterclaim_argument,
                amount_usd=response.counterclaim_amount_usd,
                other_relief=response.counterclaim_other_relief,
            )
        thread[stage] = Filing(
            stage=stage,
            argument=response.argument or "",
            counterclaim=counterclaim,
            evidence_demands=response.evidence_demands if spec.allows_evidence_demands else None,
            evidence_response=response.evidence_response if spec.allows_evidence_response else None,
            documents=[
                FiledDocument(name=file.name, kind=_document_kind(file.kind))
                for file in response.evidence_files
                if file.name
            ],
        )
    return thread


def _document_kind(kind: str | None) -> DocumentKind:
    try:
        return DocumentKind(kind or "")
    except ValueError:
        return DocumentKind.SUPPORTING


def response_due(responses: Collection[CaseResponse]) -> datetime | None:
    """When the open round is due: the latest submitted round carries the deadline."""
    latest = max(responses, key=lambda item: item.round, default=None)
    return latest.due_at if latest is not None else None


def stage_to_file(case: ArbitrationCase) -> Stage:
    """The phase this key must file now, or an explanation of why it cannot.

    Raises ``SignatureRequired`` when the case waits on this party's signature
    rather than its filing, and ``ActionRequired`` for any other web step, so
    nothing is composed that the API would refuse with a 409.
    """
    if case.status != "awaiting_response" or case.next_round is None:
        pending = f"next action: {case.next_action}" if case.next_action else "no action pending"
        where = f" at {case.action_url}" if case.action_url else ""
        raise LiveCaseError(
            f"Case {case.id} is not awaiting a response (status: {case.status}; {pending}{where})."
        )
    if case.current_turn != case.role:
        raise LiveCaseError(
            f"It is the {case.current_turn}'s turn on case {case.id}; this key is the {case.role}."
        )
    if case.next_action and case.next_action != NEXT_ACTION_RESPOND:
        where = f" at {case.action_url}" if case.action_url else ""
        if case.next_action == NEXT_ACTION_SIGN:
            raise SignatureRequired(
                case,
                f"The {case.role} must sign the arbitration terms before filing on case "
                f"{case.id}{where}.",
            )
        raise ActionRequired(
            case,
            f"Case {case.id} needs '{case.next_action}' from the {case.role} before it can "
            f"file{where}.",
        )
    stage = ROUND_STAGES.get(case.next_round)
    if stage is None:
        raise LiveCaseError(f"Case {case.id} reports an unexpected round {case.next_round}.")
    return stage


def response_form(
    filing: Filing, accepted_fields: Collection[str], affirmed: bool
) -> dict[str, str]:
    """Multipart text fields for ``POST /cases/{id}/responses``.

    Only fields the case lists in ``accepted_fields`` are sent; a structured part
    the round cannot carry is an error rather than something silently dropped.
    """
    if not affirmed:
        raise LiveCaseError(
            "The affirmation that the filing is accurate and complete must be given by the "
            "party; refusing to submit without it."
        )
    fields = {"argument": filing.argument.strip(), "affirmation": "true"}
    optional: dict[str, str | None] = {
        "evidence_demands": filing.evidence_demands,
        "evidence_response": filing.evidence_response,
    }
    if filing.counterclaim:
        optional["counterclaim_argument"] = filing.counterclaim.argument
        optional["counterclaim_amount_usd"] = (
            format_usd(filing.counterclaim.amount_usd)
            if filing.counterclaim.amount_usd is not None
            else None
        )
        optional["counterclaim_other_relief"] = filing.counterclaim.other_relief
    for name, value in optional.items():
        if not value or not value.strip():
            continue
        if accepted_fields and name not in accepted_fields:
            raise LiveCaseError(
                f"The case does not accept {name!r} in round {STAGE_ROUNDS[filing.stage]} "
                f"(accepted: {', '.join(sorted(accepted_fields))})."
            )
        fields[name] = value.strip()
    return fields


class DecisionRecord(BaseModel):
    """A published award fetched from a live case; saved as decision.json beside award.txt."""

    case_id: str
    decision_id: str
    base_url: str
    fetched_at: datetime
    published_at: datetime | None = None
    pdf_url: str | None = None
    award_section: str | None = None


def save_decision(directory: Path, decision: Decision, base_url: str) -> DecisionRecord:
    """Write the award text and its record the way a simulation run saves its award.

    The PDF link stops working when the public window closes, so the record
    keeps it with the time it was fetched; the text remains.
    """
    directory.mkdir(parents=True, exist_ok=True)
    record = DecisionRecord(
        case_id=decision.case_id,
        decision_id=decision.id,
        base_url=base_url,
        fetched_at=datetime.now(UTC),
        published_at=decision.published_at,
        pdf_url=decision.pdf_url,
        award_section=extract_award_section(decision.text),
    )
    payload = record.model_dump(mode="json", exclude_none=True)
    (directory / DECISION_FILENAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (directory / AWARD_FILENAME).write_text(decision.text.rstrip() + "\n", encoding="utf-8")
    return record
