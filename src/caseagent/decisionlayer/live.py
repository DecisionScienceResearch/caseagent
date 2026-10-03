"""Map the agent's filings onto a real case's turns (``/api/v1/cases``).

The API tells each party what the case is waiting for: ``status``,
``current_turn``, ``next_round`` and ``accepted_fields``. Rounds map onto the
phases as 1 = Answer, 2 = Reply, 3 = Response; the Claim is the case itself.
A round can be submitted once, and the party's affirmation that the filing is
accurate and complete is a human act, so ``response_form`` requires it to be
given explicitly.
"""

from __future__ import annotations

from collections.abc import Collection

from ..core.models import Counterclaim, Filing, Stage, format_usd
from .client import ArbitrationCase, CaseResponse

ROUND_STAGES: dict[int, Stage] = {1: Stage.ANSWER, 2: Stage.REPLY, 3: Stage.RESPONSE}
STAGE_ROUNDS: dict[Stage, int] = {stage: number for number, stage in ROUND_STAGES.items()}


class LiveCaseError(Exception):
    pass


def thread_from_api(
    case: ArbitrationCase, responses: Collection[CaseResponse]
) -> dict[Stage, Filing]:
    """Rebuild the filings so far from the case and its response thread."""
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
        )
    return thread


def stage_to_file(case: ArbitrationCase) -> Stage:
    """The phase this key must file now, or an explanation of why it cannot."""
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
