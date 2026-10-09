"""What one party can see when it writes one filing.

This is the information boundary of the agent. At a given phase a party sees
the claim form, the contract, every earlier filing, the exhibits filed with
those earlier filings, its own exhibits for the filing it is writing, and its
own principal's statements. It does not see later filings, the other side's
unfiled exhibits, or the other side's statements.

A filing read from a live case names the files filed with it. One that is not
in the case folder is known by name only; ``StageView.unavailable_documents``
lists those so the caller can say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .casefile import CaseBundle
from .documents import extract_text
from .models import Filing, Party, Role, Stage, Statement

_LABEL_RE = re.compile(r"^\s*(exhibit\s+[A-Za-z]{0,3}-?\d+[A-Za-z]?)\b", re.IGNORECASE)


def exhibit_label(file_name: str) -> str:
    """Short citation label for an exhibit: ``Exhibit A-2`` from its file name."""
    stem = Path(file_name).stem
    match = _LABEL_RE.match(stem)
    return match.group(1) if match else stem


@dataclass(frozen=True)
class ExhibitView:
    name: str
    label: str
    filed_by: Role
    filed_with: Stage
    text: str
    is_new: bool
    warning: str | None = None


@dataclass(frozen=True)
class StageView:
    case_key: str
    title: str
    stage: Stage
    party: Party
    opponent: Party
    question_for_arbitration: str
    financial_demand_usd: Decimal
    other_relief: str | None
    arbitration_clause: str | None
    contract_text: str
    record: tuple[Filing, ...]
    exhibits: tuple[ExhibitView, ...]
    statements: tuple[Statement, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def role(self) -> Role:
        return self.stage.role

    @property
    def claimant(self) -> Party:
        return self.party if self.role is Role.CLAIMANT else self.opponent

    @property
    def respondent(self) -> Party:
        return self.party if self.role is Role.RESPONDENT else self.opponent

    def party_for(self, role: Role) -> Party:
        return self.claimant if role is Role.CLAIMANT else self.respondent

    @property
    def counterclaim_filed(self) -> bool:
        return any(filing.counterclaim for filing in self.record)

    @property
    def pending_demands(self) -> str | None:
        """Documentation the opponent demanded in its latest filing, if any."""
        for filing in reversed(self.record):
            if filing.role is self.role.opponent:
                return filing.evidence_demands
        return None

    @property
    def new_exhibits(self) -> tuple[ExhibitView, ...]:
        return tuple(exhibit for exhibit in self.exhibits if exhibit.is_new)

    @property
    def unavailable_documents(self) -> tuple[str, ...]:
        """Files the record says were filed whose text is not among the exhibits."""
        on_hand = {exhibit.name for exhibit in self.exhibits}
        missing: dict[str, None] = {}
        for filing in self.record:
            for document in filing.documents:
                if document.name not in on_hand:
                    missing.setdefault(document.name, None)
        return tuple(missing)


def build_stage_view(bundle: CaseBundle, stage: Stage, thread: dict[Stage, Filing]) -> StageView:
    """Assemble the view for ``stage`` from the case and the filings made so far.

    ``thread`` holds the filings already made. Only phases before ``stage`` are
    read from it; a phase missing from it was skipped and contributes neither a
    filing nor exhibits.
    """
    case = bundle.case
    role = stage.role
    record = tuple(thread[prior] for prior in stage.prior if prior in thread)

    exhibits: list[ExhibitView] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for filed_with in (*stage.prior, stage):
        if filed_with is not stage and filed_with not in thread:
            continue
        for name in case.evidence.get(filed_with, []):
            if name in seen:
                continue
            seen.add(name)
            extracted = extract_text(bundle.evidence_path(name))
            if extracted.warning:
                warnings.append(extracted.warning)
            exhibits.append(
                ExhibitView(
                    name=name,
                    label=exhibit_label(name),
                    filed_by=filed_with.role,
                    filed_with=filed_with,
                    text=extracted.text,
                    is_new=filed_with is stage,
                    warning=extracted.warning,
                )
            )

    contract = bundle.contract()
    if contract.warning:
        warnings.append(contract.warning)

    return StageView(
        case_key=case.key,
        title=case.title,
        stage=stage,
        party=case.party(role),
        opponent=case.party(role.opponent),
        question_for_arbitration=case.question_for_arbitration.strip(),
        financial_demand_usd=case.financial_demand_usd,
        other_relief=(case.other_relief or "").strip() or None,
        arbitration_clause=(case.arbitration_clause or "").strip() or None,
        contract_text=contract.text,
        record=record,
        exhibits=tuple(exhibits),
        statements=tuple(case.statements.get(role, [])),
        warnings=tuple(warnings),
    )
