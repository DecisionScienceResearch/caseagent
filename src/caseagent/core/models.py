"""Domain model: parties, the four phases, filings, and the case file.

Phase names follow DecisionLayer Arbitration Rule 1(b): Claim, Answer, Reply,
Response. The claimant files the Claim and the Reply; the respondent files the
Answer and the Response.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator


class Role(StrEnum):
    CLAIMANT = "claimant"
    RESPONDENT = "respondent"

    @property
    def opponent(self) -> Role:
        return Role.RESPONDENT if self is Role.CLAIMANT else Role.CLAIMANT

    @property
    def title(self) -> str:
        return self.value.capitalize()


class Stage(StrEnum):
    """A substantive phase of the arbitration, in filing order."""

    CLAIM = "claim"
    ANSWER = "answer"
    REPLY = "reply"
    RESPONSE = "response"

    @property
    def spec(self) -> StageSpec:
        return _SPECS[self]

    @property
    def role(self) -> Role:
        return self.spec.role

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def index(self) -> int:
        return STAGES.index(self)

    @property
    def prior(self) -> tuple[Stage, ...]:
        """Stages filed before this one, in order."""
        return STAGES[: self.index]

    @classmethod
    def parse(cls, value: str) -> Stage:
        """Accept a stage name, its 1-based number, or a common alias."""
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in _ALIASES:
            return _ALIASES[key]
        raise ValueError(
            f"Unknown stage {value!r}. Use one of: {', '.join(s.value for s in STAGES)} (or 1-4)."
        )


@dataclass(frozen=True)
class StageSpec:
    """What a phase is and which parts of a filing it may contain."""

    stage: Stage
    role: Role
    title: str
    rule: str
    purpose: str
    allows_counterclaim: bool = False
    allows_evidence_demands: bool = False
    allows_evidence_response: bool = False


STAGES: tuple[Stage, ...] = (Stage.CLAIM, Stage.ANSWER, Stage.REPLY, Stage.RESPONSE)

_SPECS: dict[Stage, StageSpec] = {
    Stage.CLAIM: StageSpec(
        stage=Stage.CLAIM,
        role=Role.CLAIMANT,
        title="Claim",
        rule="Rule 1(b)(1)",
        purpose="The claimant specifies its claim, explains the remedies sought, and "
        "lists any documentation it demands from the respondent.",
        allows_evidence_demands=True,
    ),
    Stage.ANSWER: StageSpec(
        stage=Stage.ANSWER,
        role=Role.RESPONDENT,
        title="Answer",
        rule="Rule 1(b)(2)",
        purpose="The respondent provides its responses, defenses, and affirmative "
        "defenses, may assert counterclaims and the remedies sought on them, responds "
        "to the claimant's documentation demands, and may demand documentation itself.",
        allows_counterclaim=True,
        allows_evidence_demands=True,
        allows_evidence_response=True,
    ),
    Stage.REPLY: StageSpec(
        stage=Stage.REPLY,
        role=Role.CLAIMANT,
        title="Reply",
        rule="Rule 1(b)(3)",
        purpose="The claimant provides further support for its claims, presents any "
        "defenses to counterclaims, and responds to the respondent's documentation "
        "demands.",
        allows_evidence_response=True,
    ),
    Stage.RESPONSE: StageSpec(
        stage=Stage.RESPONSE,
        role=Role.RESPONDENT,
        title="Response",
        rule="Rule 1(b)(4)",
        purpose="The respondent provides its final response in support of its "
        "defenses and any counterclaims.",
    ),
}

_ALIASES: dict[str, Stage] = {
    **{s.value: s for s in STAGES},
    **{str(i + 1): s for i, s in enumerate(STAGES)},
    "opening": Stage.CLAIM,
    "claimant_argument": Stage.CLAIM,
    "respondent_argument": Stage.ANSWER,
    "claimant_rebuttal": Stage.REPLY,
    "rebuttal": Stage.REPLY,
    "respondent_rebuttal": Stage.RESPONSE,
    "rejoinder": Stage.RESPONSE,
    "counter_response": Stage.RESPONSE,
}


def stages_for(role: Role) -> tuple[Stage, ...]:
    return tuple(s for s in STAGES if s.role is role)


def to_decimal(value: Any) -> Decimal | None:
    """Parse a money amount from a number or a string like ``$1,200.50``."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        raise ValueError("a boolean is not an amount")
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value))
    text = str(value).strip().replace(",", "").replace("$", "")
    text = text.removeprefix("USD").removesuffix("USD").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"not a money amount: {value!r}") from exc


def format_usd(amount: Decimal | None) -> str:
    """Decimal string the API expects, e.g. ``3500.00``."""
    return f"{(amount or Decimal(0)):.2f}"


def display_usd(amount: Decimal | None) -> str:
    """Amount for people to read, e.g. ``$3,500.00``."""
    return f"${(amount or Decimal(0)):,.2f}"


class Party(BaseModel):
    name: str
    description: str | None = None


class Counterclaim(BaseModel):
    """A claim the respondent asserts back against the claimant in its Answer."""

    argument: str
    amount_usd: Decimal | None = None
    other_relief: str | None = None

    @field_validator("amount_usd", mode="before")
    @classmethod
    def _amount(cls, value: Any) -> Decimal | None:
        return to_decimal(value)

    @field_serializer("amount_usd")
    def _amount_out(self, value: Decimal | None) -> str | None:
        return None if value is None else format_usd(value)


class DocumentKind(StrEnum):
    """Which part of a filing a submitted file supports."""

    SUPPORTING = "supporting"
    EVIDENCE_RESPONSE = "evidence_response"
    COUNTERCLAIM = "counterclaim"

    @property
    def describe(self) -> str:
        return _KIND_DESCRIPTIONS[self]


_KIND_DESCRIPTIONS: dict[DocumentKind, str] = {
    DocumentKind.SUPPORTING: "filed in support of the argument",
    DocumentKind.EVIDENCE_RESPONSE: "produced in response to documentation demands",
    DocumentKind.COUNTERCLAIM: "filed in support of the counterclaim",
}


class FiledDocument(BaseModel):
    """A file submitted with a filing, known by name.

    Its text is on hand only when a file of that name is also in the case
    folder; on a live case the other side's exhibits are known by name alone.
    """

    name: str
    kind: DocumentKind = DocumentKind.SUPPORTING


class Filing(BaseModel):
    """One party's submission for one phase.

    ``documents`` names the files submitted with it. A thread kept locally
    leaves it empty, since ``CaseFile.evidence`` already lists the exhibits per
    phase; a thread read from the API fills it from each round's file list.
    """

    stage: Stage
    argument: str
    counterclaim: Counterclaim | None = None
    evidence_demands: str | None = None
    evidence_response: str | None = None
    documents: list[FiledDocument] = Field(default_factory=list)

    @property
    def role(self) -> Role:
        return self.stage.role

    @property
    def word_count(self) -> int:
        return len(self.argument.split())

    @model_validator(mode="after")
    def _fits_stage(self) -> Filing:
        spec = self.stage.spec
        if self.counterclaim and not spec.allows_counterclaim:
            raise ValueError(f"a counterclaim cannot be filed with the {spec.title}")
        if self.evidence_demands and not spec.allows_evidence_demands:
            raise ValueError(f"documentation demands cannot be filed with the {spec.title}")
        if self.evidence_response and not spec.allows_evidence_response:
            raise ValueError(
                f"a response to documentation demands cannot be filed with the {spec.title}"
            )
        return self


class Author(StrEnum):
    """Who wrote a filing in the thread."""

    AGENT = "agent"
    HUMAN = "human"


class Statement(BaseModel):
    """One answer a party gave to the agent's intake questions.

    Statements are the party's own account of the facts, private to that party's
    side: the agent writing for the other side never sees them.
    """

    question: str
    answer: str
    given_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class CaseFile(BaseModel):
    """A dispute and everything a party has to argue it with.

    ``evidence`` maps each phase to the exhibit file names filed with it, so the
    Claim and Reply lists are the claimant's exhibits and the Answer and Response
    lists are the respondent's. The filings themselves live beside this file in
    the case folder (see ``core.thread``), never inside it.
    """

    key: str
    title: str
    claimant: Party
    respondent: Party
    question_for_arbitration: str
    financial_demand_usd: Decimal = Decimal(0)
    other_relief: str | None = None
    arbitration_clause: str | None = None
    governing_contract: str | None = None
    contract_file: str | None = None
    evidence: dict[Stage, list[str]] = Field(default_factory=dict)
    statements: dict[Role, list[Statement]] = Field(default_factory=dict)
    source: dict[str, Any] | None = None

    @field_validator("financial_demand_usd", mode="before")
    @classmethod
    def _demand(cls, value: Any) -> Decimal:
        return to_decimal(value) or Decimal(0)

    @field_serializer("financial_demand_usd")
    def _demand_out(self, value: Decimal) -> str:
        return format_usd(value)

    @property
    def is_sample(self) -> bool:
        return bool(self.source and self.source.get("kind") == "decisionlayer-sample")

    def party(self, role: Role) -> Party:
        return self.claimant if role is Role.CLAIMANT else self.respondent

    def evidence_names(self) -> list[str]:
        """Every exhibit file name, once, in filing order."""
        seen: dict[str, None] = {}
        for stage in STAGES:
            for name in self.evidence.get(stage, []):
                seen.setdefault(name, None)
        return list(seen)

    def missing_essentials(self) -> list[str]:
        """Fields the arbitration needs that are still blank (a scaffolded case)."""
        missing = []
        if not self.claimant.name.strip():
            missing.append("claimant name")
        if not self.respondent.name.strip():
            missing.append("respondent name")
        if not self.question_for_arbitration.strip():
            missing.append("the dispute (claim form statement)")
        if not (self.governing_contract or "").strip() and not self.contract_file:
            missing.append("the contract (text or file)")
        return missing
