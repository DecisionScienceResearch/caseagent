"""Read the model's tagged reply back into a Filing.

Replies use plain ``<section>...</section>`` tags rather than JSON: long prose
full of quotation marks survives tags far better than it survives JSON escaping,
and every chat model can produce them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from .models import Counterclaim, Filing, Stage
from .options import OutputPlan

SECTION_TAGS = (
    "argument",
    "counterclaim_argument",
    "counterclaim_amount_usd",
    "counterclaim_other_relief",
    "evidence_demands",
    "evidence_response",
    "notes_for_principal",
    "review_notes",
)

_EMPTY_RE = re.compile(r"^\W*(none|n/a|not applicable|nil|nothing)\W*$", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


class FormatError(ValueError):
    """The reply did not follow the required output format."""


@dataclass(frozen=True)
class ParsedReply:
    filing: Filing
    notes_for_principal: str | None = None
    review_notes: str | None = None


def extract_sections(text: str) -> dict[str, str]:
    """Return the body of each known section.

    When a tag appears more than once (a model echoing its draft before the
    final version), the last complete occurrence wins.
    """
    sections: dict[str, str] = {}
    for tag in SECTION_TAGS:
        matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
        if matches:
            sections[tag] = matches[-1].strip()
        elif re.search(rf"<{tag}>", text, re.IGNORECASE):
            raise FormatError(
                f"the <{tag}> section was opened but never closed, so the reply looks cut off"
            )
    return sections


def clean_optional(value: str | None) -> str | None:
    """Treat an empty section, or one that just says NONE, as absent."""
    if value is None:
        return None
    value = value.strip()
    if not value or _EMPTY_RE.match(value):
        return None
    return value


def strip_markdown(text: str) -> str:
    """Remove the Markdown markup models add out of habit; filings are plain text."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"\1", text)
    text = re.sub(r"(?<![\w_])__([^_\n]+?)__(?![\w_])", r"\1", text)
    text = re.sub(r"^[ \t]*#{1,6}[ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^([ \t]*)\*[ \t]+", r"\1- ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_amount(value: str | None) -> Decimal | None:
    """Pull a dollar amount out of text like ``$12,500.00`` or ``USD 12500``."""
    value = clean_optional(value)
    if value is None:
        return None
    match = _AMOUNT_RE.search(value)
    if not match:
        return None
    amount = Decimal(match.group(0).replace(",", ""))
    return amount if amount > 0 else None


def parse_filing(stage: Stage, raw: str, plan: OutputPlan) -> ParsedReply:
    """Build the Filing for ``stage`` from a tagged reply.

    Sections the plan does not allow are ignored, so a model that volunteers a
    counterclaim in a Reply cannot produce an invalid filing.
    """
    sections = extract_sections(raw)
    argument = clean_optional(sections.get("argument"))
    if argument is None:
        raise FormatError("the reply has no <argument> section")

    counterclaim = None
    if plan.counterclaim:
        counterclaim_argument = clean_optional(sections.get("counterclaim_argument"))
        if counterclaim_argument is not None:
            counterclaim = Counterclaim(
                argument=strip_markdown(counterclaim_argument),
                amount_usd=parse_amount(sections.get("counterclaim_amount_usd")),
                other_relief=clean_optional(sections.get("counterclaim_other_relief")),
            )

    demands = clean_optional(sections.get("evidence_demands")) if plan.evidence_demands else None
    response = clean_optional(sections.get("evidence_response")) if plan.evidence_response else None

    filing = Filing(
        stage=stage,
        argument=strip_markdown(argument),
        counterclaim=counterclaim,
        evidence_demands=strip_markdown(demands) if demands else None,
        evidence_response=strip_markdown(response) if response else None,
    )
    return ParsedReply(
        filing=filing,
        notes_for_principal=clean_optional(sections.get("notes_for_principal")),
        review_notes=clean_optional(sections.get("review_notes")),
    )


def render_sections(parsed: ParsedReply, plan: OutputPlan) -> str:
    """Render a parsed reply back into the tagged format (used to hand a draft to review)."""
    filing = parsed.filing
    parts = [_section("argument", filing.argument)]
    if plan.counterclaim:
        counterclaim = filing.counterclaim
        amount = (
            f"{counterclaim.amount_usd:.2f}"
            if counterclaim and counterclaim.amount_usd is not None
            else None
        )
        parts.append(
            _section("counterclaim_argument", counterclaim.argument if counterclaim else None)
        )
        parts.append(_section("counterclaim_amount_usd", amount))
        parts.append(
            _section(
                "counterclaim_other_relief", counterclaim.other_relief if counterclaim else None
            )
        )
    if plan.evidence_demands:
        parts.append(_section("evidence_demands", filing.evidence_demands))
    if plan.evidence_response:
        parts.append(_section("evidence_response", filing.evidence_response))
    parts.append(_section("notes_for_principal", parsed.notes_for_principal))
    return "\n\n".join(parts)


def _section(tag: str, body: str | None) -> str:
    return f"<{tag}>\n{body or 'NONE'}\n</{tag}>"


_QUESTION_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL | re.IGNORECASE)
_READY_RE = re.compile(r"<ready\s*/?>", re.IGNORECASE)


def parse_questions(text: str, limit: int) -> list[str]:
    """Intake questions from an interview reply; empty when the model says it is ready."""
    if _READY_RE.search(text) and not _QUESTION_RE.search(text):
        return []
    questions = [" ".join(q.split()) for q in _QUESTION_RE.findall(text)]
    questions = [q for q in questions if q]
    if not questions:
        raise FormatError("the reply has no <question> sections and no <ready/> marker")
    return questions[:limit]
