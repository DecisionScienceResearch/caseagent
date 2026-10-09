from decimal import Decimal

import pytest

from caseagent.core.casefile import (
    CaseError,
    add_evidence,
    load_case,
    resolve_case,
    save_case,
    scaffold_case,
)
from caseagent.core.models import (
    CaseFile,
    Counterclaim,
    DocumentKind,
    Filing,
    Role,
    Stage,
    Statement,
    display_usd,
    format_usd,
    stages_for,
    to_decimal,
)


def test_stage_roles_and_order():
    assert Stage.CLAIM.role is Role.CLAIMANT
    assert Stage.ANSWER.role is Role.RESPONDENT
    assert Stage.REPLY.role is Role.CLAIMANT
    assert Stage.RESPONSE.role is Role.RESPONDENT
    assert Stage.REPLY.prior == (Stage.CLAIM, Stage.ANSWER)
    assert stages_for(Role.RESPONDENT) == (Stage.ANSWER, Stage.RESPONSE)
    assert Role.CLAIMANT.opponent is Role.RESPONDENT


@pytest.mark.parametrize(
    ("text", "stage"),
    [
        ("claim", Stage.CLAIM),
        ("1", Stage.CLAIM),
        ("Answer", Stage.ANSWER),
        ("respondent_argument", Stage.ANSWER),
        ("rebuttal", Stage.REPLY),
        ("4", Stage.RESPONSE),
        ("respondent-rebuttal", Stage.RESPONSE),
    ],
)
def test_stage_parse(text, stage):
    assert Stage.parse(text) is stage


def test_stage_parse_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown stage"):
        Stage.parse("verdict")


def test_only_the_answer_may_carry_a_counterclaim():
    Filing(stage=Stage.ANSWER, argument="x", counterclaim=Counterclaim(argument="y"))
    with pytest.raises(ValueError, match="counterclaim"):
        Filing(stage=Stage.REPLY, argument="x", counterclaim=Counterclaim(argument="y"))
    with pytest.raises(ValueError, match="demands"):
        Filing(stage=Stage.RESPONSE, argument="x", evidence_demands="docs")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (24500.0, Decimal("24500.0")),
        ("$1,200.50", Decimal("1200.50")),
        ("USD 300", Decimal("300")),
        (None, None),
        ("", None),
    ],
)
def test_to_decimal(raw, expected):
    assert to_decimal(raw) == expected


def test_money_formats():
    assert format_usd(Decimal("42000")) == "42000.00"
    assert display_usd(Decimal("42000")) == "$42,000.00"
    assert format_usd(None) == "0.00"


def test_case_roundtrip_and_money_serialization(case_bundle):
    loaded = load_case(case_bundle.root)
    assert loaded.case == case_bundle.case
    raw = (case_bundle.root / "case.json").read_text()
    assert '"financial_demand_usd": "42000.00"' in raw
    assert '"claim": [' in raw  # stage keys serialize by value
    assert "baseline" not in raw


def test_statements_roundtrip(case_bundle):
    case = case_bundle.case.model_copy(
        update={
            "statements": {
                Role.CLAIMANT: [Statement(question="When did you open?", answer="July 1, 2022")]
            }
        }
    )
    save_case(case, case_bundle.root)
    loaded = load_case(case_bundle.root).case
    assert loaded.statements[Role.CLAIMANT][0].answer == "July 1, 2022"
    assert Role.RESPONDENT not in loaded.statements


def test_load_case_reports_missing_evidence(case_bundle):
    case_bundle.evidence_path("Exhibit R-1 - Notice.txt").unlink()
    with pytest.raises(CaseError, match="Exhibit R-1 - Notice.txt"):
        load_case(case_bundle.root)


def test_resolve_case_by_key_or_path(case_bundle, tmp_path):
    cases_dir = case_bundle.root.parent
    assert resolve_case("lease", cases_dir).case.key == "lease"
    assert resolve_case(str(case_bundle.root), tmp_path / "elsewhere").case.key == "lease"
    with pytest.raises(CaseError, match="No case named 'nope'"):
        resolve_case("nope", cases_dir)


def test_scaffold_case_is_blank_and_reports_what_is_missing(tmp_path):
    bundle = scaffold_case(tmp_path / "my-dispute", "my-dispute")
    case = load_case(bundle.root).case
    assert case.missing_essentials() == [
        "claimant name",
        "respondent name",
        "the dispute (claim form statement)",
        "the contract (text or file)",
    ]
    assert not case.is_sample
    with pytest.raises(CaseError, match="already exists"):
        scaffold_case(tmp_path / "my-dispute", "my-dispute")


def test_add_evidence_copies_and_registers(case_bundle, tmp_path):
    receipt = tmp_path / "Receipt.txt"
    receipt.write_text("paid $500")
    bundle = add_evidence(case_bundle, Stage.CLAIM, [receipt])
    assert bundle.case.evidence[Stage.CLAIM][-1] == "Receipt.txt"
    assert bundle.evidence_path("Receipt.txt").read_text() == "paid $500"
    again = add_evidence(bundle, Stage.CLAIM, [receipt])  # idempotent
    assert again.case.evidence[Stage.CLAIM].count("Receipt.txt") == 1
    other = tmp_path / "other" / "Receipt.txt"
    other.parent.mkdir()
    other.write_text("different")
    with pytest.raises(CaseError, match="different file"):
        add_evidence(again, Stage.CLAIM, [other])


def test_save_case_strips_none(tmp_path):
    case = CaseFile(
        key="k",
        title="t",
        claimant={"name": "A"},
        respondent={"name": "B"},
        question_for_arbitration="q",
    )
    save_case(case, tmp_path / "k")
    assert '"other_relief"' not in (tmp_path / "k" / "case.json").read_text()


def test_filing_documents_roundtrip():
    filing = Filing(
        stage=Stage.ANSWER,
        argument="a",
        documents=[{"name": "Exhibit R-1.pdf", "kind": "counterclaim"}],
    )
    assert filing.documents[0].kind is DocumentKind.COUNTERCLAIM
    assert Filing.model_validate_json(filing.model_dump_json()) == filing
    assert Filing(stage=Stage.CLAIM, argument="c").documents == []
    assert DocumentKind.EVIDENCE_RESPONSE.describe.startswith("produced in response")
