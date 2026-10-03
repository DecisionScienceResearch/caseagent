from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from caseagent.core.casefile import CaseBundle, save_case
from caseagent.core.llm import ChatRequest, ChatResponse, Usage
from caseagent.core.models import Author, CaseFile, Filing, Party, Stage
from caseagent.core.thread import ThreadEntry, ThreadStore


def make_pdf(text: str) -> bytes:
    """A minimal one-page PDF whose text pypdf can extract."""
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


SAMPLE_RAW = {
    "arbitration_rules": "Administered by DecisionLayer under New York law.",
    "governing_contract": "LEASE\r\nSection 2. The Commencement Date is the date the Tenant opens, "
    "so long as such date is not in excess of sixty (60) days after execution.",
    "question_for_arbitration": "Whether rent accrued for May and June 2022.",
    "financial_demand_usd": 42000.0,
    "other_relief": "",
    "plaintiff": {"name": "BevMart Inc."},
    "respondent": {"name": "Arcology LLC"},
    "plaintiff_argument": {
        "argument": "SAMPLE CLAIM: rent did not accrue.\r\nWe could not open.",
        "documents": [],
        "images": [],
    },
    "respondent_argument": {
        "argument": "SAMPLE ANSWER: the 60-day cap applies.",
        "documents": [],
        "images": [],
    },
    "plaintiff_rebuttal": {
        "rebuttal": "SAMPLE REPLY: contra proferentem applies.",
        "documents": [],
        "images": [],
    },
    "respondent_rebuttal": {"rebuttal": "", "documents": [], "images": []},
    "attachments": {
        "plaintiff_documents": ["attachments/lease/Exhibit A-1 - POS Ledger.txt"],
        "respondent_documents": [
            "attachments/lease/Exhibit R-1 - Notice.pdf",
            {"file": "attachments/lease/Exhibit R-2 - Ledger.pdf"},
        ],
        "plaintiff_rebuttal_documents": [
            "attachments/lease/Exhibit A-1 - POS Ledger.txt",
            "attachments/lease/Exhibit A-3 - Emails.pdf",
        ],
        "respondent_rebuttal_documents": [],
    },
}

SAMPLE_FILES = {
    "attachments/lease/Exhibit A-1 - POS Ledger.txt": b"Transaction,Amount\nTX-1,48.00\n",
    "attachments/lease/Exhibit R-1 - Notice.pdf": make_pdf(
        "Notice of Lease Commencement May 1 2022"
    ),
    "attachments/lease/Exhibit R-2 - Ledger.pdf": make_pdf("Tenant AR Ledger total 42000"),
    "attachments/lease/Exhibit A-3 - Emails.pdf": make_pdf("Email of April 18 2022 about the SLA"),
}

HUMAN_TEXT = {
    Stage.CLAIM: "HUMAN CLAIM: rent did not accrue because we could not legally open.",
    Stage.ANSWER: "HUMAN ANSWER: the 60-day cap fixed the Commencement Date.",
    Stage.REPLY: "HUMAN REPLY: 'so long as' is a condition, not a trigger.",
    Stage.RESPONSE: "HUMAN RESPONSE: both parties are sophisticated.",
}


@pytest.fixture
def case_bundle(tmp_path: Path) -> CaseBundle:
    """A small four-phase case with text exhibits and an empty thread."""
    case = CaseFile(
        key="lease",
        title="Lease Commencement",
        claimant=Party(name="BevMart Inc."),
        respondent=Party(name="Arcology LLC"),
        question_for_arbitration="Whether rent accrued for May and June 2022.",
        financial_demand_usd=Decimal("42000.00"),
        arbitration_clause="Administered by DecisionLayer under New York law.",
        governing_contract="LEASE\nSection 2. Commencement Date: the date the Tenant opens, "
        "so long as such date is not in excess of sixty (60) days after execution.",
        evidence={
            Stage.CLAIM: ["Exhibit A-1 - POS Ledger.txt", "Exhibit A-2 - License.txt"],
            Stage.ANSWER: ["Exhibit R-1 - Notice.txt"],
            Stage.REPLY: ["Exhibit A-3 - Emails.txt"],
            Stage.RESPONSE: ["Exhibit R-3 - LOI.txt"],
        },
        source={"kind": "own"},
    )
    root = tmp_path / "cases" / "lease"
    bundle = save_case(case, root)
    texts = {
        "Exhibit A-1 - POS Ledger.txt": "Transaction,Amount\nTX-2022-0701-001,48.00\n",
        "Exhibit A-2 - License.txt": "License issued June 24, 2022.",
        "Exhibit R-1 - Notice.txt": "Notice: Commencement Date established as May 1, 2022.",
        "Exhibit A-3 - Emails.txt": "Email of April 18, 2022: our rent clock has not started.",
        "Exhibit R-3 - LOI.txt": "LOI redline: landlord rejected an open-ended concession.",
    }
    for name, text in texts.items():
        bundle.evidence_path(name).write_text(text, encoding="utf-8")
    return bundle


def human_filing(stage: Stage, **extra) -> Filing:
    return Filing(stage=stage, argument=HUMAN_TEXT[stage], **extra)


def seed_thread(bundle: CaseBundle, *stages: Stage) -> ThreadStore:
    """Record human filings for ``stages`` (which must start at the Claim)."""
    store = ThreadStore(bundle.root)
    for stage in stages:
        store.append(ThreadEntry(filing=human_filing(stage), author=Author.HUMAN))
    return store


def tagged_reply(stage: Stage, counterclaim: bool = False, extra: str = "") -> str:
    """A well-formed model reply for ``stage``."""
    parts = [f"<argument>\nAGENT {stage.title.upper()}: the record shows it.{extra}\n</argument>"]
    if stage is Stage.ANSWER:
        if counterclaim:
            parts += [
                "<counterclaim_argument>\nUnpaid rent is owed under Section 2.\n</counterclaim_argument>",
                "<counterclaim_amount_usd>\n$42,000.00\n</counterclaim_amount_usd>",
                "<counterclaim_other_relief>\nNONE\n</counterclaim_other_relief>",
            ]
        else:
            parts += [
                "<counterclaim_argument>\nNONE\n</counterclaim_argument>",
                "<counterclaim_amount_usd>\nNONE\n</counterclaim_amount_usd>",
                "<counterclaim_other_relief>\nNONE\n</counterclaim_other_relief>",
            ]
    parts.append("<notes_for_principal>\nVerify the opening date.\n</notes_for_principal>")
    return "\n\n".join(parts)


class ScriptedClient:
    """Returns canned replies in order; records every request."""

    model = "scripted:test"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest, on_progress=None) -> ChatResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError(f"no scripted reply left for {request.label}")
        text = self.replies.pop(0)
        if on_progress:
            on_progress(len(text))
        return ChatResponse(
            text=text,
            model=self.model,
            finish_reason="stop",
            usage=Usage(calls=1, prompt_tokens=1000, completion_tokens=100, cost_usd=0.01),
            seconds=0.5,
        )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
