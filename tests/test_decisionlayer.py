import json
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote

import httpx
import pytest
from conftest import SAMPLE_FILES, SAMPLE_RAW, human_filing

from caseagent.core.models import Counterclaim, Filing, Stage
from caseagent.decisionlayer import client as client_module
from caseagent.decisionlayer.client import (
    ArbitrationCase,
    Attachment,
    CaseResponse,
    DecisionLayerClient,
    DecisionLayerError,
    SimulationFailed,
    SimulationTimeout,
)
from caseagent.decisionlayer.live import (
    LiveCaseError,
    response_form,
    stage_to_file,
    thread_from_api,
)
from caseagent.decisionlayer.samples import (
    SAMPLE_ARGUMENTS_FILENAME,
    SampleStore,
    attachment_refs,
    fetch_sample,
    ingest_sample,
    sample_to_case,
    title_from_filename,
)
from caseagent.decisionlayer.simulator import (
    SimulationInputError,
    SimulationRecord,
    await_award,
    build_simulation_request,
    extract_award_section,
    load_simulation,
    render_for_simulator,
    save_simulation,
    submit_simulation,
)

SAMPLE_AWARD = """ARBITRATION DECISION
==================================================
PARTIES:
Claimant: LogisticsCo LLC
Respondent: SolidEdge, Inc.
FINDINGS OF LAW:
1. The clause is unenforceable [3].
AWARD:
Respondent, SolidEdge, Inc., is ordered to pay Claimant, LogisticsCo LLC, the sum of $100.00 USD. Each party shall bear its own costs and fees.
CITATIONS:
1. Arbitration Rules
==================================================
Generated on: 2026-08-26 21:19:04 UTC
"""


def parse_multipart(request: httpx.Request) -> list[tuple[str, str | None, bytes]]:
    """(name, filename, body) for each part of a multipart request."""
    content_type = request.headers["content-type"]
    boundary = content_type.split("boundary=")[1].encode()
    parts = []
    for raw in request.content.split(b"--" + boundary)[1:-1]:
        header, _, body = raw.lstrip(b"\r\n").partition(b"\r\n\r\n")
        disposition = header.decode().split("\r\n")[0]
        name = re.search(r'name="([^"]+)"', disposition).group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        parts.append((name, filename_match.group(1) if filename_match else None, body[:-2]))
    return parts


def api_error(status: int, message: str, details=(), reason=None) -> httpx.Response:
    error = {"status": status, "message": message, "details": list(details)}
    if reason:
        error["reason"] = reason
    return httpx.Response(status, json={"error": error})


def make_client(handler, key="dvarb_live_123", sleeps=None) -> DecisionLayerClient:
    sleeps = sleeps if sleeps is not None else []
    return DecisionLayerClient(
        key, "https://staging.example", transport=httpx.MockTransport(handler), sleep=sleeps.append
    )


def simulation_json(status="processing", **extra) -> dict:
    return {
        "id": "case_sim1",
        "status": status,
        "page_url": "https://x/sim",
        "created_at": "2026-10-02T00:00:00Z",
        **extra,
    }


# ------------------------------------------------------------------------ client


def test_me_and_error_shape():
    def handler(request):
        assert request.headers["authorization"] == "Bearer dvarb_live_123"
        if request.url.path == "/api/v1/me":
            return httpx.Response(
                200,
                json={
                    "id": "u1",
                    "email": "a@b.c",
                    "first_name": "A",
                    "last_name": "B",
                    "can_file_cases": False,
                    "created_at": "2026-01-01T00:00:00Z",
                },
            )
        return api_error(404, "Not found", ["No such case."], reason="not_a_party")

    client = make_client(handler)
    assert client.me().email == "a@b.c"
    assert client.is_test_key is False
    with pytest.raises(DecisionLayerError) as excinfo:
        client.get_case("case_x")
    assert excinfo.value.status == 404 and excinfo.value.reason == "not_a_party"
    assert "No such case." in str(excinfo.value)


def test_create_simulation_sends_multipart_with_idempotency(tmp_path: Path):
    exhibit = tmp_path / "Exhibit A-1 - Ledger.txt"
    exhibit.write_text("TX,48.00")
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        captured["parts"] = parse_multipart(request)
        return httpx.Response(201, json=simulation_json())

    client = make_client(handler)
    simulation = client.create_simulation(
        {"question_for_arbitration": "Q?", "plaintiff_argument": "P", "other_relief": ""},
        [Attachment("plaintiff_documents", exhibit)],
        idempotency_key="idem-1",
    )
    assert simulation.id == "case_sim1" and simulation.status == "processing"
    assert captured["headers"]["idempotency-key"] == "idem-1"
    assert captured["headers"]["content-type"].startswith("multipart/form-data")
    parts = captured["parts"]
    assert ("question_for_arbitration", None, b"Q?") in parts
    assert ("plaintiff_argument", None, b"P") in parts
    assert not any(name == "other_relief" for name, _, _ in parts)  # empty fields are omitted
    file_part = next(part for part in parts if part[0] == "plaintiff_documents")
    assert file_part[1] == "Exhibit A-1 - Ledger.txt" and file_part[2] == b"TX,48.00"


def test_large_files_go_through_upload_tickets(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(client_module, "LARGE_FILE_BYTES", 10)
    small = tmp_path / "small.txt"
    small.write_text("tiny")
    big = tmp_path / "big.txt"
    big.write_text("x" * 50)
    seen = []

    def handler(request):
        seen.append((request.method, request.url.host, request.url.path))
        if request.url.path == "/api/v1/uploads":
            files = json.loads(request.content)["files"]
            assert [f["filename"] for f in files] == ["small.txt", "big.txt"]
            uploads = [
                {
                    "upload_id": f"u{i}",
                    "filename": f["filename"],
                    "content_type": "text/plain",
                    "size": f["size"],
                    "upload_url": f"https://storage.example/{f['filename']}",
                    "ticket": f"ticket-{i}",
                    "expires_at": "2026-10-02T00:00:00Z",
                }
                for i, f in enumerate(files)
            ]
            return httpx.Response(200, json={"batch_id": "b1", "uploads": uploads})
        if request.url.host == "storage.example":
            assert "authorization" not in request.headers
            assert request.headers["content-type"] == "text/plain"
            return httpx.Response(200)
        parts = parse_multipart(request)
        assert ("plaintiff_documents_tickets", None, b"ticket-0") in parts
        assert ("contract_file_ticket", None, b"ticket-1") in parts
        assert not any(filename for _, filename, _ in parts)
        return httpx.Response(201, json=simulation_json())

    client = make_client(handler)
    client.create_simulation(
        {"q": "x"}, [Attachment("plaintiff_documents", small), Attachment("contract_file", big)]
    )
    assert seen[0][2] == "/api/v1/uploads"
    assert seen[1][1] == "storage.example"
    assert seen[-1][2] == "/api/v1/simulations"


def test_wait_for_simulation_polls_until_ready():
    statuses = iter(["processing", "retrying", "ready"])

    def handler(request):
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "id": "case_sim1",
                    "case_id": "case_sim1",
                    "text": SAMPLE_AWARD,
                    "published_at": "2026-10-02T00:00:00Z",
                    "pdf_url": "https://x/a.pdf",
                    "page_url": "https://x/sim",
                },
            )
        return httpx.Response(200, json=simulation_json(next(statuses)))

    sleeps = []
    client = make_client(handler, sleeps=sleeps)
    polled = []
    simulation = client.wait_for_simulation(
        "case_sim1", timeout=100, interval=7, on_poll=polled.append
    )
    assert simulation.status == "ready" and sleeps == [7, 7]
    assert [s.status for s in polled] == ["processing", "retrying", "ready"]
    assert client.get_simulation_result("case_sim1").pdf_url == "https://x/a.pdf"


def test_wait_for_simulation_failure_and_timeout(monkeypatch):
    def failed(request):
        return httpx.Response(200, json=simulation_json("failed", message="generation stopped"))

    with pytest.raises(SimulationFailed, match="generation stopped"):
        make_client(failed).wait_for_simulation("case_sim1")

    def forever(request):
        return httpx.Response(200, json=simulation_json("processing"))

    clock = iter([0.0, 0.0, 50.0, 1000.0])
    monkeypatch.setattr(client_module.time, "monotonic", lambda: next(clock))
    with pytest.raises(SimulationTimeout, match="caseagent sim result case_sim1"):
        make_client(forever).wait_for_simulation("case_sim1", timeout=900, interval=1)


def test_retries_transient_failures_with_same_idempotency_key():
    attempts = []

    def handler(request):
        attempts.append(request.headers.get("idempotency-key"))
        if len(attempts) == 1:
            raise httpx.ConnectError("boom")
        if len(attempts) == 2:
            return api_error(503, "busy")
        return httpx.Response(201, json=simulation_json())

    sleeps = []
    client = make_client(handler, sleeps=sleeps)
    assert client.create_simulation({"a": "b"}, idempotency_key="same").id == "case_sim1"
    assert attempts == ["same", "same", "same"] and len(sleeps) == 2


def test_client_rejects_missing_key_and_bad_attachments(tmp_path):
    with pytest.raises(DecisionLayerError, match="DECISIONLAYER_API_KEY"):
        DecisionLayerClient("")
    client = make_client(lambda request: httpx.Response(500))
    with pytest.raises(DecisionLayerError, match="not a file field"):
        client.create_simulation({}, [Attachment("nope", tmp_path / "x")])
    with pytest.raises(DecisionLayerError, match="not found"):
        client.create_simulation({}, [Attachment("evidence", tmp_path / "missing.pdf")])


# ----------------------------------------------------------------------- samples


def test_sample_to_case_keeps_the_dispute_and_drops_the_arguments():
    case = sample_to_case("lease", SAMPLE_RAW, title="Lease")
    assert case.claimant.name == "BevMart Inc." and case.respondent.name == "Arcology LLC"
    assert str(case.financial_demand_usd) == "42000.0"
    assert case.other_relief is None
    assert "\r" not in case.governing_contract
    assert case.is_sample
    dumped = case.model_dump_json()
    assert "SAMPLE CLAIM" not in dumped and "SAMPLE ANSWER" not in dumped
    assert case.evidence[Stage.ANSWER] == ["Exhibit R-1 - Notice.pdf", "Exhibit R-2 - Ledger.pdf"]
    assert case.evidence[Stage.REPLY] == [
        "Exhibit A-1 - POS Ledger.txt",
        "Exhibit A-3 - Emails.pdf",
    ]
    assert case.evidence_names() == [
        "Exhibit A-1 - POS Ledger.txt",
        "Exhibit R-1 - Notice.pdf",
        "Exhibit R-2 - Ledger.pdf",
        "Exhibit A-3 - Emails.pdf",
    ]
    assert (
        attachment_refs(SAMPLE_RAW)["Exhibit R-2 - Ledger.pdf"]
        == "attachments/lease/Exhibit R-2 - Ledger.pdf"
    )
    assert (
        title_from_filename("Commercial Lease Commencement Dispute.json")
        == "Commercial Lease Commencement Dispute"
    )
    assert title_from_filename("saas-outage-dispute.json") == "SaaS Outage Dispute"


def test_ingest_sample_writes_case_folder_and_reference_file(tmp_path):
    bundle = ingest_sample(
        "lease", SAMPLE_RAW, tmp_path, lambda path: SAMPLE_FILES[path], title="Lease"
    )
    assert (bundle.root / "case.json").exists()
    assert sorted(p.name for p in bundle.evidence_dir.iterdir()) == sorted(
        bundle.case.evidence_names()
    )
    assert bundle.case.source["attachments"] == 4
    assert bundle.missing_files() == []
    assert not (bundle.root / "filings").exists()
    reference = (bundle.root / SAMPLE_ARGUMENTS_FILENAME).read_text()
    assert "Reference only" in reference and "SAMPLE ANSWER" in reference


def test_fetch_sample_from_store(tmp_path):
    def handler(request):
        path = request.url.path
        if path == "/static/js/simulate_cases.json":
            return httpx.Response(200, json={"lease": "Commercial Lease Commencement Dispute.json"})
        if path == "/samples/arbitration_cases/Commercial Lease Commencement Dispute.json":
            return httpx.Response(200, json=SAMPLE_RAW)
        rel = unquote(httpx.URL(str(request.url)).path.removeprefix("/samples/arbitration_cases/"))
        return httpx.Response(200, content=SAMPLE_FILES[rel])

    store = SampleStore("https://staging.example", transport=httpx.MockTransport(handler))
    bundle = fetch_sample("lease", tmp_path, store)
    assert bundle.case.title == "Commercial Lease Commencement Dispute"
    assert bundle.evidence_path("Exhibit R-1 - Notice.pdf").read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------- simulator


def test_build_simulation_request(case_bundle):
    thread = {stage: human_filing(stage) for stage in Stage}
    thread[Stage.ANSWER] = Filing(
        stage=Stage.ANSWER,
        argument="Denied.",
        counterclaim=Counterclaim(
            argument="Rent owed.", amount_usd="42000", other_relief="Interest."
        ),
        evidence_demands="The POS export.",
    )
    request = build_simulation_request(case_bundle, thread)
    fields = request.fields
    assert (
        fields["plaintiff_name"] == "BevMart Inc." and fields["respondent_name"] == "Arcology LLC"
    )
    assert fields["financial_demand_usd"] == "42000.00"
    assert fields["governing_contract"].startswith("LEASE")
    assert fields["plaintiff_argument"].startswith("HUMAN CLAIM")
    assert "COUNTERCLAIM" in fields["respondent_argument"]
    assert "Amount demanded on the counterclaim: $42000.00" in fields["respondent_argument"]
    assert "DOCUMENTATION DEMANDED" in fields["respondent_argument"]
    assert fields["respondent_rebuttal"].startswith("HUMAN RESPONSE")
    assert "other_relief" not in fields
    assert [(a.field, a.path.name) for a in request.attachments] == [
        ("plaintiff_documents", "Exhibit A-1 - POS Ledger.txt"),
        ("plaintiff_documents", "Exhibit A-2 - License.txt"),
        ("respondent_documents", "Exhibit R-1 - Notice.txt"),
        ("plaintiff_rebuttal_documents", "Exhibit A-3 - Emails.txt"),
        ("respondent_rebuttal_documents", "Exhibit R-3 - LOI.txt"),
    ]
    assert request.warnings == ()
    assert request.summary()["attachments"][0]["file"] == "Exhibit A-1 - POS Ledger.txt"


def test_two_phase_request_has_no_rebuttal_fields(case_bundle):
    request = build_simulation_request(
        case_bundle,
        {Stage.CLAIM: human_filing(Stage.CLAIM), Stage.ANSWER: human_filing(Stage.ANSWER)},
    )
    assert (
        "plaintiff_rebuttal" not in request.fields and "respondent_rebuttal" not in request.fields
    )
    assert all(
        a.field in ("plaintiff_documents", "respondent_documents") for a in request.attachments
    )


def test_simulation_request_needs_claim_and_answer(case_bundle):
    with pytest.raises(SimulationInputError, match="Answer"):
        build_simulation_request(case_bundle, {Stage.CLAIM: human_filing(Stage.CLAIM)})


def test_render_for_simulator_plain_filing():
    assert (
        render_for_simulator(Filing(stage=Stage.REPLY, argument="  Just this.  ")) == "Just this."
    )


def test_extract_award_section():
    section = extract_award_section(SAMPLE_AWARD)
    assert section.startswith("Respondent, SolidEdge, Inc., is ordered to pay")
    assert "CITATIONS" not in section
    assert extract_award_section("no award here") is None
    assert extract_award_section("AWARD: Claim denied.\nCITATIONS:\n1. x") == "Claim denied."


def test_submit_await_and_save_simulation(tmp_path, case_bundle):
    def handler(request):
        if request.method == "POST":
            return httpx.Response(201, json=simulation_json())
        if request.url.path.endswith("/result"):
            return httpx.Response(
                200,
                json={
                    "id": "case_sim1",
                    "case_id": "case_sim1",
                    "text": SAMPLE_AWARD,
                    "published_at": "2026-10-02T00:00:00Z",
                    "pdf_url": "https://x/a.pdf",
                    "page_url": "https://x/sim",
                },
            )
        return httpx.Response(200, json=simulation_json("ready", result_url="https://x/a.pdf"))

    client = make_client(handler)
    thread = {stage: human_filing(stage) for stage in Stage}
    request = build_simulation_request(case_bundle, thread)
    record = submit_simulation(client, request, "idem-9")
    assert record.simulation_id == "case_sim1" and record.status == "processing"
    result = await_award(client, record, timeout=10, interval=0)
    assert record.status == "ready" and record.pdf_url == "https://x/a.pdf"
    assert record.award_section.startswith("Respondent, SolidEdge")
    save_simulation(tmp_path, record, award_text=result.text)
    loaded = load_simulation(tmp_path)
    assert isinstance(loaded, SimulationRecord) and loaded.simulation_id == "case_sim1"
    assert (tmp_path / "award.txt").read_text().startswith("ARBITRATION DECISION")


def test_await_award_records_failure(case_bundle):
    def handler(request):
        return httpx.Response(
            200, json=simulation_json("failed", id="case_f", message="model error")
        )

    client = make_client(handler)
    record = SimulationRecord(
        simulation_id="case_f",
        status="processing",
        base_url=client.base_url,
        idempotency_key="k",
        submitted_at=datetime.now(UTC),
    )
    with pytest.raises(SimulationFailed, match="model error"):
        await_award(client, record, timeout=5, interval=0)
    assert record.status == "failed" and record.message == "model error"


# -------------------------------------------------------------------------- live


def live_case(**overrides) -> ArbitrationCase:
    base = {
        "id": "case_1",
        "status": "awaiting_response",
        "role": "respondent",
        "current_turn": "respondent",
        "next_round": 1,
        "accepted_fields": [
            "affirmation",
            "argument",
            "counterclaim_amount_usd",
            "counterclaim_argument",
            "counterclaim_other_relief",
            "evidence",
            "evidence_demands",
            "evidence_response",
        ],
        "action_required": True,
        "next_action": "respond",
        "question_for_arbitration": "Q",
        "financial_demand_usd": "3500.00",
        "argument": "The deposit was due back.",
        "test": True,
    }
    base.update(overrides)
    return ArbitrationCase.model_validate(base)


def test_thread_from_api_and_stage_to_file():
    case = live_case()
    assert stage_to_file(case) is Stage.ANSWER
    responses = [
        CaseResponse(
            id="case_1:r1",
            case_id="case_1",
            round=1,
            submitted_by="respondent",
            argument="Forfeited under 4.2.",
            counterclaim_argument="Cleaning costs.",
            counterclaim_amount_usd="1200.00",
            evidence_demands="March statement.",
        ),
    ]
    thread = thread_from_api(case, responses)
    assert thread[Stage.CLAIM].argument == "The deposit was due back."
    assert thread[Stage.ANSWER].counterclaim.amount_usd == 1200
    assert thread[Stage.ANSWER].evidence_demands == "March statement."

    with pytest.raises(LiveCaseError, match="claimant's turn"):
        stage_to_file(live_case(current_turn="claimant"))
    with pytest.raises(LiveCaseError, match="not awaiting a response"):
        stage_to_file(
            live_case(
                status="awaiting_signature",
                next_round=None,
                next_action="sign_terms",
                action_url="https://x/terms",
            )
        )


def test_response_form_requires_affirmation_and_accepted_fields():
    filing = Filing(
        stage=Stage.ANSWER,
        argument="Denied.",
        counterclaim=Counterclaim(argument="Owed.", amount_usd="1200"),
    )
    with pytest.raises(LiveCaseError, match="affirmation"):
        response_form(filing, ["argument", "affirmation"], affirmed=False)
    fields = response_form(filing, live_case().accepted_fields, affirmed=True)
    assert fields == {
        "argument": "Denied.",
        "affirmation": "true",
        "counterclaim_argument": "Owed.",
        "counterclaim_amount_usd": "1200.00",
    }
    with pytest.raises(LiveCaseError, match="does not accept 'counterclaim_argument'"):
        response_form(filing, ["argument", "affirmation", "evidence"], affirmed=True)
