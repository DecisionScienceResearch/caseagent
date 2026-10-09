import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from conftest import HUMAN_TEXT, Stage, seed_thread
from typer.testing import CliRunner

from caseagent.cli import app
from caseagent.core.options import ComposeOptions
from caseagent.core.runs import finish_run, start_run
from caseagent.core.thread import ThreadStore
from caseagent.decisionlayer.client import DecisionLayerClient

runner = CliRunner()
SRC = Path(__file__).resolve().parents[1] / "src" / "caseagent"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def invoke(args: list[str], **kwargs) -> SimpleNamespace:
    """Run the CLI and return its exit code and output with any terminal styling removed."""
    result = runner.invoke(app, args, **kwargs)
    return SimpleNamespace(exit_code=result.exit_code, output=ANSI_RE.sub("", result.output))


def test_core_does_not_depend_on_api_or_model_implementations():
    """The agent (core) must stay independent of the DecisionLayer API and model vendors."""
    forbidden = {
        "caseagent.decisionlayer",
        "caseagent.llm",
        "caseagent.cli",
        "caseagent.ui",
        "httpx",
        "openai",
        "requests",
    }
    for path in (SRC / "core").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:  # relative import inside core
                    module = (
                        "caseagent.core." + module if node.level == 1 else "caseagent." + module
                    )
                names = [module]
            for name in names:
                assert not any(name == f or name.startswith(f + ".") for f in forbidden), (
                    f"{path.name} imports {name}"
                )


def test_decisionlayer_never_imports_llm():
    for path in (SRC / "decisionlayer").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "caseagent.llm" not in text and "from ..llm" not in text, path.name


def test_nothing_reads_the_sample_arguments_file():
    """The sample's own arguments are reference material; no code path loads them."""
    loaders = ("read_text(", "read_bytes(", "open(", "load(", "loads(")
    for path in SRC.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            mentions = "SAMPLE_ARGUMENTS_FILENAME" in line or "sample-arguments" in line
            if mentions and any(token in line for token in loaders):
                raise AssertionError(f"{path.name} reads the sample arguments file: {line.strip()}")


@pytest.fixture
def env(case_bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("CASEAGENT_CASES_DIR", str(case_bundle.root.parent))
    monkeypatch.setenv("CASEAGENT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("CASEAGENT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("DECISIONLAYER_API_KEY", raising=False)
    monkeypatch.setenv("COLUMNS", "200")  # keep Rich tables on one line
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.chdir(tmp_path)  # no stray .env
    return tmp_path


def test_next_writes_one_filing_at_a_time(env, case_bundle):
    store = ThreadStore(case_bundle.root)
    first = invoke(["next", "lease", "-m", "fake:demo", "--depth", "quick", "--no-interview"])
    assert first.exit_code == 0, first.output
    assert "writing the Claim for BevMart Inc." in first.output
    assert [e.stage for e in store.entries()] == [Stage.CLAIM]
    assert store.entries()[0].model == "fake:demo"

    second = invoke(["reply", "lease", "-m", "fake:demo", "--depth", "quick"])
    assert second.exit_code == 0, second.output
    assert "writing the Answer for Arcology LLC" in second.output
    assert [e.stage for e in store.entries()] == [Stage.CLAIM, Stage.ANSWER]

    wrong = invoke(["start", "lease", "-m", "fake:demo", "--depth", "quick"])
    assert wrong.exit_code == 1 and "already has a Answer" in wrong.output.replace(
        "an Answer", "a Answer"
    )


def test_add_filing_then_next_replies_to_it(env, case_bundle, tmp_path):
    claim = tmp_path / "claim.txt"
    claim.write_text(HUMAN_TEXT[Stage.CLAIM])
    added = invoke(["case", "add-filing", "lease", "claim", str(claim)])
    assert added.exit_code == 0, added.output
    assert "Recorded the Claim for BevMart Inc." in added.output

    result = invoke(["next", "lease", "-m", "fake:demo", "--depth", "quick"])
    assert result.exit_code == 0, result.output
    entries = ThreadStore(case_bundle.root).entries()
    assert [(e.stage, e.author.value) for e in entries] == [
        (Stage.CLAIM, "human"),
        (Stage.ANSWER, "agent"),
    ]
    # The agent's Answer was written against the human Claim.
    materials = (case_bundle.root / "work" / "2-answer" / "materials.txt").read_text()
    assert "HUMAN CLAIM" in materials

    shown = invoke(["case", "show", "lease", "--full"])
    assert (
        shown.exit_code == 0
        and "filed · human" in shown.output
        and "filed · agent (fake:demo)" in shown.output
    )


def test_simulate_dry_run_plays_both_sides_in_a_run_and_leaves_the_case(env, case_bundle, tmp_path):
    claim = tmp_path / "claim.txt"
    claim.write_text(HUMAN_TEXT[Stage.CLAIM])
    invoke(["case", "add-filing", "lease", "claim", str(claim)])

    result = invoke(
        [
            "simulate",
            "lease",
            "-m",
            "fake:demo",
            "--depth",
            "quick",
            "--dry-run",
            "--label",
            "smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "to write · agent (fake:demo)" in result.output
    assert "Submit it with `caseagent submit" in result.output
    run_dirs = list((env / "runs").iterdir())
    assert len(run_dirs) == 1 and run_dirs[0].name.endswith("-lease-smoke")
    run_store = ThreadStore(run_dirs[0])
    assert [(e.stage, e.author.value) for e in run_store.entries()] == [
        (Stage.CLAIM, "human"),
        (Stage.ANSWER, "agent"),
        (Stage.REPLY, "agent"),
        (Stage.RESPONSE, "agent"),
    ]
    record = json.loads((run_dirs[0] / "run.json").read_text())
    assert record["models"] == {"claimant": "fake:demo", "respondent": "fake:demo"}
    assert record["copied_filings"] == 1
    # The case folder still has only the human Claim.
    assert [e.stage for e in ThreadStore(case_bundle.root).entries()] == [Stage.CLAIM]

    shown = invoke(["show", "latest"])
    assert shown.exit_code == 0 and "## Claim: BevMart Inc. (claimant)" in shown.output
    listed = invoke(["runs"])
    assert listed.exit_code == 0 and "lease" in listed.output


def test_simulate_through_answer_only_needs_the_respondent(env, case_bundle, tmp_path):
    claim = tmp_path / "claim.txt"
    claim.write_text(HUMAN_TEXT[Stage.CLAIM])
    invoke(["case", "add-filing", "lease", "claim", str(claim)])
    result = invoke(
        [
            "simulate",
            "lease",
            "--respondent-model",
            "fake:demo",
            "--through",
            "answer",
            "--depth",
            "quick",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    run_dir = next((env / "runs").iterdir())
    assert [e.stage for e in ThreadStore(run_dir).entries()] == [Stage.CLAIM, Stage.ANSWER]
    assert json.loads((run_dir / "run.json").read_text())["models"] == {"respondent": "fake:demo"}


def test_simulate_refuses_fake_output_without_dry_run(env):
    result = invoke(["simulate", "lease", "-m", "fake:demo", "--yes"])
    assert result.exit_code == 1 and "fake model" in result.output


def test_submit_needs_a_production_key(env, monkeypatch):
    invoke(["simulate", "lease", "-m", "fake:demo", "--depth", "quick", "--dry-run"])
    monkeypatch.setenv("DECISIONLAYER_API_KEY", "dvarb_test_abc")
    result = invoke(["submit", "latest", "--yes"])
    assert result.exit_code == 1 and ("test key" in result.output or "fake model" in result.output)


def test_next_on_a_blank_case_needs_a_terminal(env):
    created = invoke(["case", "init", str(env / "my dispute")])
    assert created.exit_code == 0 and (env / "my dispute" / "case.json").exists()
    result = invoke(["next", str(env / "my dispute"), "-m", "fake:demo"])
    assert result.exit_code == 1 and "missing claimant name" in result.output


def test_reset_clears_the_thread(env, case_bundle, tmp_path):
    claim = tmp_path / "claim.txt"
    claim.write_text(HUMAN_TEXT[Stage.CLAIM])
    invoke(["case", "add-filing", "lease", "claim", str(claim)])
    result = invoke(["case", "reset", "lease", "--yes"])
    assert result.exit_code == 0, result.output
    assert ThreadStore(case_bundle.root).entries() == []


def test_case_list_and_missing_case(env):
    listed = invoke(["case", "list"])
    assert listed.exit_code == 0 and "Lease Commencement" in listed.output
    missing = invoke(["case", "show", "nope"])
    assert missing.exit_code == 1 and "No case named" in missing.output


def test_next_on_a_blank_case_asks_for_the_essentials(env, monkeypatch):
    """Interactive intake: the essentials are prompted for, saved, and the Claim written."""
    import caseagent.cli as cli

    monkeypatch.setattr(cli, "interactive", lambda: True)
    invoke(["case", "init", str(env / "deposit"), "--title", "Deposit dispute"])
    answers = "\n".join(
        [
            "Avery Quinn",  # claimant
            "Jordan Chen",  # respondent
            "Jordan kept a $3,500 deposit after cancelling the job.",  # dispute
            "3500",  # amount
            "",  # other relief
            "The deposit is refundable if the work is not delivered within 30 days.",  # contract
            "",  # documents to attach
        ]
    )
    result = invoke(
        ["next", str(env / "deposit"), "-m", "fake:demo", "--depth", "quick"],
        input=answers + "\n",
    )
    assert result.exit_code == 0, result.output
    assert "The agent will ask Avery Quinn a few questions" in result.output
    assert "writing the Claim for Avery Quinn" in result.output
    case = json.loads((env / "deposit" / "case.json").read_text())
    assert case["claimant"]["name"] == "Avery Quinn"
    assert case["financial_demand_usd"] == "3500.00"
    assert case["governing_contract"].startswith("The deposit is refundable")
    assert [e.stage for e in ThreadStore(env / "deposit").entries()] == [Stage.CLAIM]


# ---------------------------------------------------------------------------- tour


def test_tour_prints_every_step_without_running_anything(env):
    """Outside a terminal the tour is a printed walkthrough; --print forces that."""
    for args in (["tour"], ["tour", "--print"]):
        result = invoke(args)
        assert result.exit_code == 0, result.output
        assert "Printing the walkthrough only" in result.output
        for number in range(1, 9):
            assert f"Step {number} of 8" in result.output
        assert "$ caseagent doctor" in result.output
        assert (
            "$ caseagent simulate saas-outage -m fake:demo --depth quick --dry-run" in result.output
        )
        assert "$ caseagent submit latest" in result.output
        assert "Run it?" not in result.output
        assert "Tour complete" in result.output


def test_tour_runs_the_real_commands_and_can_stop_and_resume(env, monkeypatch):
    import caseagent.cli as cli

    monkeypatch.setattr(cli, "interactive", lambda: True)
    # Step 3: run the fake simulation, run `show latest`, then quit at the pause.
    result = invoke(["tour", "--case", "lease", "--from", "3"], input="y\ny\nq\n")
    assert result.exit_code == 0, result.output
    assert "Step 3 of 8" in result.output
    assert "Simulation plan: Lease Commencement" in result.output  # simulate ran
    assert "Not simulated yet." in result.output  # show latest ran
    assert "Stopped at step 3" in result.output
    assert "caseagent tour --from 3" in result.output
    runs = list((env / "runs").iterdir())
    assert len(runs) == 1 and (runs[0] / "transcript.md").is_file()


def test_tour_skips_commands_that_need_a_missing_key(env, monkeypatch):
    import caseagent.cli as cli

    monkeypatch.setattr(cli, "interactive", lambda: True)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # Step 4: next and reply are skipped for lack of a key; decline the other two; quit.
    result = invoke(["tour", "--case", "lease", "--from", "4"], input="n\nn\nq\n")
    assert result.exit_code == 0, result.output
    assert result.output.count("Skipped: OPENROUTER_API_KEY is not set") == 2
    assert "writing the Claim" not in result.output
    assert ThreadStore(env / "cases" / "lease").entries() == []
    assert "Stopped at step 4" in result.output


def test_tour_reports_a_failed_command_and_offers_to_carry_on(env, monkeypatch):
    import caseagent.cli as cli

    monkeypatch.setattr(cli, "interactive", lambda: True)
    result = invoke(["tour", "--case", "nope", "--from", "3"], input="y\nn\n")
    assert result.exit_code == 0, result.output
    assert "No case named" in result.output
    assert "did not succeed (exit 1)" in result.output
    assert "Stopped at step 3" in result.output


def test_tour_from_out_of_range(env):
    result = invoke(["tour", "--from", "9"])
    assert result.exit_code == 1 and "--from must be between 1 and 8" in result.output


# ---------------------------------------------------------------------- live and submit

LIVE_CASE = {
    "id": "case_1",
    "status": "awaiting_response",
    "role": "respondent",
    "current_turn": "respondent",
    "next_round": 1,
    "accepted_fields": ["affirmation", "argument", "evidence", "evidence_tickets"],
    "action_required": True,
    "next_action": "respond",
    "action_url": "https://x/respond",
    "view_url": "https://x/case/case_1",
    "question_for_arbitration": "Q",
    "financial_demand_usd": "3500.00",
    "argument": "The deposit was due back.",
    "respondent_first_name": "Jordan",
    "respondent_last_name": "Chen",
    "test": False,
}

ROUND_ONE = {
    "id": "case_1:r1",
    "case_id": "case_1",
    "round": 1,
    "submitted_by": "respondent",
    "argument": "Forfeited under 4.2.",
    "evidence_files": [{"name": "Exhibit R-1 - Lease.pdf", "kind": "supporting"}],
    "submitted_at": "2026-10-02T00:00:00Z",
    "due_at": "2026-10-16T00:00:00Z",
}

SIGNING = {
    "signing_url": "https://x/sign/abc",
    "expires_at": "2026-10-02T01:00:00Z",
    "provider": "signwell",
}


def mock_api(monkeypatch, handler):
    """Route the CLI's DecisionLayer client to ``handler`` and give it a production key."""
    import caseagent.cli as cli

    def factory(api_key, base_url="https://staging.example", **kwargs):
        return DecisionLayerClient(api_key, base_url, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(cli, "DecisionLayerClient", factory)
    monkeypatch.setenv("DECISIONLAYER_API_KEY", "dvarb_live_x")


def test_live_respond_stops_for_the_signature_and_prints_a_signing_link(env, monkeypatch):
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/api/v1/cases/case_1":
            unsigned = {**LIVE_CASE, "next_action": "sign_terms", "action_url": "https://x/terms"}
            return httpx.Response(200, json=unsigned)
        if request.url.path == "/api/v1/cases/case_1/sign":
            return httpx.Response(200, json=SIGNING)
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    mock_api(monkeypatch, handler)
    result = invoke(["live", "respond", "case_1", "--materials", "lease", "-m", "fake:demo"])
    assert result.exit_code == 1, result.output
    assert "must sign the arbitration terms" in result.output
    assert "https://x/sign/abc" in result.output
    # Nothing was composed: the thread was never even fetched.
    assert ("GET", "/api/v1/cases/case_1/responses") not in calls


def test_live_status_shows_deadline_files_and_links(env, monkeypatch):
    def handler(request):
        if request.url.path == "/api/v1/cases/case_1":
            waiting = {
                **LIVE_CASE,
                "current_turn": "claimant",
                "next_round": 2,
                "next_action": None,
                "action_required": False,
                "action_url": None,
            }
            return httpx.Response(200, json=waiting)
        if request.url.path == "/api/v1/cases/case_1/responses":
            return httpx.Response(200, json=[ROUND_ONE])
        raise AssertionError(request.url.path)

    mock_api(monkeypatch, handler)
    result = invoke(["live", "status", "case_1"])
    assert result.exit_code == 0, result.output
    assert "Response due: 2026-10-16 00:00 UTC" in result.output
    assert "Files: Exhibit R-1 - Lease.pdf (supporting)" in result.output
    assert "Case page: https://x/case/case_1" in result.output
    assert "Respondent: Jordan Chen" in result.output


def test_live_sign_prints_the_signing_link(env, monkeypatch):
    def handler(request):
        assert request.method == "POST" and request.url.path == "/api/v1/cases/case_1/sign"
        return httpx.Response(200, json={**SIGNING, "provider": "hosted"})

    mock_api(monkeypatch, handler)
    result = invoke(["live", "sign", "case_1"])
    assert result.exit_code == 0, result.output
    assert "https://x/sign/abc" in result.output and "hosted" in result.output


def test_submit_discloses_the_public_award_and_saves_it(env, case_bundle, monkeypatch):
    seed_thread(case_bundle, *Stage)
    record, store = start_run(case_bundle, env / "runs", {}, ComposeOptions(), label="as-filed")
    finish_run(record, case_bundle, store)
    award = "ARBITRATION DECISION\nAWARD:\nClaim denied.\nCITATIONS:\n1. Rules\n"
    queued = {
        "id": "case_sim1",
        "status": "processing",
        "page_url": "https://x/sim",
        "created_at": "2026-10-02T00:00:00Z",
    }

    def handler(request):
        if request.method == "POST" and request.url.path == "/api/v1/simulations":
            return httpx.Response(201, json=queued)
        if request.url.path == "/api/v1/simulations/case_sim1/result":
            return httpx.Response(
                200,
                json={
                    "id": "case_sim1",
                    "case_id": "case_sim1",
                    "text": award,
                    "published_at": "2026-10-02T00:00:00Z",
                    "pdf_url": "https://x/a.pdf",
                    "page_url": "https://x/sim",
                },
            )
        if request.url.path == "/api/v1/simulations/case_sim1":
            return httpx.Response(200, json={**queued, "status": "ready"})
        raise AssertionError(f"unexpected {request.method} {request.url.path}")

    mock_api(monkeypatch, handler)
    result = invoke(["submit", "latest", "--yes"])
    assert result.exit_code == 0, result.output
    assert "public link for 30 days" in result.output
    assert "Queued simulation case_sim1" in result.output
    assert "Claim denied." in result.output
    assert (store.root / "award.txt").read_text().startswith("ARBITRATION DECISION")


def test_live_claim_reports_the_result_and_points_to_signing(env, monkeypatch):
    def handler(request):
        assert request.method == "POST" and request.url.path == "/api/v1/cases/case_1/claim"
        assert json.loads(request.content) == {"verification_code": "ABC-123"}
        claimed = {**LIVE_CASE, "next_action": "sign_terms", "action_url": "https://x/terms"}
        return httpx.Response(200, json=claimed)

    mock_api(monkeypatch, handler)
    result = invoke(["live", "claim", "case_1", "ABC-123"])
    assert result.exit_code == 0, result.output
    assert "Claimed case case_1 as the respondent" in result.output
    assert "caseagent live sign case_1" in result.output
    assert "https://x/sign" not in result.output  # it points to the step; it does not take it


def test_live_commands_explain_why_a_case_cannot_be_opened(env, monkeypatch):
    def unclaimed(request):
        error = {"status": 404, "message": "Case not found", "details": [], "reason": "not_claimed"}
        return httpx.Response(404, json={"error": error})

    mock_api(monkeypatch, unclaimed)
    status = invoke(["live", "status", "case_1"])
    assert status.exit_code == 1 and "caseagent live claim case_1 CODE" in status.output
    respond = invoke(["live", "respond", "case_1", "--materials", "lease"])
    assert respond.exit_code == 1 and "caseagent live claim case_1 CODE" in respond.output

    def stranger(request):
        error = {"status": 404, "message": "Case not found", "details": [], "reason": "not_a_party"}
        return httpx.Response(404, json={"error": error})

    mock_api(monkeypatch, stranger)
    result = invoke(["live", "sign", "case_1"])
    assert result.exit_code == 1 and "not a party to case case_1" in result.output


DECISION = {
    "id": "dec_1",
    "case_id": "case_1",
    "text": "ARBITRATION DECISION\nAWARD:\nClaim denied.\nCITATIONS:\n1. Rules\n",
    "published_at": "2026-10-20T00:00:00Z",
    "pdf_url": "https://x/d.pdf",
}


def test_live_decision_prints_and_saves_the_award(env, monkeypatch):
    def handler(request):
        if request.url.path == "/api/v1/cases/case_1/decision":
            return httpx.Response(200, json=DECISION)
        raise AssertionError(request.url.path)

    mock_api(monkeypatch, handler)
    result = invoke(["live", "decision", "case_1"])
    assert result.exit_code == 0, result.output
    assert "Claim denied." in result.output and "PDF: https://x/d.pdf" in result.output
    saved = env / "runs" / "live-case_1"
    assert (saved / "award.txt").read_text().startswith("ARBITRATION DECISION")
    assert json.loads((saved / "decision.json").read_text())["decision_id"] == "dec_1"


def test_live_decision_explains_when_nothing_is_published(env, monkeypatch):
    def handler(request):
        if request.url.path == "/api/v1/cases/case_1/decision":
            error = {
                "status": 404,
                "message": "Not published",
                "details": [],
                "reason": "not_published",
            }
            return httpx.Response(404, json={"error": error})
        if request.url.path == "/api/v1/cases/case_1":
            return httpx.Response(200, json=LIVE_CASE)
        raise AssertionError(request.url.path)

    mock_api(monkeypatch, handler)
    result = invoke(["live", "decision", "case_1"])
    assert result.exit_code == 1, result.output
    assert "No award has been published" in result.output
    assert "awaiting_response" in result.output and "respondent's turn" in result.output


def test_live_status_fetches_the_award_once_decided(env, monkeypatch):
    def handler(request):
        if request.url.path == "/api/v1/cases/case_1":
            decided = {
                **LIVE_CASE,
                "status": "decided",
                "current_turn": None,
                "next_round": None,
                "accepted_fields": [],
                "action_required": False,
                "next_action": None,
                "action_url": None,
            }
            return httpx.Response(200, json=decided)
        if request.url.path == "/api/v1/cases/case_1/responses":
            return httpx.Response(200, json=[ROUND_ONE])
        if request.url.path == "/api/v1/cases/case_1/decision":
            return httpx.Response(200, json=DECISION)
        raise AssertionError(request.url.path)

    mock_api(monkeypatch, handler)
    result = invoke(["live", "status", "case_1"])
    assert result.exit_code == 0, result.output
    assert "status decided" in result.output and "Claim denied." in result.output
    assert (env / "runs" / "live-case_1" / "award.txt").exists()


def test_live_respond_fills_the_respondent_name_and_warns_about_the_contract(env, monkeypatch):
    invoke(["case", "init", str(env / "blank")])  # blank parties, no contract

    def handler(request):
        if request.url.path == "/api/v1/cases/case_1":
            return httpx.Response(200, json=LIVE_CASE)
        if request.url.path == "/api/v1/cases/case_1/responses":
            return httpx.Response(200, json=[])
        raise AssertionError(request.url.path)

    mock_api(monkeypatch, handler)
    result = invoke(
        [
            "live",
            "respond",
            "case_1",
            "--materials",
            str(env / "blank"),
            "-m",
            "fake:demo",
            "--depth",
            "quick",
        ]
    )
    assert result.exit_code == 0, result.output
    assert "Jordan Chen" in result.output  # the respondent's name came from the API
    assert "no claimant name" in result.output
    assert "no contract text or file" in result.output
    assert "Saved draft" in result.output


def test_doctor_reports_whether_the_account_may_file_cases(env, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def handler(request):
        assert request.url.path == "/api/v1/me"
        me = {
            "id": "u1",
            "email": "a@b.c",
            "first_name": "A",
            "last_name": "B",
            "can_file_cases": False,
            "created_at": "2026-01-01T00:00:00Z",
        }
        return httpx.Response(200, json=me)

    mock_api(monkeypatch, handler)
    result = invoke(["doctor"])
    assert result.exit_code == 0, result.output
    assert "production key" in result.output
    assert "Case filing: not approved" in result.output
