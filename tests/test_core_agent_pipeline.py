import json
from decimal import Decimal

import pytest
from conftest import ScriptedClient, human_filing, seed_thread, tagged_reply

from caseagent.core.agent import CaseAgent
from caseagent.core.llm import ChatRequest, ChatResponse, LLMTruncated
from caseagent.core.models import Author, Role, Stage
from caseagent.core.options import ComposeOptions, Depth
from caseagent.core.parsing import FormatError
from caseagent.core.pipeline import PlanError, continue_thread, remaining_stages
from caseagent.core.runs import finish_run, load_run, render_transcript, resolve_run, start_run
from caseagent.core.thread import ThreadEntry, ThreadError, ThreadStore
from caseagent.core.view import build_stage_view
from caseagent.llm.fake import FakeClient


def test_thorough_compose_runs_brief_draft_review(case_bundle):
    client = ScriptedClient(
        [
            "BRIEF",
            tagged_reply(Stage.ANSWER, counterclaim=True),
            tagged_reply(Stage.ANSWER, counterclaim=True, extra=" Revised."),
        ]
    )
    agent = CaseAgent(client, ComposeOptions())
    view = build_stage_view(case_bundle, Stage.ANSWER, {Stage.CLAIM: human_filing(Stage.CLAIM)})
    result = agent.compose(view)

    assert [step.name for step in result.steps] == ["brief", "draft", "review"]
    assert result.brief == "BRIEF"
    assert result.filing.argument.endswith("Revised.")
    assert result.filing.counterclaim.amount_usd == Decimal("42000.00")
    assert result.notes_for_principal == "Verify the opening date."
    assert result.usage.calls == 3 and result.usage.cost_usd == pytest.approx(0.03)
    # Every call shares the same cacheable prefix.
    assert len({(r.system, r.context) for r in client.requests}) == 1
    assert "BRIEF" in client.requests[1].instruction
    assert "<draft>" in client.requests[2].instruction


def test_quick_depth_is_one_call_and_repairs_bad_format(case_bundle):
    client = ScriptedClient(["Sorry, here is prose without tags.", tagged_reply(Stage.CLAIM)])
    agent = CaseAgent(client, ComposeOptions(depth=Depth.QUICK))
    result = agent.compose(build_stage_view(case_bundle, Stage.CLAIM, {}))
    assert [step.name for step in result.steps] == ["draft", "draft-repair"]
    assert "Correction needed" in client.requests[1].instruction
    assert "prose without tags" in client.requests[1].instruction
    assert result.filing.stage is Stage.CLAIM


def test_repair_failure_raises(case_bundle):
    client = ScriptedClient(["no tags", "still no tags"])
    agent = CaseAgent(client, ComposeOptions(depth=Depth.QUICK))
    with pytest.raises(FormatError):
        agent.compose(build_stage_view(case_bundle, Stage.CLAIM, {}))


def test_unusable_review_keeps_draft_with_warning(case_bundle):
    client = ScriptedClient(["BRIEF", tagged_reply(Stage.CLAIM), "garbage", "more garbage"])
    agent = CaseAgent(client, ComposeOptions())
    result = agent.compose(build_stage_view(case_bundle, Stage.CLAIM, {}))
    assert result.filing.argument.startswith("AGENT CLAIM")
    assert any("review pass" in warning for warning in result.warnings)
    assert [step.name for step in result.steps] == ["brief", "draft", "review", "review-repair"]


def test_truncation_propagates(case_bundle):
    class Truncating:
        model = "t"

        def complete(self, request: ChatRequest, on_progress=None) -> ChatResponse:
            raise LLMTruncated("cut")

    with pytest.raises(LLMTruncated):
        CaseAgent(Truncating(), ComposeOptions(depth=Depth.QUICK)).compose(
            build_stage_view(case_bundle, Stage.CLAIM, {})
        )


def test_events_are_emitted(case_bundle):
    events = []
    client = ScriptedClient([tagged_reply(Stage.CLAIM)])
    CaseAgent(client, ComposeOptions(depth=Depth.QUICK), on_event=events.append).compose(
        build_stage_view(case_bundle, Stage.CLAIM, {})
    )
    assert [(e.step, e.phase) for e in events] == [
        ("draft", "start"),
        ("draft", "progress"),
        ("draft", "end"),
    ]


def test_questions_and_ready(case_bundle):
    client = ScriptedClient(
        [
            "<question>When did you open?</question><question>Do you have the license?</question>",
            "<ready/>",
        ]
    )
    agent = CaseAgent(client, ComposeOptions())
    view = build_stage_view(case_bundle, Stage.CLAIM, {})
    assert agent.questions(view, max_questions=5) == [
        "When did you open?",
        "Do you have the license?",
    ]
    assert "questions for BevMart Inc." in client.requests[0].instruction
    assert agent.questions(view, max_questions=5, follow_up=True) == []
    assert "answered your questions" in client.requests[1].instruction


# ------------------------------------------------------------------------- thread


def test_thread_store_appends_in_order(case_bundle):
    store = ThreadStore(case_bundle.root)
    assert store.entries() == [] and store.next_stage() is Stage.CLAIM
    with pytest.raises(ThreadError, match="next filing due is Claim"):
        store.append(ThreadEntry(filing=human_filing(Stage.ANSWER), author=Author.HUMAN))
    seed_thread(case_bundle, Stage.CLAIM, Stage.ANSWER)
    assert [e.stage for e in store.entries()] == [Stage.CLAIM, Stage.ANSWER]
    assert store.next_stage() is Stage.REPLY
    assert (store.filings_dir / "2-answer.txt").read_text().startswith("HUMAN ANSWER")
    assert store.remove_last().stage is Stage.ANSWER
    assert store.next_stage() is Stage.ANSWER
    assert store.clear() == 1 and store.entries() == []


def test_thread_store_rejects_gaps(case_bundle):
    store = seed_thread(case_bundle, Stage.CLAIM, Stage.ANSWER)
    (store.filings_dir / "1-claim.json").unlink()
    with pytest.raises(ThreadError, match="contiguous"):
        store.entries()


def test_remaining_stages(case_bundle):
    store = seed_thread(case_bundle, Stage.CLAIM)
    assert remaining_stages(store) == [Stage.ANSWER, Stage.REPLY, Stage.RESPONSE]
    assert remaining_stages(store, Stage.ANSWER) == [Stage.ANSWER]
    with pytest.raises(PlanError, match="already has the Claim"):
        remaining_stages(store, Stage.CLAIM)
    seed_thread(case_bundle, Stage.ANSWER, Stage.REPLY, Stage.RESPONSE)
    assert remaining_stages(store) == []


def test_continue_thread_plays_each_side_against_the_real_record(case_bundle):
    store = seed_thread(case_bundle, Stage.CLAIM)
    claimant = ScriptedClient([tagged_reply(Stage.REPLY)])
    respondent = ScriptedClient(
        [tagged_reply(Stage.ANSWER, counterclaim=True), tagged_reply(Stage.RESPONSE)]
    )
    agents = {
        Role.CLAIMANT: CaseAgent(claimant, ComposeOptions(depth=Depth.QUICK)),
        Role.RESPONDENT: CaseAgent(respondent, ComposeOptions(depth=Depth.QUICK)),
    }
    seen = []
    written = continue_thread(
        case_bundle,
        store,
        agents,
        {Role.CLAIMANT: "model-c", Role.RESPONDENT: "model-r"},
        on_stage=lambda stage, phase: seen.append((stage, phase)),
    )
    assert [entry.stage for entry in written] == [Stage.ANSWER, Stage.REPLY, Stage.RESPONSE]
    assert [entry.model for entry in written] == ["model-r", "model-c", "model-r"]
    thread = store.thread()
    assert thread[Stage.CLAIM].argument.startswith("HUMAN CLAIM")
    assert thread[Stage.ANSWER].counterclaim is not None
    # The claimant's Reply saw the agent-written Answer, including its counterclaim.
    assert "AGENT ANSWER" in claimant.requests[0].context
    assert "Unpaid rent is owed" in claimant.requests[0].context
    # The respondent's Response saw the Reply; nobody saw the other side's brief.
    assert "AGENT REPLY" in respondent.requests[1].context
    assert "private_brief" not in respondent.requests[1].context
    assert seen[0] == (Stage.ANSWER, "start") and seen[-1] == (Stage.RESPONSE, "end")
    assert store.usage().calls == 3


def test_continue_thread_needs_an_agent_for_each_side(case_bundle):
    store = seed_thread(case_bundle, Stage.CLAIM)
    with pytest.raises(PlanError, match="No agent is playing the claimant"):
        continue_thread(
            case_bundle,
            store,
            {Role.RESPONDENT: CaseAgent(FakeClient(), ComposeOptions(depth=Depth.QUICK))},
            {Role.RESPONDENT: "fake:demo"},
        )
    # Limiting to the Answer only needs the respondent.
    written = continue_thread(
        case_bundle,
        store,
        {Role.RESPONDENT: CaseAgent(FakeClient(), ComposeOptions(depth=Depth.QUICK))},
        {Role.RESPONDENT: "fake:demo"},
        through=Stage.ANSWER,
    )
    assert [entry.stage for entry in written] == [Stage.ANSWER]


def test_run_copies_the_case_thread_and_leaves_the_case_alone(case_bundle, tmp_path):
    case_store = seed_thread(case_bundle, Stage.CLAIM)
    models = {Role.CLAIMANT: "fake:demo", Role.RESPONDENT: "fake:demo"}
    record, store = start_run(
        case_bundle, tmp_path / "runs", models, ComposeOptions(), label="demo"
    )
    assert record.copied_filings == 1 and store.next_stage() is Stage.ANSWER
    agents = {role: CaseAgent(FakeClient(), ComposeOptions()) for role in Role}
    continue_thread(case_bundle, store, agents, models)
    finish_run(record, case_bundle, store)

    assert store.root.name.endswith("-lease-demo")
    assert (store.root / "filings" / "4-response.txt").read_text().startswith("[PLACEHOLDER")
    assert (store.root / "work" / "2-answer" / "1-brief.task.txt").exists()
    assert (store.root / "work" / "2-answer" / "brief.md").exists()
    assert not (store.root / "work" / "1-claim").exists()  # a human filing leaves no work files
    assert case_store.next_stage() is Stage.ANSWER  # the case folder was not advanced

    loaded = load_run(store.root)
    assert loaded.models == models and loaded.usage.calls == 9
    assert loaded.describe_sides() == "claimant=fake:demo respondent=fake:demo"
    transcript = (store.root / "transcript.md").read_text()
    assert (
        "## Claim: BevMart Inc. (claimant)" in transcript and "provided by a person" in transcript
    )
    assert "written by the agent (fake:demo)" in transcript
    assert render_transcript(case_bundle, store.entries()).startswith("# Lease Commencement")
    assert resolve_run("latest", tmp_path / "runs") == store.root
    assert resolve_run(record.run_id, tmp_path / "runs") == store.root
    payload = json.loads((store.root / "run.json").read_text())
    assert payload["case_key"] == "lease"
