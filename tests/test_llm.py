import json
import time

import httpx
import pytest

from caseagent.core.llm import ChatRequest, LLMError, LLMRefused, LLMTruncated
from caseagent.llm.catalog import Catalog, ModelInfo, load_catalog
from caseagent.llm.factory import ModelOptions, build_client, model_label, resolve_params
from caseagent.llm.openai_compat import OpenAICompatClient, RequestParams
from caseagent.llm.providers import OPENAI, OPENROUTER, split_model_spec

REQUEST = ChatRequest(system="SYS", context="CONTEXT " * 10, instruction="TASK", label="t")


def sse(*events: dict, done: bool = True) -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    for event in events:
        lines += ["data: " + json.dumps(event), ""]
    if done:
        lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def chunk(
    content: str | None = None, finish: str | None = None, usage: dict | None = None, **extra
):
    delta = {"role": "assistant", "content": content if content is not None else ""}
    delta.update(extra)
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    if finish:
        choice["native_finish_reason"] = finish
    event = {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [choice],
    }
    if usage is not None:
        event["usage"] = usage
    return event


def make_client(
    handler, provider=OPENROUTER, model="anthropic/claude-opus-5.5", params=None, **kwargs
):
    transport = httpx.MockTransport(handler)
    return OpenAICompatClient(
        provider,
        model,
        "key",
        params or RequestParams(),
        http_client=httpx.Client(transport=transport),
        max_retries=0,
        **kwargs,
    )


def test_streams_content_usage_and_cost():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["authorization"]
        body = sse(
            chunk("<argument>Hel", reasoning="thinking..."),
            chunk("lo</argument>", finish="stop"),
            chunk(
                finish="stop",
                usage={
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "cost": 0.0012,
                    "prompt_tokens_details": {"cached_tokens": 80},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
            ),
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    progress = []
    client = make_client(
        handler,
        params=RequestParams(
            max_tokens=30000, effort="high", cache_prompt=True, context_length=200000
        ),
        session_id="sess-1",
    )
    response = client.complete(REQUEST, on_progress=progress.append)

    assert response.text == "<argument>Hello</argument>"
    assert response.finish_reason == "stop"
    assert response.usage.prompt_tokens == 100 and response.usage.cached_tokens == 80
    assert response.usage.reasoning_tokens == 5 and response.usage.cost_usd == pytest.approx(0.0012)
    assert progress and progress[-1] == len("<argument>Hello</argument>") + len("thinking...")

    body = captured["body"]
    assert captured["auth"] == "Bearer key"
    assert body["model"] == "anthropic/claude-opus-5.5" and body["stream"] is True
    assert body["max_tokens"] == 30000 and "max_completion_tokens" not in body
    assert body["reasoning"] == {"effort": "high"} and body["session_id"] == "sess-1"
    assert "temperature" not in body
    system, user = body["messages"]
    assert system == {"role": "system", "content": "SYS"}
    assert user["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert user["content"][0]["text"].startswith("CONTEXT") and user["content"][1]["text"] == "TASK"


def test_plain_string_content_without_cache_and_openai_dialect():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                chunk("ok", finish="stop"),
                chunk(finish="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}),
            ),
        )

    client = make_client(
        handler,
        provider=OPENAI,
        model="gpt-6.1-sol",
        params=RequestParams(max_tokens=1000, effort="high", temperature=0.2),
    )
    client.complete(REQUEST)
    body = captured["body"]
    assert body["max_completion_tokens"] == 1000 and "max_tokens" not in body
    assert body["reasoning_effort"] == "high" and "reasoning" not in body
    assert body["temperature"] == 0.2
    assert isinstance(body["messages"][1]["content"], str)
    assert body["messages"][1]["content"].endswith("\n\nTASK")


def test_length_refusal_and_empty_replies_raise():
    def length_handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                chunk("partial", finish="length"),
                chunk(
                    finish="length",
                    usage={
                        "prompt_tokens": 5,
                        "completion_tokens": 50,
                        "completion_tokens_details": {"reasoning_tokens": 49},
                    },
                ),
            ),
        )

    with pytest.raises(LLMTruncated, match="49 of 50"):
        make_client(length_handler, params=RequestParams(max_tokens=50)).complete(REQUEST)

    def refusal_handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(chunk("", finish="refusal")),
        )

    with pytest.raises(LLMRefused):
        make_client(refusal_handler).complete(REQUEST)

    def empty_handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(chunk("", finish="stop")),
        )

    with pytest.raises(LLMError, match="empty reply"):
        make_client(empty_handler).complete(REQUEST)


def test_status_errors_are_explained():
    def handler(request):
        return httpx.Response(
            401, json={"error": {"code": 401, "message": "No auth credentials found"}}
        )

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_client(handler).complete(REQUEST)

    def not_found(request):
        return httpx.Response(404, json={"error": {"code": 404, "message": "No endpoints found"}})

    with pytest.raises(LLMError, match="caseagent models"):
        make_client(not_found).complete(REQUEST)

    def mid_stream_error(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(
                {"error": {"code": 502, "message": "Provider returned error"}, "choices": []}
            ),
        )

    with pytest.raises(LLMError, match="Provider returned error"):
        make_client(mid_stream_error).complete(REQUEST)


def test_materials_too_large_for_context_window():
    client = make_client(
        lambda request: httpx.Response(500), params=RequestParams(context_length=1000)
    )
    big = ChatRequest(system="s", context="x" * 10000, instruction="t", label="big")
    with pytest.raises(LLMError, match="Nothing was truncated"):
        client.complete(big)


def test_mid_stream_transport_error_is_retried_once():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadError("connection reset")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(chunk("fine", finish="stop")),
        )

    assert make_client(handler).complete(REQUEST).text == "fine"
    assert calls["n"] == 2


# ----------------------------------------------------------------- catalog/factory

RAW_MODEL = {
    "id": "anthropic/claude-opus-5.5",
    "name": "Anthropic: Claude Opus 5.5",
    "created": 1790094732,
    "context_length": 1000000,
    "pricing": {"prompt": "0.000004", "completion": "0.00002"},
    "top_provider": {"context_length": 1000000, "max_completion_tokens": 128000},
    "supported_parameters": ["max_tokens", "reasoning", "temperature"],
    "reasoning": {
        "mandatory": True,
        "supported_efforts": ["max", "xhigh", "high", "medium", "low"],
        "default_effort": "high",
    },
    "architecture": {"output_modalities": ["text"]},
}
RAW_PLAIN = {
    "id": "meta-llama/llama-4-maverick",
    "name": "Meta: Llama 4 Maverick",
    "created": 1,
    "context_length": 1048576,
    "pricing": {"prompt": "0.00000019", "completion": "0.00000065"},
    "top_provider": {"max_completion_tokens": 16384},
    "supported_parameters": ["max_tokens", "temperature"],
    "architecture": {"output_modalities": ["text"]},
}


def test_model_info_and_effort_semantics():
    info = ModelInfo.from_api(RAW_MODEL)
    assert info.prompt_price == pytest.approx(4.0) and info.completion_price == pytest.approx(20.0)
    assert info.max_output_tokens == 128000
    assert info.accepts_effort("high") and not info.accepts_effort("minimal")
    assert not ModelInfo.from_api(RAW_PLAIN).accepts_effort("high")
    any_effort = ModelInfo.from_api({**RAW_MODEL, "reasoning": {"supported_efforts": None}})
    assert any_effort.accepts_effort("whatever")
    no_selection = ModelInfo.from_api({**RAW_MODEL, "reasoning": {"mandatory": True}})
    assert not no_selection.accepts_effort("high")


def test_catalog_search_get_and_cache(tmp_path):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    RAW_MODEL,
                    RAW_PLAIN,
                    {
                        **RAW_PLAIN,
                        "id": "x/image-only",
                        "architecture": {"output_modalities": ["image"]},
                    },
                ]
            },
        )

    catalog = load_catalog(tmp_path, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert set(catalog.models) == {"anthropic/claude-opus-5.5", "meta-llama/llama-4-maverick"}
    assert catalog.get("anthropic/claude-opus-5.5:nitro").id == "anthropic/claude-opus-5.5"
    assert [m.id for m in catalog.search("claude opus")] == ["anthropic/claude-opus-5.5"]
    assert catalog.suggest("anthropic/claude-opus-5") == ["anthropic/claude-opus-5.5"]

    # Served from cache next time, even with the network gone.
    def offline(request):
        raise httpx.ConnectError("offline")

    cached = load_catalog(tmp_path, client=httpx.Client(transport=httpx.MockTransport(offline)))
    assert cached is not None and set(cached.models) == set(catalog.models) and not cached.stale
    stale = load_catalog(
        tmp_path, refresh=True, client=httpx.Client(transport=httpx.MockTransport(offline))
    )
    assert stale.stale is True


def test_split_model_spec_and_labels():
    assert split_model_spec("anthropic/claude-opus-5.5") == (
        "openrouter",
        "anthropic/claude-opus-5.5",
    )
    assert split_model_spec("qwen/qwen3.8-27b:free") == ("openrouter", "qwen/qwen3.8-27b:free")
    assert split_model_spec("openai:gpt-6.1-sol") == ("openai", "gpt-6.1-sol")
    assert split_model_spec("fake:demo") == ("fake", "demo")
    assert model_label("openrouter:anthropic/claude-opus-5.5") == "anthropic/claude-opus-5.5"
    assert model_label("openai:gpt-6.1-sol") == "openai:gpt-6.1-sol"


def test_resolve_params_fits_the_model():
    info = ModelInfo.from_api(RAW_MODEL)
    params, notes = resolve_params(OPENROUTER, info.id, info, ModelOptions())
    assert params.max_tokens == 64000 and params.effort == "high" and params.cache_prompt
    assert params.context_length == 1000000 and notes == []

    plain = ModelInfo.from_api(RAW_PLAIN)
    params, notes = resolve_params(
        OPENROUTER,
        plain.id,
        plain,
        ModelOptions(effort="high", effort_explicit=True, max_tokens=200000),
    )
    assert params.max_tokens == 16384 and params.effort is None and not params.cache_prompt
    assert any("does not accept effort=high" in note for note in notes)
    assert any("caps output" in note for note in notes)

    # No catalog data: nothing is assumed; explicit requests are honored.
    params, _ = resolve_params(OPENROUTER, "unknown/model", None, ModelOptions())
    assert params.max_tokens is None and params.effort is None
    params, _ = resolve_params(
        OPENROUTER,
        "unknown/model",
        None,
        ModelOptions(effort="low", effort_explicit=True, max_tokens=5000),
    )
    assert params.effort == "low" and params.max_tokens == 5000
    params, _ = resolve_params(OPENROUTER, info.id, info, ModelOptions(effort="default"))
    assert params.effort is None


def test_build_client_validates_key_and_catalog(monkeypatch):
    catalog = Catalog([ModelInfo.from_api(RAW_MODEL)], fetched_at=time.time())
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        build_client("anthropic/claude-opus-5.5", ModelOptions(), catalog)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    with pytest.raises(LLMError, match="Did you mean anthropic/claude-opus-5.5"):
        build_client("anthropic/claude-opus-5", ModelOptions(), catalog)
    built = build_client("anthropic/claude-opus-5.5", ModelOptions(), catalog)
    assert built.label == "anthropic/claude-opus-5.5" and built.info is not None
    fake = build_client("fake:demo", ModelOptions(), catalog)
    assert fake.client.model == "fake:demo"
