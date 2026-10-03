"""Chat client for OpenAI-compatible endpoints, OpenRouter first among them.

Requests are streamed. Filings from reasoning models can take minutes, and a
streamed response keeps the connection alive and lets the CLI show progress.
Nothing is sent that the caller did not ask for: no temperature, no reasoning
settings, and no output cap unless ``RequestParams`` carries one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
import openai

from ..core.llm import (
    ChatRequest,
    ChatResponse,
    LLMError,
    LLMRefused,
    LLMTruncated,
    ProgressCallback,
    Usage,
)
from .providers import Provider

# Sizing estimate only. Legal prose runs near 4 characters per token; 3 leaves
# headroom for tokenizers that split more finely.
CHARS_PER_TOKEN = 3.0
CONTEXT_MARGIN_TOKENS = 2_048
MIN_OUTPUT_TOKENS = 4_096
STREAM_ATTEMPTS = 2


@dataclass(frozen=True)
class RequestParams:
    """Per-model request settings, already checked against what the model accepts."""

    max_tokens: int | None = None
    effort: str | None = None
    temperature: float | None = None
    cache_prompt: bool = False
    context_length: int | None = None


def estimate_tokens(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN)


class OpenAICompatClient:
    def __init__(
        self,
        provider: Provider,
        model_id: str,
        api_key: str | None,
        params: RequestParams | None = None,
        *,
        label: str | None = None,
        session_id: str | None = None,
        timeout: float = 900.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.provider = provider
        self.model_id = model_id
        self.model = label or model_id
        self.params = params or RequestParams()
        self.session_id = session_id
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed",
            base_url=provider.base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            max_retries=max_retries,
            http_client=http_client,
        )

    def complete(
        self, request: ChatRequest, on_progress: ProgressCallback | None = None
    ) -> ChatResponse:
        kwargs = self._request_kwargs(request)
        started = time.monotonic()
        last_error: Exception | None = None
        for _attempt in range(STREAM_ATTEMPTS):
            try:
                response = self._stream(kwargs, on_progress)
            except (openai.APIConnectionError, httpx.TransportError) as exc:
                # The SDK already retried connecting; this covers a stream that
                # dropped partway through.
                last_error = exc
                continue
            except openai.APIStatusError as exc:
                raise self._status_error(exc) from exc
            except openai.APIError as exc:
                raise LLMError(f"{self.model}: the provider reported an error: {exc}") from exc
            response.seconds = time.monotonic() - started
            return self._checked(response, kwargs)
        raise LLMError(
            f"{self.model}: could not reach {self.provider.name} ({last_error}). "
            "Check your connection and try again."
        ) from last_error

    # ------------------------------------------------------------------ request

    def _request_kwargs(self, request: ChatRequest) -> dict[str, Any]:
        params = self.params
        max_tokens = params.max_tokens
        if params.context_length:
            prompt_tokens = estimate_tokens(request.char_count)
            room = params.context_length - prompt_tokens - CONTEXT_MARGIN_TOKENS
            if room < MIN_OUTPUT_TOKENS:
                raise LLMError(
                    f"The materials for {request.label or 'this request'} are roughly "
                    f"{prompt_tokens:,} tokens, which does not fit {self.model}'s "
                    f"{params.context_length:,}-token context window with room to answer. "
                    "Pick a model with a larger window, or take exhibits out of this phase's "
                    "evidence list in case.json. Nothing was truncated."
                )
            if max_tokens:
                max_tokens = min(max_tokens, room)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._messages(request),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            kwargs[self.provider.max_tokens_param] = max_tokens
        if params.temperature is not None:
            kwargs["temperature"] = params.temperature

        extra_body: dict[str, Any] = {}
        if params.effort:
            if self.provider.reasoning_style == "openrouter":
                extra_body["reasoning"] = {"effort": params.effort}
            elif self.provider.reasoning_style == "openai":
                kwargs["reasoning_effort"] = params.effort
        if self.session_id and self.provider.session_id:
            extra_body["session_id"] = self.session_id
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _messages(self, request: ChatRequest) -> list[dict[str, Any]]:
        if self.params.cache_prompt and self.provider.cache_control:
            # The breakpoint caches the system prompt and the case materials, which
            # are identical across the brief, draft, and review calls for a filing.
            content: Any = [
                {
                    "type": "text",
                    "text": request.context,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": request.instruction},
            ]
        else:
            content = f"{request.context}\n\n{request.instruction}"
        return [
            {"role": "system", "content": request.system},
            {"role": "user", "content": content},
        ]

    # ----------------------------------------------------------------- response

    def _stream(self, kwargs: dict[str, Any], on_progress: ProgressCallback | None) -> ChatResponse:
        parts: list[str] = []
        received = 0
        finish_reason: str | None = None
        native_finish_reason: str | None = None
        model_name = self.model
        usage = Usage(calls=1)

        with self._client.chat.completions.create(**kwargs) as stream:
            for chunk in stream:
                if chunk.usage is not None:
                    usage = _usage(chunk.usage)
                for choice in chunk.choices or []:
                    delta = choice.delta
                    text = getattr(delta, "content", None) or ""
                    # OpenRouter streams the model's reasoning summary in a separate
                    # field; it counts as progress but is not part of the reply.
                    reasoning = getattr(delta, "reasoning", None) or ""
                    if text:
                        parts.append(text)
                    if text or reasoning:
                        received += len(text) + len(reasoning)
                        if on_progress is not None:
                            on_progress(received)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
                    native = getattr(choice, "native_finish_reason", None)
                    if native:
                        native_finish_reason = native

        response = ChatResponse(
            text="".join(parts).strip(),
            model=model_name,
            finish_reason=finish_reason,
            usage=usage,
        )
        if native_finish_reason == "refusal":
            response.finish_reason = "refusal"
        return response

    def _checked(self, response: ChatResponse, kwargs: dict[str, Any]) -> ChatResponse:
        reason = response.finish_reason
        if reason in ("refusal", "content_filter"):
            raise LLMRefused(
                f"{self.model} declined to answer (finish reason: {reason}). "
                "Try a different model for this side."
            )
        if reason == "length":
            cap = kwargs.get(self.provider.max_tokens_param)
            limit = f"the {cap:,}-token output limit" if cap else "its output limit"
            raise LLMTruncated(
                f"{self.model} hit {limit} before finishing "
                f"({response.usage.reasoning_tokens:,} of {response.usage.completion_tokens:,} "
                "output tokens went to reasoning). Raise --max-tokens or lower --effort."
            )
        if not response.text:
            raise LLMError(
                f"{self.model} returned an empty reply (finish reason: {reason or 'none'})."
            )
        return response

    def _status_error(self, exc: openai.APIStatusError) -> LLMError:
        detail = _error_detail(exc)
        status = exc.status_code
        key_hint = self.provider.api_key_env[0] if self.provider.api_key_env else "the API key"
        if status == 401:
            return LLMError(
                f"{self.provider.name} rejected the API key (401: {detail}). Check {key_hint}."
            )
        if status == 402:
            return LLMError(f"{self.provider.name} reports insufficient credits (402: {detail}).")
        if status == 404:
            return LLMError(
                f"{self.model} is not available on {self.provider.name} (404: {detail}). "
                "Run `caseagent models` to see what is."
            )
        if status == 429:
            return LLMError(
                f"{self.provider.name} is rate limiting {self.model} (429: {detail}). "
                "Wait a moment and retry."
            )
        return LLMError(f"{self.model}: {self.provider.name} returned {status}: {detail}")


def _usage(raw: Any) -> Usage:
    completion_details = getattr(raw, "completion_tokens_details", None)
    prompt_details = getattr(raw, "prompt_tokens_details", None)
    extra = getattr(raw, "model_extra", None) or {}
    cost = extra.get("cost")
    return Usage(
        calls=1,
        prompt_tokens=getattr(raw, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw, "completion_tokens", 0) or 0,
        reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


def _error_detail(exc: openai.APIStatusError) -> str:
    """The provider's own explanation, which is usually more useful than the status."""
    body = exc.body
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        message = error.get("message")
        metadata = error.get("metadata")
        raw = metadata.get("raw") if isinstance(metadata, dict) else None
        if message and raw:
            return f"{message} ({raw})"
        if message:
            return str(message)
    return str(exc.message if hasattr(exc, "message") else exc)
