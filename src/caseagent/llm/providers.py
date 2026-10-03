"""Where a model is served from.

A model is named ``[provider:]model``. With no prefix the default provider is
used (OpenRouter), so ``anthropic/claude-opus-5.5`` and
``openrouter:anthropic/claude-opus-5.5`` are the same thing. Every provider here
speaks the OpenAI chat-completions protocol; they differ only in the details
captured by ``Provider``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: tuple[str, ...]
    requires_key: bool = True
    # Request-field name for the output cap. OpenAI's own API retired
    # ``max_tokens`` in favor of ``max_completion_tokens``.
    max_tokens_param: str = "max_tokens"
    # How reasoning effort is expressed: OpenRouter's ``reasoning: {effort}``
    # object, OpenAI's top-level ``reasoning_effort``, or not at all.
    reasoning_style: str | None = None
    # Whether Anthropic-style ``cache_control`` breakpoints may be placed on
    # message content parts (OpenRouter passes them through to Claude models).
    cache_control: bool = False
    # Whether the endpoint accepts OpenRouter's ``session_id`` routing hint.
    session_id: bool = False


OPENROUTER = Provider(
    name="openrouter",
    base_url="https://openrouter.ai/api/v1",
    api_key_env=("OPENROUTER_API_KEY",),
    reasoning_style="openrouter",
    cache_control=True,
    session_id=True,
)

OPENAI = Provider(
    name="openai",
    base_url="https://api.openai.com/v1",
    api_key_env=("OPENAI_API_KEY",),
    max_tokens_param="max_completion_tokens",
    reasoning_style="openai",
)

FAKE = "fake"
CUSTOM = "custom"
DEFAULT_PROVIDER = OPENROUTER.name


def custom_provider() -> Provider:
    """An OpenAI-compatible endpoint of your own (Ollama, vLLM, LM Studio, a proxy)."""
    base_url = os.environ.get("CASEAGENT_LLM_BASE_URL", "").strip()
    if not base_url:
        raise ValueError(
            "The custom: provider needs CASEAGENT_LLM_BASE_URL, for example "
            "http://localhost:11434/v1 for Ollama."
        )
    return Provider(
        name=CUSTOM,
        base_url=base_url,
        api_key_env=("CASEAGENT_LLM_API_KEY",),
        requires_key=False,
    )


def known_providers() -> tuple[str, ...]:
    return (OPENROUTER.name, OPENAI.name, CUSTOM, FAKE)


def split_model_spec(spec: str, default_provider: str = DEFAULT_PROVIDER) -> tuple[str, str]:
    """Split ``[provider:]model`` into (provider name, model id).

    Only a known provider name counts as a prefix, because OpenRouter model ids
    use colons themselves (``qwen/qwen3.8-27b:free``).
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty model name")
    head, sep, tail = spec.partition(":")
    if sep and head.lower() in known_providers() and tail:
        return head.lower(), tail
    return default_provider, spec


def get_provider(name: str) -> Provider:
    if name == OPENROUTER.name:
        return OPENROUTER
    if name == OPENAI.name:
        return OPENAI
    if name == CUSTOM:
        return custom_provider()
    raise ValueError(f"Unknown provider {name!r}. Known: {', '.join(known_providers())}")


def api_key_for(provider: Provider) -> str | None:
    for variable in provider.api_key_env:
        value = os.environ.get(variable, "").strip()
        if value:
            return value
    return None
