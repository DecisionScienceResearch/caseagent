"""Turn a model name into a ready client.

This is where the model picker's choice meets what each model actually accepts.
Reasoning effort is only sent to a model whose catalog entry lists that level,
and the output cap is kept within the model's limit, so the same command line
works across vendors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from ..core.llm import ChatClient, LLMError
from .catalog import Catalog, ModelInfo
from .fake import FakeClient
from .openai_compat import OpenAICompatClient, RequestParams
from .providers import (
    DEFAULT_PROVIDER,
    FAKE,
    OPENROUTER,
    Provider,
    api_key_for,
    get_provider,
    split_model_spec,
)

DEFAULT_MODEL = "anthropic/claude-opus-5.5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 64_000
# Effort values that mean "send nothing and let the model use its own default".
EFFORT_UNSET = ("", "default", "auto")


@dataclass(frozen=True)
class ModelOptions:
    """User-facing model settings, before they are checked against a model."""

    effort: str | None = DEFAULT_EFFORT
    effort_explicit: bool = False
    max_tokens: int | None = None
    temperature: float | None = None
    timeout: float = 900.0
    cache_prompt: bool = True


@dataclass(frozen=True)
class BuiltClient:
    client: ChatClient
    label: str
    info: ModelInfo | None
    notes: tuple[str, ...] = ()


def is_fake(spec: str) -> bool:
    return split_model_spec(spec)[0] == FAKE


def model_label(spec: str) -> str:
    """Canonical display name: the bare id for the default provider, else prefixed."""
    provider, model_id = split_model_spec(spec)
    return model_id if provider == DEFAULT_PROVIDER else f"{provider}:{model_id}"


def build_client(
    spec: str,
    options: ModelOptions,
    catalog: Catalog | None = None,
    *,
    reload_catalog: Callable[[], Catalog | None] | None = None,
    session_id: str | None = None,
    http_client: httpx.Client | None = None,
) -> BuiltClient:
    """Create the client for ``[provider:]model``.

    Raises LLMError for a missing key or, on OpenRouter, an id the catalog does
    not list (with close matches), so mistakes surface before any work starts.
    """
    provider_name, model_id = split_model_spec(spec)
    label = model_label(spec)
    if provider_name == FAKE:
        return BuiltClient(client=FakeClient(model=label), label=label, info=None)

    try:
        provider = get_provider(provider_name)
    except ValueError as exc:
        raise LLMError(str(exc)) from exc
    api_key = api_key_for(provider)
    if provider.requires_key and not api_key:
        raise LLMError(
            f"No API key for {provider.name}. Set {provider.api_key_env[0]} in your "
            "environment or in a .env file."
        )

    info: ModelInfo | None = None
    if provider is OPENROUTER and catalog is not None:
        info = catalog.get(model_id)
        if info is None and reload_catalog is not None:
            catalog = reload_catalog() or catalog
            info = catalog.get(model_id)
        if info is None:
            suggestions = catalog.suggest(model_id)
            hint = f" Did you mean {', '.join(suggestions)}?" if suggestions else ""
            raise LLMError(
                f"{model_id!r} is not in OpenRouter's catalog.{hint} "
                "Run `caseagent models` to browse."
            )

    params, notes = resolve_params(provider, model_id, info, options)
    client = OpenAICompatClient(
        provider,
        model_id,
        api_key,
        params,
        label=label,
        session_id=session_id,
        timeout=options.timeout,
        http_client=http_client,
    )
    return BuiltClient(client=client, label=label, info=info, notes=tuple(notes))


def resolve_params(
    provider: Provider,
    model_id: str,
    info: ModelInfo | None,
    options: ModelOptions,
) -> tuple[RequestParams, list[str]]:
    """Fit the requested settings to one model. Returns the params and any notes."""
    notes: list[str] = []
    reasoning_style = provider.reasoning_style

    # Output cap. Reasoning tokens count against it, so the default is generous.
    # Without catalog data the cap is left to the provider unless set explicitly.
    max_tokens = options.max_tokens
    if max_tokens is None and info is not None:
        max_tokens = min(DEFAULT_MAX_TOKENS, info.max_output_tokens or DEFAULT_MAX_TOKENS)
    elif max_tokens and info and info.max_output_tokens and max_tokens > info.max_output_tokens:
        notes.append(
            f"{model_id} caps output at {info.max_output_tokens:,} tokens; "
            f"using that instead of {max_tokens:,}."
        )
        max_tokens = info.max_output_tokens

    # Reasoning effort.
    effort: str | None = None
    wanted = (options.effort or "").strip().lower()
    if wanted not in EFFORT_UNSET:
        if reasoning_style == "openrouter" and info is not None:
            if info.accepts_effort(wanted):
                effort = wanted
            elif options.effort_explicit:
                supported = ", ".join(info.efforts) or "none"
                notes.append(
                    f"{model_id} does not accept effort={wanted} (it lists: {supported}); "
                    "using the model's default."
                )
        elif reasoning_style is not None and options.effort_explicit:
            # No catalog entry to check against: send what the user asked for.
            effort = wanted
        elif reasoning_style is None and options.effort_explicit:
            notes.append(f"Reasoning effort is not sent to {model_id} on a custom endpoint.")

    # Claude models cache only where a breakpoint is placed; other vendors cache
    # a repeated prefix on their own.
    cache_prompt = (
        options.cache_prompt and provider.cache_control and model_id.startswith("anthropic/")
    )

    params = RequestParams(
        max_tokens=max_tokens,
        effort=effort,
        temperature=options.temperature,
        cache_prompt=cache_prompt,
        context_length=info.context_length if info else None,
    )
    return params, notes
