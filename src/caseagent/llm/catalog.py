"""OpenRouter's public model catalog: the data behind the model picker.

``GET https://openrouter.ai/api/v1/models`` needs no key. Besides names and
prices it reports each model's context window, output cap, and which reasoning
effort levels it accepts, which lets requests be shaped per model instead of
guessing.
"""

from __future__ import annotations

import difflib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

CATALOG_URL = "https://openrouter.ai/api/v1/models"
CACHE_FILENAME = "openrouter-models.json"
MAX_AGE_SECONDS = 24 * 60 * 60

# Offered first by the interactive picker. Entries missing from the live catalog
# are dropped, so a stale id here is harmless.
FEATURED = (
    "anthropic/claude-opus-5.5",
    "anthropic/claude-fable-5.1",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-6.1-sol",
    "openai/gpt-6-luna",
    "google/gemini-3.8-flash",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.7",
    "deepseek/deepseek-v4-pro",
    "qwen/qwen3.8-max-0902",
    "mistralai/mistral-medium-3-5",
    "meta-llama/llama-4-maverick",
)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    created: int
    context_length: int | None
    max_output_tokens: int | None
    prompt_price: float | None  # USD per million input tokens
    completion_price: float | None  # USD per million output tokens
    supported_parameters: frozenset[str]
    reasoning: dict[str, Any] | None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> ModelInfo:
        pricing = raw.get("pricing") or {}
        top = raw.get("top_provider") or {}
        return cls(
            id=raw["id"],
            name=raw.get("name") or raw["id"],
            created=int(raw.get("created") or 0),
            context_length=raw.get("context_length") or top.get("context_length"),
            max_output_tokens=top.get("max_completion_tokens"),
            prompt_price=_per_million(pricing.get("prompt")),
            completion_price=_per_million(pricing.get("completion")),
            supported_parameters=frozenset(raw.get("supported_parameters") or ()),
            reasoning=raw.get("reasoning"),
        )

    def to_cache(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created": self.created,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "prompt_price": self.prompt_price,
            "completion_price": self.completion_price,
            "supported_parameters": sorted(self.supported_parameters),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_cache(cls, raw: dict[str, Any]) -> ModelInfo:
        return cls(
            id=raw["id"],
            name=raw["name"],
            created=raw["created"],
            context_length=raw["context_length"],
            max_output_tokens=raw["max_output_tokens"],
            prompt_price=raw["prompt_price"],
            completion_price=raw["completion_price"],
            supported_parameters=frozenset(raw["supported_parameters"]),
            reasoning=raw["reasoning"],
        )

    @property
    def efforts(self) -> tuple[str, ...]:
        """Effort levels the model lists, highest first; empty if it lists none."""
        if not self.reasoning:
            return ()
        return tuple(self.reasoning.get("supported_efforts") or ())

    def accepts_effort(self, effort: str) -> bool:
        """Whether ``reasoning.effort = effort`` is valid for this model.

        Per OpenRouter: a missing ``reasoning`` object means a non-reasoning
        model, a missing ``supported_efforts`` key means no effort selection,
        and an explicit null means every effort value is accepted.
        """
        if not self.reasoning or "supported_efforts" not in self.reasoning:
            return False
        supported = self.reasoning["supported_efforts"]
        return supported is None or effort in supported


def _per_million(price: Any) -> float | None:
    try:
        return float(price) * 1_000_000
    except (TypeError, ValueError):
        return None


class Catalog:
    def __init__(self, models: list[ModelInfo], fetched_at: float, stale: bool = False) -> None:
        self.models = {model.id: model for model in models}
        self.fetched_at = fetched_at
        self.stale = stale

    def get(self, model_id: str) -> ModelInfo | None:
        """Look up a model, falling back to the base id for variants like ``:nitro``."""
        if model_id in self.models:
            return self.models[model_id]
        base, sep, _variant = model_id.rpartition(":")
        return self.models.get(base) if sep else None

    def search(self, query: str = "", limit: int | None = None) -> list[ModelInfo]:
        """Models whose id or name contains every word of ``query``, newest first."""
        terms = query.lower().split()
        found = [
            model
            for model in self.models.values()
            if all(term in model.id.lower() or term in model.name.lower() for term in terms)
        ]
        found.sort(key=lambda model: (-model.created, model.id))
        return found[:limit] if limit else found

    def featured(self) -> list[ModelInfo]:
        return [self.models[model_id] for model_id in FEATURED if model_id in self.models]

    def suggest(self, model_id: str, count: int = 5) -> list[str]:
        return difflib.get_close_matches(model_id, list(self.models), n=count, cutoff=0.5)


def load_catalog(
    cache_dir: Path,
    refresh: bool = False,
    max_age: float = MAX_AGE_SECONDS,
    client: httpx.Client | None = None,
) -> Catalog | None:
    """Return the catalog from a fresh cache, the network, or a stale cache, in that order.

    Returns None only when there is no cache and the network is unreachable; the
    caller then proceeds without per-model metadata.
    """
    path = cache_dir / CACHE_FILENAME
    cached = _read_cache(path)
    if cached and not refresh and time.time() - cached.fetched_at < max_age:
        return cached
    try:
        response = (client or httpx).get(CATALOG_URL, timeout=20.0)
        response.raise_for_status()
        models = [
            ModelInfo.from_api(raw)
            for raw in response.json()["data"]
            if "text" in ((raw.get("architecture") or {}).get("output_modalities") or ["text"])
        ]
    except (httpx.HTTPError, KeyError, ValueError):
        if cached:
            cached.stale = True
        return cached
    catalog = Catalog(models, fetched_at=time.time())
    _write_cache(path, catalog)
    return catalog


def _read_cache(path: Path) -> Catalog | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = [ModelInfo.from_cache(entry) for entry in raw["models"]]
        return Catalog(models, fetched_at=float(raw["fetched_at"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, catalog: Catalog) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": catalog.fetched_at,
            "models": [model.to_cache() for model in catalog.models.values()],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass  # the cache is an optimization; a read-only home directory is fine
