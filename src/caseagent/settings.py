"""Configuration from environment variables and a ``.env`` file in the working directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from .decisionlayer.client import DEFAULT_BASE_URL
from .llm.factory import DEFAULT_EFFORT, DEFAULT_MODEL


@dataclass(frozen=True)
class Settings:
    decisionlayer_base_url: str
    decisionlayer_api_key: str | None
    model: str
    effort: str | None
    effort_explicit: bool
    max_tokens: int | None
    llm_timeout: float
    cases_dir: Path
    runs_dir: Path
    cache_dir: Path
    dotenv_path: Path | None

    @classmethod
    def load(cls) -> Settings:
        dotenv = find_dotenv(usecwd=True)
        if dotenv:
            load_dotenv(dotenv, override=False)
        env = os.environ
        effort = env.get("CASEAGENT_EFFORT")
        max_tokens = env.get("CASEAGENT_MAX_TOKENS")
        return cls(
            decisionlayer_base_url=env.get("DECISIONLAYER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            decisionlayer_api_key=env.get("DECISIONLAYER_API_KEY", "").strip() or None,
            model=env.get("CASEAGENT_MODEL", "").strip() or DEFAULT_MODEL,
            effort=(effort.strip() if effort else DEFAULT_EFFORT),
            effort_explicit=bool(effort),
            max_tokens=int(max_tokens) if max_tokens else None,
            llm_timeout=float(env.get("CASEAGENT_LLM_TIMEOUT", "900")),
            cases_dir=Path(env.get("CASEAGENT_CASES_DIR", "cases")).expanduser(),
            runs_dir=Path(env.get("CASEAGENT_RUNS_DIR", "runs")).expanduser(),
            cache_dir=Path(
                env.get("CASEAGENT_CACHE_DIR", str(Path.home() / ".cache" / "caseagent"))
            ).expanduser(),
            dotenv_path=Path(dotenv) if dotenv else None,
        )
