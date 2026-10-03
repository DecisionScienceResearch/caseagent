"""The model interface the agent depends on.

The core owns this small port; ``caseagent.llm`` provides implementations. A
request separates the large, stable case materials (``context``) from the
step-specific task (``instruction``) so an implementation can cache the former
across the several calls the agent makes for one filing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ChatRequest:
    system: str
    context: str
    instruction: str
    label: str = ""

    @property
    def char_count(self) -> int:
        return len(self.system) + len(self.context) + len(self.instruction)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.calls += other.calls
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd


@dataclass
class ChatResponse:
    text: str
    model: str
    finish_reason: str | None = None
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0


ProgressCallback = Callable[[int], None]
"""Called with the number of characters received so far while a reply streams."""


class ChatClient(Protocol):
    model: str

    def complete(
        self, request: ChatRequest, on_progress: ProgressCallback | None = None
    ) -> ChatResponse: ...


class LLMError(Exception):
    """The model call failed in a way the user needs to act on."""


class LLMTruncated(LLMError):
    """The reply was cut off by the output-token limit."""


class LLMRefused(LLMError):
    """The model or provider declined to answer."""
