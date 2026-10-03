"""An offline stand-in model: ``fake:demo``.

It returns clearly labeled placeholder text in the required output format, so
the whole pipeline (ingest, plan, compose, save, build the simulator payload)
can be exercised without an API key. Its output is not an argument and must
never be sent to the simulator; the CLI refuses to.
"""

from __future__ import annotations

import re

from ..core.llm import ChatRequest, ChatResponse, ProgressCallback, Usage

FAKE_PREFIX = "fake:"
PLACEHOLDER_MARK = "[PLACEHOLDER from the offline fake model, not a real argument]"

_SECTION_RE = re.compile(r"^<([a-z_]+)>$", re.MULTILINE)


class FakeClient:
    def __init__(self, model: str = "fake:demo") -> None:
        self.model = model
        self.requests: list[ChatRequest] = []

    def complete(
        self, request: ChatRequest, on_progress: ProgressCallback | None = None
    ) -> ChatResponse:
        self.requests.append(request)
        text = self._reply(request)
        if on_progress is not None:
            on_progress(len(text))
        usage = Usage(
            calls=1,
            prompt_tokens=request.char_count // 4,
            completion_tokens=len(text) // 4,
        )
        return ChatResponse(text=text, model=self.model, finish_reason="stop", usage=usage)

    def _reply(self, request: ChatRequest) -> str:
        task = request.instruction
        if "# Your task now: questions for" in task:
            return "<ready/>"
        if "## Output format" not in task:
            return f"{PLACEHOLDER_MARK}\n\nPrivate brief for {request.label}."
        # Answer with every section the task's output format asks for.
        format_spec = task.split("## Output format", 1)[1]
        sections = []
        for tag in _SECTION_RE.findall(format_spec):
            if tag == "argument":
                body = (
                    f"{PLACEHOLDER_MARK}\n\n"
                    f"This stands in for the filing {request.label}. Run with a real model "
                    "to get an argument."
                )
            else:
                body = "NONE"
            sections.append(f"<{tag}>\n{body}\n</{tag}>")
        return "\n\n".join(sections)
