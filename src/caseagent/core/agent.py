"""The advocate agent: writes one filing for one party at one phase.

The loop is deliberately plain and model-agnostic (no tool calling):

1. brief   - a private analysis of burdens, record facts, the other side's
             points, weaknesses, relief, and possible counterclaims;
2. draft   - the filing itself, in tagged sections;
3. review  - a cite-check and revision against the record.

``Depth`` selects how many of those passes run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from .llm import ChatClient, ChatRequest, ChatResponse, Usage
from .models import Filing, Stage
from .options import ComposeOptions, Depth, OutputPlan
from .parsing import FormatError, ParsedReply, parse_filing, parse_questions, render_sections
from .prompts import (
    brief_task,
    case_materials,
    draft_task,
    interview_task,
    plan_output,
    repair_task,
    review_task,
    system_prompt,
)
from .view import StageView


@dataclass(frozen=True)
class StepEvent:
    """Progress notification: a model call started, is streaming, or finished."""

    stage: Stage
    step: str
    phase: str  # "start" | "progress" | "end"
    chars: int = 0
    response: ChatResponse | None = None


EventCallback = Callable[[StepEvent], None]


@dataclass
class StepTrace:
    name: str
    task: str
    reply: str
    model: str
    finish_reason: str | None
    usage: Usage
    seconds: float


@dataclass
class ComposeResult:
    filing: Filing
    notes_for_principal: str | None = None
    brief: str | None = None
    review_notes: str | None = None
    warnings: list[str] = field(default_factory=list)
    steps: list[StepTrace] = field(default_factory=list)
    system: str = ""
    materials: str = ""

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.steps:
            total.add(step.usage)
        return total

    @property
    def seconds(self) -> float:
        return sum(step.seconds for step in self.steps)


class CaseAgent:
    def __init__(
        self,
        llm: ChatClient,
        options: ComposeOptions | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.llm = llm
        self.options = options or ComposeOptions()
        self.on_event = on_event

    def compose(self, view: StageView) -> ComposeResult:
        """Write the filing for ``view.stage`` on behalf of ``view.party``."""
        options = self.options
        plan = plan_output(view, options)
        system = system_prompt(view)
        materials = case_materials(view)
        steps: list[StepTrace] = []
        warnings = list(view.warnings)

        brief = None
        if options.depth is not Depth.QUICK:
            brief = self._call(view, "brief", system, materials, brief_task(view, options), steps)

        draft = self._parsed_call(
            view, "draft", system, materials, draft_task(view, options, brief), plan, steps
        )

        final = draft
        if options.depth is Depth.THOROUGH:
            task = review_task(view, options, render_sections(draft, plan))
            try:
                final = self._parsed_call(view, "review", system, materials, task, plan, steps)
            except FormatError as exc:
                warnings.append(
                    f"The review pass for the {view.stage.title} returned an unusable reply "
                    f"({exc}); the unreviewed draft was kept."
                )

        words = final.filing.word_count
        if words > options.max_words * 1.25:
            warnings.append(
                f"The {view.stage.title} runs {words} words, over the {options.max_words}-word target."
            )

        return ComposeResult(
            filing=final.filing,
            notes_for_principal=final.notes_for_principal or draft.notes_for_principal,
            brief=brief,
            review_notes=final.review_notes,
            warnings=warnings,
            steps=steps,
            system=system,
            materials=materials,
        )

    def questions(
        self, view: StageView, max_questions: int = 8, follow_up: bool = False
    ) -> list[str]:
        """Ask what the agent needs from its principal before writing ``view.stage``.

        Returns an empty list when the model says it has what it needs.
        """
        steps: list[StepTrace] = []
        task = interview_task(view, max_questions, follow_up=follow_up)
        system = system_prompt(view)
        materials = case_materials(view)
        reply = self._call(view, "interview", system, materials, task, steps)
        try:
            return parse_questions(reply, max_questions)
        except FormatError as exc:
            retry = repair_task(task, reply, str(exc))
            reply = self._call(view, "interview-repair", system, materials, retry, steps)
            return parse_questions(reply, max_questions)

    def _parsed_call(
        self,
        view: StageView,
        name: str,
        system: str,
        materials: str,
        task: str,
        plan: OutputPlan,
        steps: list[StepTrace],
    ) -> ParsedReply:
        """Run a step whose reply must parse as a filing, with one repair attempt."""
        reply = self._call(view, name, system, materials, task, steps)
        try:
            return parse_filing(view.stage, reply, plan)
        except FormatError as exc:
            retry = repair_task(task, reply, str(exc))
            reply = self._call(view, f"{name}-repair", system, materials, retry, steps)
            return parse_filing(view.stage, reply, plan)

    def _call(
        self,
        view: StageView,
        name: str,
        system: str,
        materials: str,
        task: str,
        steps: list[StepTrace],
    ) -> str:
        stage = view.stage
        request = ChatRequest(
            system=system,
            context=materials,
            instruction=task,
            label=f"{view.case_key}/{stage.value}/{name}",
        )
        self._emit(StepEvent(stage, name, "start"))

        def on_progress(chars: int) -> None:
            self._emit(StepEvent(stage, name, "progress", chars=chars))

        response = self.llm.complete(request, on_progress=on_progress)
        steps.append(
            StepTrace(
                name=name,
                task=task,
                reply=response.text,
                model=response.model,
                finish_reason=response.finish_reason,
                usage=response.usage,
                seconds=response.seconds,
            )
        )
        self._emit(StepEvent(stage, name, "end", chars=len(response.text), response=response))
        return response.text.strip()

    def _emit(self, event: StepEvent) -> None:
        if self.on_event is not None:
            self.on_event(event)
