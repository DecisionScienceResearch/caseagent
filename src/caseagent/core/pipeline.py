"""Continue a thread: write every phase that has not been filed yet, in order.

Each phase is written by the agent for the side that files it, against the
thread as it actually stands at that moment, and appended to the store before
the next one starts. In a simulation both sides are agents; they are separate
instances that share nothing but the filed record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .agent import CaseAgent
from .casefile import CaseBundle
from .models import STAGES, Role, Stage
from .thread import ThreadEntry, ThreadStore
from .view import build_stage_view


class PlanError(ValueError):
    """The requested continuation cannot be played."""


def remaining_stages(store: ThreadStore, through: Stage | None = None) -> list[Stage]:
    """Phases still to be filed, up to and including ``through`` (default: the Response)."""
    first = store.next_stage()
    if first is None:
        return []
    last = through or STAGES[-1]
    if last.index < first.index:
        raise PlanError(
            f"The thread already has the {STAGES[first.index - 1].title}; "
            f"nothing is left to write through the {last.title}."
        )
    return list(STAGES[first.index : last.index + 1])


StageCallback = Callable[[Stage, str], None]
"""Called with the phase and "start" or "end" as the continuation advances."""


def continue_thread(
    bundle: CaseBundle,
    store: ThreadStore,
    agents: Mapping[Role, CaseAgent],
    models: Mapping[Role, str],
    through: Stage | None = None,
    on_stage: StageCallback | None = None,
) -> list[ThreadEntry]:
    """Write and append every remaining phase. Returns the new entries."""
    stages = remaining_stages(store, through)
    for stage in stages:
        if stage.role not in agents:
            raise PlanError(
                f"No agent is playing the {stage.role.value}, who files the {stage.title}."
            )
    written: list[ThreadEntry] = []
    for stage in stages:
        if on_stage:
            on_stage(stage, "start")
        view = build_stage_view(bundle, stage, store.thread())
        result = agents[stage.role].compose(view)
        entry = ThreadEntry.from_result(result, models[stage.role])
        store.append(entry, result)
        written.append(entry)
        if on_stage:
            on_stage(stage, "end")
    return written
