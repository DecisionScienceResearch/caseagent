"""Knobs for how a filing is composed."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Depth(StrEnum):
    """How many passes the agent makes over one filing."""

    QUICK = "quick"  # draft only (1 model call)
    STANDARD = "standard"  # private brief, then draft (2 calls)
    THOROUGH = "thorough"  # brief, draft, then a cite-check and revision (3 calls)


class CounterclaimPolicy(StrEnum):
    AUTO = "auto"  # assert a counterclaim in the Answer only if the record supports one
    NEVER = "never"


@dataclass(frozen=True)
class ComposeOptions:
    depth: Depth = Depth.THOROUGH
    counterclaims: CounterclaimPolicy = CounterclaimPolicy.AUTO
    evidence_demands: bool = False
    max_words: int = 1500


@dataclass(frozen=True)
class OutputPlan:
    """Which optional sections a particular filing may contain."""

    counterclaim: bool = False
    evidence_demands: bool = False
    evidence_response: bool = False
