"""Shared data types for the production filter rule and sampler stack."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


FILTER_RULES_VERSION = "filter.rules.v3"


class RejectReason(str, Enum):
    """Stable coarse reject buckets emitted by the filter sampler."""

    PRECONDITION = "precondition"
    CONSTRAINT_INFEASIBLE = "constraint_infeasible"
    PROGRAM_RULE = "program_rule"
    DEAD_END = "dead_end"


@dataclass(frozen=True)
class FilterConfig:
    """Configuration values shared across filter calls."""

    max_step_retries: int = 100
    collect_stats: bool = False


@dataclass(frozen=True)
class FilterContext:
    """Immutable context for one sampled program prefix."""

    depth: int
    n_shapes: int
    vocab: frozenset[str]
    selector_types: frozenset[str]
    palette: tuple[int, ...] | None = None
    initial_grid_shapes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RuleResult:
    """Structured rule failure returned by filter rules.

    Truthiness intentionally matches the canonical ``RuleViolation`` contract:
    a result object is truthy and means the candidate was rejected.
    """

    rule: str
    reason: str
    canonical_rule: str | None = None

    def __bool__(self) -> bool:
        return True


@dataclass
class SamplerStats:
    """Lightweight per-call sampler diagnostics.

    The object is mutable by design so callers can pass one instance through a
    generation call and later serialize ``as_dict()`` into run metadata.
    """

    candidate_counts: list[int] = field(default_factory=list)
    valid_candidate_counts: list[int] = field(default_factory=list)
    reject_counts: Counter[str] = field(default_factory=Counter)
    dead_end_count: int = 0
    accepted_depth: int = 0

    def record_candidates(self, total: int, valid: int) -> None:
        self.candidate_counts.append(int(total))
        self.valid_candidate_counts.append(int(valid))

    def record_reject(self, reason: str) -> None:
        self.reject_counts[str(reason)] += 1

    def record_dead_end(self) -> None:
        self.dead_end_count += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_counts": list(self.candidate_counts),
            "valid_candidate_counts": list(self.valid_candidate_counts),
            "reject_counts": dict(sorted(self.reject_counts.items())),
            "dead_end_count": int(self.dead_end_count),
            "accepted_depth": int(self.accepted_depth),
        }
