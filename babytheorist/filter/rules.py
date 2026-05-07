"""Meaning-named production filter rule engine."""

from __future__ import annotations

from babytheorist.core import Program
from babytheorist.filter.rules_program import (
    check_count_limits,
    check_add_border_d4_split_shortcut,
    check_d4_stabilizer_shortcut,
    check_dead_writes,
    check_grid_shape_cancellation,
    check_joint_stabilizer_shortcut,
    check_minimal_word,
    check_state_revisit,
    check_trace_identity,
    check_translate_cancellation,
    check_translate_joint_orchestration,
)

from .state import FilterState
from .types import RuleResult


CANONICAL_TO_MEANING_RULE = {
    "R1": "PrimitiveCountBound",
    "R3": "NoStateCycle",
    "R6": "D4MinimalWord",
    "I6": "NoJointSelectorShortcut",
    "I6b": "NoD4CosetShortcut",
    "R5": "NoAxisCancellation",
    "R5j": "NoJointSelectorShortcut",
    "R5b": "NoAxisCancellation",
    "R8": "NoAxisCancellation",
    "R4": "NoDeadWrites",
    "R7'": "NoTraceMergeShortcut",
}

FILTER_RULE_NAMES = tuple(dict.fromkeys(CANONICAL_TO_MEANING_RULE.values()))


def _convert_violation(violation) -> RuleResult | None:
    if violation is None:
        return None
    canonical_rule = getattr(violation, "rule", None)
    rule = CANONICAL_TO_MEANING_RULE.get(
        canonical_rule,
        str(canonical_rule or "UnknownRule"),
    )
    reason = getattr(violation, "reason", repr(violation))
    return RuleResult(rule=rule, reason=reason, canonical_rule=canonical_rule)


def program_is_valid(
    program: Program,
    fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> RuleResult | None:
    """Run all production program rules in cheap-first order.

    The low-level rule functions still emit canonical rule ids such as R1/R5;
    this wrapper maps them to stable meaning-based names for diagnostics.
    """

    checks = (
        lambda: check_count_limits(program, fs),
        lambda: check_state_revisit(program, fs),
        lambda: check_minimal_word(program, fs),
        lambda: check_joint_stabilizer_shortcut(
            program, fs, selector_types=selector_types,
        ),
        lambda: check_d4_stabilizer_shortcut(
            program, fs, selector_types=selector_types,
        ),
        lambda: check_translate_cancellation(program, fs),
        lambda: check_translate_joint_orchestration(
            program, fs, selector_types=selector_types,
        ),
        lambda: check_grid_shape_cancellation(
            program, fs, selector_types=selector_types,
        ),
        lambda: check_add_border_d4_split_shortcut(
            program, fs, selector_types=selector_types,
        ),
        lambda: check_dead_writes(program, fs),
        lambda: check_trace_identity(
            program,
            fs,
            selector_types=selector_types,
            constraint=constraint,
            palette=palette,
        ),
    )
    for check in checks:
        result = _convert_violation(check())
        if result is not None:
            return result
    return None


__all__ = [
    "CANONICAL_TO_MEANING_RULE",
    "FILTER_RULE_NAMES",
    "program_is_valid",
]
