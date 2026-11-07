from __future__ import annotations

import pytest

from babytheorist.core import ALL, Step
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
from babytheorist.filter.rules import CANONICAL_TO_MEANING_RULE, program_is_valid
from babytheorist.filter.state import FilterState


_SELECTORS = frozenset({"ALL", "by_index"})


def _state(program):
    fs = FilterState.initial(1, grid_shapes=["hollow"])
    for step in program:
        fs = fs.apply(step)
    return fs


def _canonical_violation(program, fs):
    checks = (
        lambda: check_count_limits(program, fs),
        lambda: check_state_revisit(program, fs),
        lambda: check_minimal_word(program, fs),
        lambda: check_joint_stabilizer_shortcut(
            program, fs, selector_types=_SELECTORS,
        ),
        lambda: check_d4_stabilizer_shortcut(
            program, fs, selector_types=_SELECTORS,
        ),
        lambda: check_translate_cancellation(program, fs),
        lambda: check_translate_joint_orchestration(
            program, fs, selector_types=_SELECTORS,
        ),
        lambda: check_grid_shape_cancellation(
            program, fs, selector_types=_SELECTORS,
        ),
        lambda: check_add_border_d4_split_shortcut(
            program, fs, selector_types=_SELECTORS,
        ),
        lambda: check_dead_writes(program, fs),
        lambda: check_trace_identity(
            program,
            fs,
            selector_types=_SELECTORS,
            constraint=None,
            palette=None,
        ),
    )
    for check in checks:
        violation = check()
        if violation is not None:
            return violation
    return None


@pytest.mark.parametrize("program", [
    [Step(ALL, "rot90")],
    [Step(ALL, "flip_h"), Step(ALL, "flip_h")],
    [Step(ALL, "translate_up"), Step(ALL, "translate_down")],
    [Step(ALL, "rot90"), Step(ALL, "rot90"), Step(ALL, "flip_h")],
    [Step(ALL, "translate_up"), Step(ALL, "flip_h"), Step(ALL, "translate_down")],
])
def test_program_rules_preserve_canonical_first_violation(program):
    fs = _state(program)

    canonical = _canonical_violation(program, fs)
    result = program_is_valid(program, fs, selector_types=_SELECTORS)

    if canonical is None:
        assert result is None
        return

    assert result is not None
    assert result.canonical_rule == canonical.rule
    assert result.rule == CANONICAL_TO_MEANING_RULE[canonical.rule]
