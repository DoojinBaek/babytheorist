from __future__ import annotations

from babytheorist.core import ALL, Step
from babytheorist.filter.rules import program_is_valid
from babytheorist.filter.state import FilterState


def test_filter_rule_result_uses_meaning_name():
    program = [Step(ALL, "rot90") for _ in range(4)]
    fs = FilterState.initial(1)
    for step in program:
        fs = fs.apply(step)

    result = program_is_valid(program, fs)

    assert result is not None
    assert result.rule == "PrimitiveCountBound"
    assert result.canonical_rule == "R1"
