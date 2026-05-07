from __future__ import annotations

import numpy as np

from babytheorist.dominant import AbstractRef, AbstractStep
from babytheorist.filter.constraints import accumulate_constraint
from babytheorist.filter.rules import program_is_valid
from babytheorist.filter.sampler import sample_program_with_constraint
from babytheorist.filter.state import FilterState
from babytheorist.filter.types import SamplerStats
from babytheorist.program import SamplerDeadEnd


def _apply(program, n_shapes, initial_grid_shapes=None):
    fs = FilterState.initial(n_shapes, grid_shapes=initial_grid_shapes)
    for step in program:
        fs = fs.apply(step)
    return fs


def test_filter_sampler_returns_valid_program_and_stats():
    stats = SamplerStats()
    program, constraint = sample_program_with_constraint(
        depth=2,
        n_shapes=2,
        vocab=frozenset({"rot90", "flip_h", "translate_up"}),
        selector_types=frozenset({"ALL", "by_index"}),
        rng=np.random.default_rng(0),
        stats=stats,
    )

    fs = _apply(program, 2)
    assert len(program) == 2
    assert program_is_valid(program, fs, constraint=constraint) is None
    assert accumulate_constraint(
        program,
        fs,
        vocab=frozenset({"rot90", "flip_h", "translate_up"}),
    ) == constraint
    assert stats.accepted_depth == 2
    assert len(stats.candidate_counts) == 2
    assert all(c > 0 for c in stats.candidate_counts)


def test_filter_sampler_uses_lazy_path_for_large_candidate_sets():
    from babytheorist.filter.sampler import FULL_VOCAB

    stats = SamplerStats()
    program, constraint = sample_program_with_constraint(
        depth=3,
        n_shapes=3,
        vocab=FULL_VOCAB,
        selector_types=frozenset({
            "ALL", "by_index", "by_color", "is_hollow",
            "topmost", "bottommost", "leftmost", "rightmost",
        }),
        palette=(1, 2, 3, 4, 5, 6, 7, 8, 9),
        rng=np.random.default_rng(1),
        stats=stats,
    )

    assert len(program) == 3
    assert not constraint.is_infeasible()
    assert max(stats.candidate_counts) > 64
    assert len(stats.valid_candidate_counts) == 3


def test_filter_abstract_sampler_short_circuits_deterministic_dead_end():
    ref = AbstractRef(0, "A", "single")
    abstract = (
        AbstractStep(ref, "translate_up"),
        AbstractStep(ref, "translate_down"),
    )
    stats = SamplerStats()

    try:
        sample_program_with_constraint(
            depth=2,
            n_shapes=1,
            vocab=frozenset({"translate_up", "translate_down"}),
            selector_types=frozenset({"by_index"}),
            rng=np.random.default_rng(0),
            abstract_program=abstract,
            max_step_retries=10,
            stats=stats,
        )
    except SamplerDeadEnd:
        pass
    else:
        raise AssertionError("expected deterministic abstract grounding dead-end")

    assert stats.candidate_counts == [1, 1]
    assert stats.valid_candidate_counts == [1, 0]
