"""Tests for babytheorist.evaluation — Y_q computation and selector alternatives."""

from __future__ import annotations

import sys; sys.path.insert(0, ".")

import numpy as np

from babytheorist.core import (
    ALL, Shape, Step,
    by_color, bottommost, leftmost, rightmost, topmost,
)
from babytheorist.engine import apply_program
from babytheorist.evaluation import (
    alternatives_at_step,
    compute_valid_query_outputs,
    compute_valid_query_outputs_from_supports,
    enumerate_selector_pool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(color: int, rows: int, cols: int, pos: tuple[int, int]) -> Shape:
    return Shape(np.full((rows, cols), color, dtype=int), pos)


# ---------------------------------------------------------------------------
# enumerate_selector_pool
# ---------------------------------------------------------------------------

def test_pool_by_color_expands_per_palette():
    pool = enumerate_selector_pool(("by_color",), palette=(1, 2, 3))
    assert [s.type for s in pool] == ["by_color"] * 3
    assert [s.param for s in pool] == [1, 2, 3]


def test_pool_non_by_color_single_entries():
    pool = enumerate_selector_pool(("ALL", "topmost", "is_hollow"), palette=(7,))
    assert [s.type for s in pool] == ["ALL", "topmost", "is_hollow"]
    assert all(s.param is None for s in pool)


def test_pool_mixed_preserves_order():
    pool = enumerate_selector_pool(("ALL", "by_color", "topmost"), palette=(4, 5))
    assert [s.type for s in pool] == ["ALL", "by_color", "by_color", "topmost"]
    assert [s.param for s in pool] == [None, 4, 5, None]


def test_pool_by_index_expands_per_n_shapes():
    pool = enumerate_selector_pool(("by_index",), palette=(1,), n_shapes=3)
    assert [s.type for s in pool] == ["by_index"] * 3
    assert [s.param for s in pool] == [1, 2, 3]


def test_pool_by_index_raises_without_n_shapes():
    # Silent skip would hide misconfig; fail loudly when n_shapes is missing.
    import pytest
    with pytest.raises(ValueError, match="by_index requires n_shapes"):
        enumerate_selector_pool(("ALL", "by_index"), palette=(1,))


# ---------------------------------------------------------------------------
# alternatives_at_step
# ---------------------------------------------------------------------------

def test_alternatives_transform_equivalent_selectors():
    # Shape 0 is topmost AND leftmost AND by_color(2); shape 1 is by_color(3).
    scene = [_solid(2, 2, 2, (0, 0)), _solid(3, 2, 2, (5, 5))]
    step = Step(topmost(), "rot90")
    pool = [ALL, by_color(2), by_color(3), topmost(), leftmost()]
    alts = alternatives_at_step(step, scene, pool)
    picks = {a.selector for a in alts}
    # Selectors that pick [0] on this scene, hence equivalent to topmost
    assert picks == {topmost(), leftmost(), by_color(2)}
    # ALL picks [0, 1] → not equivalent; by_color(3) picks [1] → not equivalent
    assert by_color(3) not in picks
    assert ALL not in picks


def test_alternatives_canonical_inserted_when_pool_missing_it():
    # Pool doesn't contain the canonical step.selector — function adds it back.
    scene = [_solid(2, 2, 2, (0, 0))]
    step = Step(topmost(), "rot90")
    alts = alternatives_at_step(step, scene, pool=[])
    assert len(alts) == 1
    assert alts[0] == step


def test_alternatives_relational_cartesian_product():
    # Two shapes in opposite corners. Single-selector axes line up.
    scene = [_solid(2, 2, 2, (0, 0)), _solid(3, 2, 2, (5, 5))]
    step = Step(topmost(), "swap_colors", target=bottommost())
    # Pool mixes single and multi selectors; multi should be filtered out.
    pool = [ALL, topmost(), leftmost(), bottommost(), rightmost()]
    alts = alternatives_at_step(step, scene, pool)
    # src_alts (pick [0]): topmost, leftmost. tgt_alts (pick [1]): bottommost, rightmost.
    # Cartesian: 2 × 2 = 4
    assert len(alts) == 4
    src_types = {a.selector.type for a in alts}
    tgt_types = {a.target.type for a in alts}
    assert src_types == {"topmost", "leftmost"}
    assert tgt_types == {"bottommost", "rightmost"}


# ---------------------------------------------------------------------------
# compute_valid_query_outputs
# ---------------------------------------------------------------------------

def test_Yq_no_branching_equals_canonical_output():
    # ALL-only pool: every step has exactly one alternative (itself).
    # 2x3 rectangles are asymmetric under rot90 → apply_step succeeds.
    demo = [_solid(2, 2, 3, (0, 0)), _solid(3, 2, 3, (5, 5))]
    query = [_solid(2, 2, 3, (1, 1)), _solid(3, 2, 3, (6, 6))]
    program = [Step(ALL, "rot90")]
    pool = [ALL]
    Y_q = compute_valid_query_outputs(program, demo, query, world_size=12, pool=pool)
    assert len(Y_q) == 1
    expected, _ = apply_program(program, [s.copy() for s in query], 12)
    assert np.array_equal(Y_q[0], expected)


def test_Yq_branching_produces_multiple_outputs():
    # Demo: topmost == leftmost (both pick shape 0, at (0, 0)).
    # Query: topmost picks shape 0, leftmost picks shape 1 — they diverge.
    demo = [_solid(2, 2, 2, (0, 0)), _solid(3, 2, 2, (5, 5))]
    query = [_solid(2, 2, 2, (0, 5)), _solid(3, 2, 2, (5, 0))]
    program = [Step(topmost(), "next_color")]
    pool = [topmost(), leftmost()]
    Y_q = compute_valid_query_outputs(program, demo, query, world_size=10, pool=pool)
    # Two alternatives on demo (topmost, leftmost); they diverge on query
    # and next_color changes different shapes → two distinct grids.
    assert len(Y_q) == 2


def test_Yq_output_cap_stops_after_distinct_outputs():
    demo = [
        _solid(2, 2, 2, (0, 0)),
        _solid(3, 2, 2, (5, 5)),
        _solid(4, 2, 2, (6, 6)),
    ]
    query = [
        _solid(3, 2, 2, (0, 5)),
        _solid(4, 2, 2, (5, 0)),
        _solid(2, 2, 2, (6, 6)),
    ]
    program = [Step(topmost(), "next_color")]
    pool = [topmost(), leftmost(), by_color(2)]

    capped = compute_valid_query_outputs(
        program, demo, query, world_size=10, pool=pool, max_outputs=2,
    )
    uncapped = compute_valid_query_outputs(
        program, demo, query, world_size=10, pool=pool,
    )

    assert len(capped) == 2
    assert len(uncapped) == 3


def test_multi_support_disambiguates_query_output():
    # Support 1 leaves topmost and leftmost indistinguishable; support 2
    # separates them, so the query answer becomes unique.
    support1 = [_solid(2, 2, 2, (0, 0)), _solid(3, 2, 2, (5, 5))]
    support2 = [_solid(2, 2, 2, (0, 5)), _solid(3, 2, 2, (5, 0))]
    query = [_solid(2, 2, 2, (1, 6)), _solid(3, 2, 2, (6, 1))]
    program = [Step(topmost(), "next_color")]
    pool = [topmost(), leftmost()]

    ambiguous = compute_valid_query_outputs_from_supports(
        program, [support1], query, world_size=10, pool=pool,
    )
    unique = compute_valid_query_outputs_from_supports(
        program, [support1, support2], query, world_size=10, pool=pool,
    )
    assert len(ambiguous) == 2
    assert len(unique) == 1


def test_Yq_none_when_any_branch_fails_precondition():
    # Strict mode: query must accommodate every alternative. swap_colors
    # requires distinct primaries, so a same-color query breaks every pair
    # → Y_q is None, signalling "incompatible query" to the caller.
    demo = [_solid(2, 2, 2, (0, 0)), _solid(3, 2, 2, (5, 5))]
    query = [_solid(2, 2, 2, (0, 5)), _solid(2, 2, 2, (5, 0))]
    program = [Step(topmost(), "swap_colors", target=bottommost())]
    pool = [topmost(), bottommost(), leftmost(), rightmost()]
    Y_q = compute_valid_query_outputs(program, demo, query, world_size=10, pool=pool)
    assert Y_q is None
