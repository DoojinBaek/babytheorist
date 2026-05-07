"""Tests for babytheorist.filter.generate.

Focus: ``depth`` / ``n_shapes`` can be fixed or a ``(lo, hi)`` range;
validation of bad range arguments.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.engine import apply_program
from babytheorist.filter.generate import (
    generate_multipair_task, generate_task, _resolve_range,
)
from babytheorist.filter.sampler import D4_TRANSLATE_VOCAB
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)


def _factory() -> ShapeFactory:
    return ShapeFactory(generators=(solid,), size_range=(10, 20))


# === _resolve_range =======================================================


def test_resolve_range_int_passthrough():
    rng = np.random.default_rng(0)
    assert _resolve_range(3, rng, "depth") == 3


def test_resolve_range_tuple_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        v = _resolve_range((2, 5), rng, "depth")
        assert 2 <= v <= 5


def test_resolve_range_int_rejects_non_positive():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _resolve_range(0, rng, "depth")


def test_resolve_range_tuple_rejects_bad_bounds():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _resolve_range((3, 2), rng, "depth")
    with pytest.raises(ValueError):
        _resolve_range((0, 3), rng, "depth")


def test_resolve_range_tuple_rejects_non_int():
    rng = np.random.default_rng(0)
    with pytest.raises(TypeError):
        _resolve_range((1.0, 3.0), rng, "depth")


# === generate_task ranges =================================================


def test_generate_task_depth_range_covers_bounds():
    rng = np.random.default_rng(0)
    factory = _factory()
    seen = set()
    for _ in range(80):
        task = generate_task(
            depth=(1, 3), n_shapes=2, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
        )
        if task is not None:
            seen.add(len(task.program))
    assert seen == {1, 2, 3}, f"depth coverage incomplete: {seen}"


def test_generate_task_n_shapes_range_covers_bounds():
    rng = np.random.default_rng(0)
    factory = _factory()
    seen = set()
    for _ in range(80):
        task = generate_task(
            depth=1, n_shapes=(1, 2), factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
        )
        if task is not None:
            seen.add(len(task.input_shapes))
    assert seen == {1, 2}, f"n_shapes coverage incomplete: {seen}"


def test_generate_task_fixed_values_still_work():
    rng = np.random.default_rng(0)
    factory = _factory()
    task = generate_task(
        depth=2, n_shapes=2, factory=factory,
        palette=_PALETTE, world_size=24, rng=rng,
    )
    assert task is not None
    assert len(task.program) == 2
    assert len(task.input_shapes) == 2


# === generate_multipair_task ==============================================


def test_multipair_same_program_across_pairs():
    """All pairs in one task share the same program."""
    rng = np.random.default_rng(0)
    factory = _factory()
    mpt = generate_multipair_task(
        n_pairs=3, depth=2, n_shapes=2, factory=factory,
        palette=_PALETTE, world_size=24, rng=rng,
        vocab=D4_TRANSLATE_VOCAB,
    )
    assert mpt is not None
    assert len(mpt.pairs) == 3
    # Scenes are distinct — verify at least two pairs have different inputs.
    first = mpt.pairs[0][0]
    assert any(not np.array_equal(first, p[0]) for p in mpt.pairs[1:])


def test_multipair_each_pair_reproduces_program_output():
    rng = np.random.default_rng(1)
    factory = _factory()
    mpt = generate_multipair_task(
        n_pairs=2, depth=2, n_shapes=2, factory=factory,
        palette=_PALETTE, world_size=24, rng=rng,
        vocab=D4_TRANSLATE_VOCAB,
    )
    assert mpt is not None
    for input_grid, input_shapes, output_grid in mpt.pairs:
        out, _ = apply_program(mpt.program, input_shapes, 24)
        assert out is not None
        assert np.array_equal(out, output_grid)
        assert not np.array_equal(input_grid, output_grid)


def test_multipair_raster_order_matches_shape_slots():
    rng = np.random.default_rng(4)
    factory = _factory()
    mpt = generate_multipair_task(
        n_pairs=3, depth=2, n_shapes=3, factory=factory,
        palette=_PALETTE, world_size=32, rng=rng,
        vocab=D4_TRANSLATE_VOCAB, raster_order=True,
        max_scene_retries=1_000,
    )
    assert mpt is not None
    for _input_grid, input_shapes, _output_grid in mpt.pairs:
        positions = [s.position for s in input_shapes]
        assert positions == sorted(positions)


def test_multipair_rejects_bad_n_pairs():
    rng = np.random.default_rng(0)
    factory = _factory()
    try:
        generate_multipair_task(
            n_pairs=0, depth=1, n_shapes=2, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError on n_pairs=0")
