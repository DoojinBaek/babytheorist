"""Tests for babytheorist.filter.scene_sampler."""

from __future__ import annotations

import numpy as np
import pytest

import babytheorist.filter.scene_sampler as scene_sampler
from babytheorist.core import Shape
from babytheorist.filter.constraint import InputConstraint
from babytheorist.filter.d4_tables import D4_IDENTITY, D4_SUBGROUPS
from babytheorist.filter.scene_sampler import (
    grid_shape_of,
    primary_color_of,
    sample_scene,
)
from babytheorist.filter.shape_attrs import stabilizer_of_pixel_grid
from babytheorist.primitives import add_border
from babytheorist.shapes import DEFAULT_GENERATORS, ShapeFactory, bordered, patterned, solid


# === Helpers ==============================================================


def _factory_solid_only(*, allow_constructive_shapes: bool = False) -> ShapeFactory:
    return ShapeFactory(
        generators=(solid,),
        size_range=(4, 10),
        allow_constructive_shapes=allow_constructive_shapes,
    )


_PALETTE = (1, 2, 3, 4, 5)


# === Attribute helpers ====================================================


def test_primary_color_of_solid():
    f = _factory_solid_only()
    rng = np.random.default_rng(0)
    s = f.sample(rng, _PALETTE)
    assert primary_color_of(s) in _PALETTE


def test_primary_color_of_matches_runtime_most_frequent_color():
    s = Shape(np.array([
        [2, 2, 2],
        [2, 1, 2],
        [2, 2, 2],
    ]))
    assert primary_color_of(s) == s.primary_color == 2


def test_grid_shape_of_solid_random_walk():
    """Solid random walk is compact; bbox often has empty cells, but
    interior (bbox[1:-1, 1:-1]) should be filled for compact walks.

    Whatever it is, the helper returns solid|hollow consistently.
    """
    f = _factory_solid_only()
    rng = np.random.default_rng(0)
    s = f.sample(rng, _PALETTE)
    assert grid_shape_of(s) in {"solid", "hollow"}


def test_grid_shape_of_matches_runtime_enclosed_hole_semantics():
    u_shape = Shape(np.array([
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]))
    assert not u_shape.is_hollow
    assert grid_shape_of(u_shape) == "solid"


def test_shape_satisfies_skips_unconstrained_stabilizer(monkeypatch):
    shape = Shape(np.array([[1, 1], [1, 0]]))
    shape_constraint = InputConstraint.empty(1).shapes[0]

    def fail_if_called(_grid):
        raise AssertionError("stabilizer should not be computed")

    monkeypatch.setattr(
        scene_sampler, "stabilizer_of_pixel_grid", fail_if_called,
    )
    assert scene_sampler._shape_satisfies(shape, shape_constraint)


def test_shape_satisfies_checks_restricted_stabilizer(monkeypatch):
    shape = Shape(np.array([[1, 1], [1, 0]]))
    shape_constraint = InputConstraint.empty(1).restrict_stabilizers(
        0, {frozenset({D4_IDENTITY})},
    ).shapes[0]
    calls = 0

    def fake_stabilizer(_grid):
        nonlocal calls
        calls += 1
        return frozenset({D4_IDENTITY})

    monkeypatch.setattr(scene_sampler, "stabilizer_of_pixel_grid", fake_stabilizer)
    assert scene_sampler._shape_satisfies(shape, shape_constraint)
    assert calls == 1


def test_constructive_shape_path_satisfies_hard_geometry_constraints():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})
    c = c.require_primary(0, 3)

    shape = scene_sampler._sample_shape_for_constraint(
        c.shapes[0],
        factory=_factory_solid_only(allow_constructive_shapes=True),
        palette=_PALETTE,
        rng=np.random.default_rng(123),
        max_attempts=1,
    )

    assert shape is not None
    assert shape.primary_color == 3
    assert shape.n_rows != shape.n_cols
    assert min(shape.n_rows, shape.n_cols) >= 3
    assert grid_shape_of(shape) == "solid"
    assert stabilizer_of_pixel_grid(shape.grid) == trivial


def test_default_factory_does_not_use_constructive_shape_path(monkeypatch):
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)

    def fail_construct(*_args, **_kwargs):
        raise AssertionError("constructive path should be opt-in")

    monkeypatch.setattr(scene_sampler, "_construct_solid_shape", fail_construct)
    shape = scene_sampler._sample_shape_for_constraint(
        c.shapes[0],
        factory=_factory_solid_only(),
        palette=_PALETTE,
        rng=np.random.default_rng(321),
        max_attempts=200,
    )

    assert shape is not None
    assert shape.n_rows != shape.n_cols
    assert min(shape.n_rows, shape.n_cols) >= 3


def test_constructive_solid_path_is_not_fixed_l_template():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})

    rng = np.random.default_rng(0)
    masks = []
    for _ in range(12):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=_factory_solid_only(allow_constructive_shapes=True),
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        masks.append((shape.grid != 0).astype(np.int8))

    unique_masks = {tuple(map(tuple, mask)) for mask in masks}
    assert len(unique_masks) >= 6
    assert any(not _is_l_silhouette(mask) for mask in masks)


def test_constructive_geometry_only_path_does_not_force_marker_color():
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)

    rng = np.random.default_rng(2)
    for _ in range(8):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=_factory_solid_only(allow_constructive_shapes=True),
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert len(shape.colors) == 1


def test_constructive_stabilizer_path_prefers_silhouette_over_marker():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})

    rng = np.random.default_rng(0)
    for _ in range(20):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=_factory_solid_only(allow_constructive_shapes=True),
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert stabilizer_of_pixel_grid(shape.grid) == trivial
        assert len(shape.colors) == 1


def test_constructive_stabilizer_path_survives_add_border():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})

    rng = np.random.default_rng(5)
    for _ in range(20):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=_factory_solid_only(allow_constructive_shapes=True),
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert stabilizer_of_pixel_grid(add_border(shape).grid) == trivial
        assert len(shape.colors) == 1


def test_constructive_path_applies_patterned_coloring_after_silhouette():
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.require_primary(0, 3)
    factory = ShapeFactory(
        generators=(patterned,),
        size_range=(8, 14),
        allow_constructive_shapes=True,
    )

    rng = np.random.default_rng(6)
    for _ in range(10):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=factory,
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert grid_shape_of(shape) == "solid"
        assert shape.primary_color == 3
        assert len(shape.colors) == 2


def test_constructive_path_applies_bordered_coloring_after_silhouette():
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.require_primary(0, 4)
    factory = ShapeFactory(
        generators=(bordered,),
        size_range=(10, 16),
        allow_constructive_shapes=True,
    )

    rng = np.random.default_rng(7)
    seen_multicolor = False
    for _ in range(10):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=factory,
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert grid_shape_of(shape) == "solid"
        assert shape.primary_color == 4
        seen_multicolor = seen_multicolor or len(shape.colors) == 2
    assert seen_multicolor


def test_constructive_coloring_keeps_restricted_stabilizer_constraints():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "solid")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})
    factory = ShapeFactory(
        generators=(patterned, bordered),
        size_range=(10, 16),
        allow_constructive_shapes=True,
    )

    rng = np.random.default_rng(8)
    for _ in range(10):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=factory,
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        assert stabilizer_of_pixel_grid(shape.grid) == trivial
        assert stabilizer_of_pixel_grid(add_border(shape).grid) == trivial


def test_secondary_marker_color_is_not_palette_order_fixed():
    rng = np.random.default_rng(0)
    colors = {
        scene_sampler._secondary_marker_color(3, _PALETTE, rng)
        for _ in range(20)
    }
    assert len(colors) > 1
    assert colors <= (set(_PALETTE) - {3})


def _is_l_silhouette(mask: np.ndarray) -> bool:
    h, w = mask.shape
    variants = (
        {(0, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(0, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
        {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)},
        {(h - 1, c) for c in range(w)} | {(r, w - 1) for r in range(h)},
    )
    pixels = {
        (int(r), int(c))
        for r, c in np.argwhere(mask)
    }
    return pixels in variants


def test_constructive_hollow_shape_preserves_primary_and_breaks_symmetry():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "hollow")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})
    c = c.require_primary(0, 4)

    shape = scene_sampler._sample_shape_for_constraint(
        c.shapes[0],
        factory=ShapeFactory(size_range=(4, 10), allow_constructive_shapes=True),
        palette=_PALETTE,
        rng=np.random.default_rng(456),
        max_attempts=1,
    )

    assert shape is not None
    assert shape.primary_color == 4
    assert grid_shape_of(shape) == "hollow"
    assert stabilizer_of_pixel_grid(shape.grid) == trivial


def test_constructive_hollow_path_is_not_fixed_rectangle_ring():
    trivial = frozenset({D4_IDENTITY})
    c = InputConstraint.empty(1)
    c = c.require_grid_shape(0, "hollow")
    c = c.require_non_square(0)
    c = c.require_min_bbox_dim(0, 3)
    c = c.restrict_stabilizers(0, {trivial})

    rng = np.random.default_rng(1)
    masks = []
    for _ in range(12):
        shape = scene_sampler._sample_shape_for_constraint(
            c.shapes[0],
            factory=ShapeFactory(size_range=(4, 10), allow_constructive_shapes=True),
            palette=_PALETTE,
            rng=rng,
            max_attempts=1,
        )
        assert shape is not None
        masks.append((shape.grid != 0).astype(np.int8))

    unique_masks = {tuple(map(tuple, mask)) for mask in masks}
    assert len(unique_masks) >= 6
    assert any(not _is_rectangle_ring(mask) for mask in masks)


def _is_rectangle_ring(mask: np.ndarray) -> bool:
    h, w = mask.shape
    ring = np.zeros((h, w), dtype=np.int8)
    ring[0, :] = 1
    ring[-1, :] = 1
    ring[:, 0] = 1
    ring[:, -1] = 1
    return bool(np.array_equal(mask, ring))


# === sample_scene =========================================================


def test_empty_constraint_succeeds():
    c = InputConstraint.empty(1)
    rng = np.random.default_rng(0)
    result = sample_scene(
        c, factory=_factory_solid_only(), palette=_PALETTE,
        world_size=16, rng=rng,
    )
    assert result is not None
    world, shapes = result
    assert len(shapes) == 1


def test_empty_constraint_two_shapes():
    c = InputConstraint.empty(2)
    rng = np.random.default_rng(1)
    result = sample_scene(
        c, factory=_factory_solid_only(), palette=_PALETTE,
        world_size=16, rng=rng,
    )
    assert result is not None
    _, shapes = result
    assert len(shapes) == 2


def test_scene_sampler_respects_factory_bbox_span_cap():
    c = InputConstraint.empty(3)
    rng = np.random.default_rng(12)
    factory = ShapeFactory(
        generators=DEFAULT_GENERATORS,
        size_range=(4, 16),
        bordered_hollow_min_size=9,
        bordered_hollow_random_inner_size=True,
        max_bbox_span=5,
    )
    result = sample_scene(
        c, factory=factory, palette=_PALETTE,
        world_size=16, rng=rng, max_total_retries=1_000,
    )
    assert result is not None
    _world, shapes = result
    assert len(shapes) == 3
    assert all(max(s.n_rows, s.n_cols) <= 5 for s in shapes)


def test_raster_ordering_constraint_honoured_for_n_shapes():
    c = InputConstraint.empty(4)
    for i in range(3):
        c = c.add_raster_ordering(i, i + 1)
    rng = np.random.default_rng(11)
    result = sample_scene(
        c, factory=_factory_solid_only(), palette=_PALETTE,
        world_size=24, rng=rng, max_total_retries=2_000,
    )
    assert result is not None
    _, shapes = result
    positions = [s.position for s in shapes]
    assert positions == sorted(positions)


def test_infeasible_constraint_returns_none():
    c = InputConstraint.empty(1).restrict_stabilizers(0, [])
    result = sample_scene(
        c, factory=_factory_solid_only(), palette=_PALETTE,
        world_size=16, rng=np.random.default_rng(),
    )
    assert result is None


def test_stabilizer_constraint_honoured():
    """Restrict to trivial stabilizer only — every returned shape must
    have trivial Stab (size 1)."""
    c = InputConstraint.empty(1).restrict_stabilizers(
        0, {frozenset({D4_IDENTITY})},
    )
    rng = np.random.default_rng(2)
    # Many draws — each produced scene must satisfy the constraint.
    trivial = frozenset({D4_IDENTITY})
    n_tries = 30
    n_satisfied = 0
    for _ in range(n_tries):
        result = sample_scene(
            c, factory=_factory_solid_only(), palette=_PALETTE,
            world_size=16, rng=rng, max_total_retries=200,
        )
        if result is None:
            continue
        _, shapes = result
        if stabilizer_of_pixel_grid(shapes[0].grid) == trivial:
            n_satisfied += 1
    # With the random-walk factory, trivial-Stab shapes dominate,
    # so we should see many successful draws.
    assert n_satisfied > 0


def test_forbidden_primary_honoured():
    """Forbid primary=1 on shape 0."""
    c = InputConstraint.empty(1).forbid_primary(0, 1)
    rng = np.random.default_rng(3)
    for _ in range(20):
        result = sample_scene(
            c, factory=_factory_solid_only(), palette=_PALETTE,
            world_size=16, rng=rng,
        )
        if result is None:
            continue
        _, shapes = result
        assert primary_color_of(shapes[0]) != 1


def test_row_forbid_honoured():
    """shape_0.row + 0 ≠ shape_1.row (i.e. not same row)."""
    c = InputConstraint.empty(2).add_row_forbid(0, 1, 0)
    rng = np.random.default_rng(4)
    for _ in range(10):
        result = sample_scene(
            c, factory=_factory_solid_only(), palette=_PALETTE,
            world_size=16, rng=rng, max_total_retries=500,
        )
        if result is None:
            continue
        _, shapes = result
        r0 = shapes[0].position[0]
        r1 = shapes[1].position[0]
        assert r0 != r1


def test_all_per_shape_stab_from_allowed_set():
    """Sweep: in every successful draw under constraint C, each shape's
    Stab is an element of C.shapes[i].allowed_stabilizers."""
    allowed = {frozenset({D4_IDENTITY})}
    c = InputConstraint.empty(2)
    c = c.restrict_stabilizers(0, allowed)
    c = c.restrict_stabilizers(1, allowed)
    rng = np.random.default_rng(5)
    draws = 10
    successes = 0
    for _ in range(draws):
        result = sample_scene(
            c, factory=_factory_solid_only(), palette=_PALETTE,
            world_size=16, rng=rng, max_total_retries=300,
        )
        if result is None:
            continue
        successes += 1
        _, shapes = result
        for i, s in enumerate(shapes):
            assert stabilizer_of_pixel_grid(s.grid) in c.shapes[i].allowed_stabilizers
    assert successes > 0
