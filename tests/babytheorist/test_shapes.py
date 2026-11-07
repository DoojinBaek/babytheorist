"""Tests for babytheorist.shapes — generators, ShapeFactory, place_shapes."""

import numpy as np
import pytest

from babytheorist.core import Shape
from babytheorist.shapes import (
    DEFAULT_GENERATORS, PATTERN_TYPES, ShapeFactory,
    bordered, hollow, patterned, place_shapes, random_walk, solid,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# === random_walk ==========================================================


class TestRandomWalk:
    def test_target_size_respected(self, rng):
        pixels = random_walk(10, rng)
        assert len(pixels) == 10

    def test_connectedness(self, rng):
        pixels = random_walk(12, rng)
        # Every pixel has a 4-neighbor in the set (except the first).
        for r, c in pixels:
            has_neighbor = any(
                (r + dr, c + dc) in pixels for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if pixels != {(0, 0)}:
                assert has_neighbor, f"({r},{c}) disconnected"

    def test_rejects_zero_size(self, rng):
        with pytest.raises(ValueError):
            random_walk(0, rng)


# === Per-mode generators ==================================================


class TestSolid:
    def test_single_color(self, rng):
        s = solid(rng, 8, [3])
        assert s.colors == {3}
        assert np.count_nonzero(s.grid) == 8

    def test_multi_palette_picks_one(self, rng):
        s = solid(rng, 6, [1, 2, 3, 4])
        assert len(s.colors) == 1


class TestBordered:
    def test_two_colors(self, rng):
        s = bordered(rng, 10, [5, 7])
        assert s.colors == {5, 7}

    def test_not_hollow(self, rng):
        s = bordered(rng, 10, [1, 2])
        assert not s.is_hollow  # inner is filled

    def test_palette_too_small(self, rng):
        with pytest.raises(ValueError):
            bordered(rng, 10, [1])


class TestHollow:
    def test_single_color(self, rng):
        s = hollow(rng, 10, [4])
        assert s.colors == {4}

    def test_actually_hollow(self, rng):
        # Use larger size so the inner walk has >= 2 pixels → non-trivial interior.
        s = hollow(rng, 20, [2])
        assert s.is_hollow


class TestPatterned:
    def test_two_colors(self, rng):
        s = patterned(rng, 16, [1, 2], pattern="checkerboard")
        assert s.colors <= {1, 2}

    def test_specific_patterns(self, rng):
        for pat in PATTERN_TYPES:
            s = patterned(rng, 16, [3, 4], pattern=pat)
            assert s.colors <= {3, 4}

    def test_unknown_pattern(self, rng):
        with pytest.raises(ValueError):
            patterned(rng, 10, [1, 2], pattern="spiral")


# === ShapeFactory =========================================================


class TestShapeFactory:
    def test_default_produces_shape(self, rng):
        f = ShapeFactory()
        s = f.sample(rng, [1, 2, 3, 4, 5])
        assert isinstance(s, Shape)

    def test_weights_must_match_generators(self):
        with pytest.raises(ValueError):
            ShapeFactory(generators=(solid, hollow), weights=(1.0,))

    def test_empty_generators(self):
        with pytest.raises(ValueError):
            ShapeFactory(generators=())

    def test_mode_distribution_uniform(self, rng):
        # With only solid as generator, every output should be solid (1 color).
        f = ShapeFactory(generators=(solid,), size_range=(4, 6))
        for _ in range(20):
            s = f.sample(rng, [1, 2, 3])
            assert len(s.colors) == 1

    def test_custom_size_range(self, rng):
        f = ShapeFactory(generators=(solid,), size_range=(3, 4))
        s = f.sample(rng, [1])
        assert np.count_nonzero(s.grid) == 3

    def test_compactness_is_configurable(self, rng):
        f = ShapeFactory(generators=(solid,), size_range=(8, 9), compactness=0.0)
        s = f.sample(rng, [1])
        assert np.count_nonzero(s.grid) == 8
        assert f.compactness == 0.0

    def test_bordered_hollow_min_size_applies_to_builtin_generators(self, rng):
        baseline = ShapeFactory(
            generators=(hollow,),
            size_range=(4, 5),
        ).sample(np.random.default_rng(0), [1])
        f = ShapeFactory(
            generators=(hollow,),
            size_range=(4, 5),
            bordered_hollow_min_size=9,
        )
        s = f.sample(rng, [1])
        assert np.count_nonzero(s.grid) > np.count_nonzero(baseline.grid)

    def test_rejects_invalid_shape_knobs(self):
        with pytest.raises(ValueError, match="compactness"):
            ShapeFactory(compactness=-1.0)
        with pytest.raises(ValueError, match="bordered_hollow_min_size"):
            ShapeFactory(bordered_hollow_min_size=0)
        with pytest.raises(ValueError, match="max_bbox_span"):
            ShapeFactory(max_bbox_span=0)

    def test_max_bbox_span_rejects_oversized_samples(self, rng):
        f = ShapeFactory(
            generators=(solid,),
            size_range=(12, 16),
            max_bbox_span=5,
        )
        for _ in range(20):
            s = f.sample(rng, [1])
            assert max(s.n_rows, s.n_cols) <= 5


# === place_shapes =========================================================


class TestPlaceShapes:
    def test_places_all_shapes(self, rng):
        factory = ShapeFactory(generators=(solid,), size_range=(3, 5))
        shapes = [factory.sample(rng, [1, 2, 3]) for _ in range(3)]
        result = place_shapes(shapes, world_size=20, gap=2, rng=rng)
        assert result is not None
        world, placed = result
        assert len(placed) == 3
        for s in placed:
            assert 2 <= s.position[0] <= 20 - s.n_rows - 1
            assert 2 <= s.position[1] <= 20 - s.n_cols - 1

    def test_all_or_nothing_when_too_tight(self, rng):
        # 16x16 world, gap=2, shapes that don't fit
        big = Shape(np.ones((15, 15), dtype=np.int64))
        result = place_shapes([big, big], world_size=16, gap=2, rng=rng)
        assert result is None

    def test_gap_enforced_between_shapes(self, rng):
        factory = ShapeFactory(generators=(solid,), size_range=(3, 4))
        shapes = [factory.sample(rng, [1, 2]) for _ in range(5)]
        result = place_shapes(shapes, world_size=30, gap=2, rng=rng)
        assert result is not None
        _, placed = result
        # Pairwise: bounding boxes must be separated by at least 1 gap row/col.
        for i, a in enumerate(placed):
            for b in placed[i + 1:]:
                a_r, a_c = a.position
                b_r, b_c = b.position
                row_overlap = not (a_r + a.n_rows <= b_r or b_r + b.n_rows <= a_r)
                col_overlap = not (a_c + a.n_cols <= b_c or b_c + b.n_cols <= a_c)
                assert not (row_overlap and col_overlap), "shapes overlap"
