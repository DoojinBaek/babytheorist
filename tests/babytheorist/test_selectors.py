"""Tests for babytheorist.selectors — Selector resolution logic."""

import sys; sys.path.insert(0, ".")
import numpy as np
import pytest
from babytheorist.selectors import select, select_one
from babytheorist.core import Shape
from babytheorist.core import Selector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(color: int, rows: int, cols: int, position: tuple[int, int] = (0, 0)) -> Shape:
    """Create a solid (non-hollow) rectangle of a single color."""
    grid = np.full((rows, cols), color, dtype=int)
    return Shape(grid, position)


def _hollow(color: int, size: int, position: tuple[int, int] = (0, 0)) -> Shape:
    """Create a hollow square frame (size >= 3)."""
    grid = np.full((size, size), color, dtype=int)
    grid[1:-1, 1:-1] = 0  # carve out interior
    return Shape(grid, position)


# ---------------------------------------------------------------------------
# ALL selector
# ---------------------------------------------------------------------------

class TestALL:
    def test_returns_all_indices(self):
        shapes = [_solid(1, 2, 2, (0, 0)), _solid(2, 2, 2, (3, 3))]
        assert select(Selector("ALL"), shapes) == [0, 1]

    def test_single_shape(self):
        shapes = [_solid(1, 2, 2)]
        assert select(Selector("ALL"), shapes) == [0]

    def test_empty_list(self):
        assert select(Selector("ALL"), []) == []


# ---------------------------------------------------------------------------
# by_color selector
# ---------------------------------------------------------------------------

class TestByColor:
    def test_selects_matching_color(self):
        shapes = [_solid(3, 2, 2, (0, 0)),
                  _solid(5, 2, 2, (3, 0)),
                  _solid(3, 2, 2, (6, 0))]
        result = select(Selector("by_color", param=3), shapes)
        assert result == [0, 2]

    def test_no_match_returns_empty(self):
        shapes = [_solid(1, 2, 2), _solid(2, 2, 2)]
        result = select(Selector("by_color", param=9), shapes)
        assert result == []

    def test_all_match(self):
        shapes = [_solid(4, 2, 2, (0, 0)), _solid(4, 3, 3, (5, 5))]
        result = select(Selector("by_color", param=4), shapes)
        assert result == [0, 1]

    def test_empty_list(self):
        assert select(Selector("by_color", param=1), []) == []


# ---------------------------------------------------------------------------
# is_hollow selector
# ---------------------------------------------------------------------------

class TestIsHollow:
    def test_selects_hollow_shapes(self):
        shapes = [_hollow(1, 3, (0, 0)),   # hollow
                  _solid(2, 2, 2, (5, 0)),  # solid
                  _hollow(3, 5, (10, 0))]   # hollow
        result = select(Selector("is_hollow"), shapes)
        assert result == [0, 2]

    def test_no_hollow_shapes(self):
        shapes = [_solid(1, 2, 2), _solid(2, 3, 3)]
        result = select(Selector("is_hollow"), shapes)
        assert result == []

    def test_all_hollow(self):
        shapes = [_hollow(1, 3), _hollow(2, 4)]
        result = select(Selector("is_hollow"), shapes)
        assert result == [0, 1]

    def test_empty_list(self):
        assert select(Selector("is_hollow"), []) == []


# ---------------------------------------------------------------------------
# topmost selector
# ---------------------------------------------------------------------------

class TestTopmost:
    def test_basic(self):
        shapes = [_solid(1, 2, 2, (5, 0)),
                  _solid(2, 2, 2, (1, 0)),
                  _solid(3, 2, 2, (3, 0))]
        result = select(Selector("topmost"), shapes)
        assert result == [1]

    def test_tie_break_by_col(self):
        # Same row → pick smallest column
        shapes = [_solid(1, 2, 2, (0, 5)),
                  _solid(2, 2, 2, (0, 2)),
                  _solid(3, 2, 2, (0, 8))]
        result = select(Selector("topmost"), shapes)
        assert result == [1]  # col 2 is smallest

    def test_single_shape(self):
        shapes = [_solid(1, 2, 2, (7, 3))]
        result = select(Selector("topmost"), shapes)
        assert result == [0]

    def test_empty_list(self):
        assert select(Selector("topmost"), []) == []


# ---------------------------------------------------------------------------
# bottommost selector
# ---------------------------------------------------------------------------

class TestBottommost:
    def test_basic(self):
        shapes = [_solid(1, 2, 2, (1, 0)),
                  _solid(2, 2, 2, (5, 0)),
                  _solid(3, 2, 2, (3, 0))]
        result = select(Selector("bottommost"), shapes)
        assert result == [1]

    def test_tie_break_by_col(self):
        # Same row → pick largest column
        shapes = [_solid(1, 2, 2, (5, 2)),
                  _solid(2, 2, 2, (5, 8)),
                  _solid(3, 2, 2, (5, 5))]
        result = select(Selector("bottommost"), shapes)
        assert result == [1]  # col 8 is largest

    def test_empty_list(self):
        assert select(Selector("bottommost"), []) == []


# ---------------------------------------------------------------------------
# leftmost selector
# ---------------------------------------------------------------------------

class TestLeftmost:
    def test_basic(self):
        shapes = [_solid(1, 2, 2, (0, 5)),
                  _solid(2, 2, 2, (0, 1)),
                  _solid(3, 2, 2, (0, 3))]
        result = select(Selector("leftmost"), shapes)
        assert result == [1]

    def test_tie_break_by_row(self):
        # Same col → pick smallest row
        shapes = [_solid(1, 2, 2, (5, 0)),
                  _solid(2, 2, 2, (2, 0)),
                  _solid(3, 2, 2, (8, 0))]
        result = select(Selector("leftmost"), shapes)
        assert result == [1]  # row 2 is smallest

    def test_empty_list(self):
        assert select(Selector("leftmost"), []) == []


# ---------------------------------------------------------------------------
# rightmost selector
# ---------------------------------------------------------------------------

class TestRightmost:
    def test_basic(self):
        shapes = [_solid(1, 2, 2, (0, 1)),
                  _solid(2, 2, 2, (0, 7)),
                  _solid(3, 2, 2, (0, 4))]
        result = select(Selector("rightmost"), shapes)
        assert result == [1]

    def test_tie_break_by_row(self):
        # Same col → pick largest row
        shapes = [_solid(1, 2, 2, (2, 5)),
                  _solid(2, 2, 2, (8, 5)),
                  _solid(3, 2, 2, (5, 5))]
        result = select(Selector("rightmost"), shapes)
        assert result == [1]  # row 8 is largest

    def test_empty_list(self):
        assert select(Selector("rightmost"), []) == []


# ---------------------------------------------------------------------------
# by_index selector — 1-based raster index into the list order
# ---------------------------------------------------------------------------

class TestByIndex:
    def test_picks_k_minus_one_in_list_order(self):
        shapes = [_solid(1, 2, 2, (0, 0)),
                  _solid(2, 2, 2, (5, 5)),
                  _solid(3, 2, 2, (8, 8))]
        assert select(Selector("by_index", param=1), shapes) == [0]
        assert select(Selector("by_index", param=3), shapes) == [2]

    def test_out_of_range_returns_empty(self):
        shapes = [_solid(1, 2, 2, (0, 0))]
        assert select(Selector("by_index", param=2), shapes) == []

    def test_uses_list_order_not_spatial(self):
        # Generator is expected to raster-sort at input time; the selector
        # itself is purely list-positional.
        shapes = [_solid(1, 2, 2, (9, 9)),
                  _solid(2, 2, 2, (0, 0))]
        assert select(Selector("by_index", param=1), shapes) == [0]

    @pytest.mark.parametrize("param", [None, 0, -1])
    def test_invalid_param_raises(self, param):
        with pytest.raises(ValueError):
            Selector("by_index") if param is None else Selector("by_index", param=param)


# ---------------------------------------------------------------------------
# select_one
# ---------------------------------------------------------------------------

class TestSelectOne:
    def test_returns_index_for_single_selector(self):
        shapes = [_solid(1, 2, 2, (5, 0)),
                  _solid(2, 2, 2, (1, 0))]
        assert select_one(Selector("topmost"), shapes) == 1

    def test_returns_none_for_empty(self):
        assert select_one(Selector("bottommost"), []) is None

    def test_raises_for_ALL(self):
        shapes = [_solid(1, 2, 2)]
        with pytest.raises(ValueError, match="single-select"):
            select_one(Selector("ALL"), shapes)

    def test_raises_for_by_color(self):
        shapes = [_solid(1, 2, 2)]
        with pytest.raises(ValueError, match="single-select"):
            select_one(Selector("by_color", param=1), shapes)

    def test_raises_for_is_hollow(self):
        shapes = [_hollow(1, 3)]
        with pytest.raises(ValueError, match="single-select"):
            select_one(Selector("is_hollow"), shapes)

    def test_each_single_selector(self):
        shapes = [_solid(1, 2, 2, (3, 3))]
        for sel_type in ("topmost", "bottommost", "leftmost", "rightmost"):
            result = select_one(Selector(sel_type), shapes)
            assert result == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unknown_selector_type(self):
        """Manually construct an invalid selector to test error path."""
        sel = object.__new__(Selector)
        object.__setattr__(sel, "type", "nonexistent")
        object.__setattr__(sel, "param", None)
        shapes = [_solid(1, 2, 2)]
        with pytest.raises(ValueError, match="Unknown selector"):
            select(sel, shapes)

    def test_position_selector_deterministic_with_many_shapes(self):
        """Verify that position selectors are deterministic."""
        shapes = [
            _solid(1, 2, 2, (r, c))
            for r, c in [(3, 7), (1, 4), (5, 2), (1, 9), (5, 0)]
        ]
        # topmost: min row=1, then min col → (1,4) = index 1
        assert select(Selector("topmost"), shapes) == [1]
        # bottommost: max row=5, then max col → (5,2) = index 2
        assert select(Selector("bottommost"), shapes) == [2]
        # leftmost: min col=0, then min row → (5,0) = index 4
        assert select(Selector("leftmost"), shapes) == [4]
        # rightmost: max col=9, then max row → (1,9) = index 3
        assert select(Selector("rightmost"), shapes) == [3]
