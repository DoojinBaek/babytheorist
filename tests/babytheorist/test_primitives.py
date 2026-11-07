"""Tests for primitive implementations + metadata."""

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

from babytheorist.primitives import (
    PRIMITIVES,
    add_border,
    align_col,
    align_row,
    canceling_pairs,
    category_of,
    cyclic_orders,
    fill_interior,
    flip_h,
    flip_v,
    hollow,
    idempotents,
    involutions,
    next_color,
    primitives_by_category,
    rot90,
    swap_colors,
    translate_down,
    translate_left,
    translate_right,
    translate_up,
)
from babytheorist.core import Shape


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def square_shape():
    """3x3 solid block of colour 2 at position (3, 4)."""
    grid = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    return Shape(grid, (3, 4))


@pytest.fixture
def rect_shape():
    """2x3 rectangle, colour 5, at position (1, 2)."""
    grid = np.array([[5, 5, 5], [5, 5, 5]])
    return Shape(grid, (1, 2))


@pytest.fixture
def l_shape():
    """L-shaped piece with mixed colours (primary=1) at (0, 0).

    Grid:
        1 0
        1 0
        1 3
    """
    grid = np.array([[1, 0], [1, 0], [1, 3]])
    return Shape(grid, (0, 0))


@pytest.fixture
def hollow_ring():
    """Hollow 5x5 ring of colour 4.

    4 4 4 4 4
    4 0 0 0 4
    4 0 0 0 4
    4 0 0 0 4
    4 4 4 4 4
    """
    grid = np.array(
        [
            [4, 4, 4, 4, 4],
            [4, 0, 0, 0, 4],
            [4, 0, 0, 0, 4],
            [4, 0, 0, 0, 4],
            [4, 4, 4, 4, 4],
        ]
    )
    return Shape(grid, (2, 3))


# ==================================================================
# Object Transform tests
# ==================================================================


class TestRot90:
    def test_square_rotation(self, square_shape):
        result = rot90(square_shape)
        expected = np.rot90(square_shape.grid, 1)
        np.testing.assert_array_equal(result.grid, expected)

    def test_position_unchanged(self, square_shape):
        result = rot90(square_shape)
        assert result.position == square_shape.position

    def test_rect_rotation(self, rect_shape):
        result = rot90(rect_shape)
        assert result.grid.shape == (3, 2)  # (2,3) -> (3,2) after rot90
        expected = np.rot90(rect_shape.grid, 1)
        np.testing.assert_array_equal(result.grid, expected)

    def test_l_shape_rotation(self, l_shape):
        result = rot90(l_shape)
        expected = np.rot90(l_shape.grid, 1)
        np.testing.assert_array_equal(result.grid, expected)

    def test_four_rotations_identity(self, l_shape):
        s = l_shape
        for _ in range(4):
            s = rot90(s)
        np.testing.assert_array_equal(s.grid, l_shape.grid)


class TestFlipH:
    def test_flip_upside_down(self, l_shape):
        result = flip_h(l_shape)
        expected = np.flipud(l_shape.grid)
        np.testing.assert_array_equal(result.grid, expected)

    def test_position_unchanged(self, l_shape):
        result = flip_h(l_shape)
        assert result.position == l_shape.position

    def test_double_flip_identity(self, l_shape):
        result = flip_h(flip_h(l_shape))
        np.testing.assert_array_equal(result.grid, l_shape.grid)

    def test_symmetric_shape(self, square_shape):
        result = flip_h(square_shape)
        np.testing.assert_array_equal(result.grid, square_shape.grid)


class TestFlipV:
    def test_flip_left_right(self, l_shape):
        result = flip_v(l_shape)
        expected = np.fliplr(l_shape.grid)
        np.testing.assert_array_equal(result.grid, expected)

    def test_position_unchanged(self, l_shape):
        result = flip_v(l_shape)
        assert result.position == l_shape.position

    def test_double_flip_identity(self, rect_shape):
        result = flip_v(flip_v(rect_shape))
        np.testing.assert_array_equal(result.grid, rect_shape.grid)


class TestNextColor:
    def test_basic_cycling(self):
        grid = np.array([[1, 2], [3, 0]])
        s = Shape(grid, (0, 0))
        result = next_color(s)
        expected = np.array([[2, 3], [4, 0]])
        np.testing.assert_array_equal(result.grid, expected)

    def test_zero_stays_zero(self):
        grid = np.array([[0, 0], [0, 0]])
        s = Shape(grid, (0, 0))
        result = next_color(s)
        np.testing.assert_array_equal(result.grid, grid)

    def test_color_9_wraps_to_1(self):
        grid = np.array([[9, 8], [7, 0]])
        s = Shape(grid, (0, 0))
        result = next_color(s)
        expected = np.array([[1, 9], [8, 0]])
        np.testing.assert_array_equal(result.grid, expected)

    def test_all_colors_cycle(self):
        grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        s = Shape(grid, (0, 0))
        result = next_color(s)
        expected = np.array([[2, 3, 4], [5, 6, 7], [8, 9, 1]])
        np.testing.assert_array_equal(result.grid, expected)

    def test_position_unchanged(self, rect_shape):
        result = next_color(rect_shape)
        assert result.position == rect_shape.position

    def test_nine_cycles_returns_to_original(self):
        grid = np.array([[1, 5, 0], [9, 3, 0]])
        s = Shape(grid, (0, 0))
        for _ in range(9):
            s = next_color(s)
        np.testing.assert_array_equal(s.grid, grid)


class TestFillInterior:
    def test_fill_hollow_ring(self, hollow_ring):
        result = fill_interior(hollow_ring)
        # All cells should now be filled with primary_color (4)
        expected = np.full((5, 5), 4)
        np.testing.assert_array_equal(result.grid, expected)

    def test_non_hollow_raises(self, square_shape):
        with pytest.raises(ValueError, match="fill_interior requires a hollow shape"):
            fill_interior(square_shape)

    def test_position_preserved(self, hollow_ring):
        result = fill_interior(hollow_ring)
        assert result.position == hollow_ring.position

    def test_partial_hole(self):
        """Shape with a single interior hole pixel."""
        grid = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        s = Shape(grid, (5, 5))
        result = fill_interior(s)
        expected = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        np.testing.assert_array_equal(result.grid, expected)
        assert result.position == (5, 5)

    def test_open_shape_raises(self):
        """An open 'U' shape has no interior hole — raises ValueError."""
        grid = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]])
        s = Shape(grid, (0, 0))
        with pytest.raises(ValueError, match="fill_interior requires a hollow shape"):
            fill_interior(s)


class TestHollow:
    def test_solid_becomes_hollow(self, square_shape):
        result = hollow(square_shape)
        expected = np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]])
        np.testing.assert_array_equal(result.grid, expected)

    def test_already_hollow_unchanged(self, hollow_ring):
        result = hollow(hollow_ring)
        np.testing.assert_array_equal(result.grid, hollow_ring.grid)

    def test_l_shape_all_contour(self, l_shape):
        """L-shape has no interior pixel — hollow should leave it unchanged."""
        result = hollow(l_shape)
        np.testing.assert_array_equal(result.grid, l_shape.grid)

    def test_position_unchanged(self, square_shape):
        result = hollow(square_shape)
        assert result.position == square_shape.position

    def test_larger_solid(self):
        """4x4 solid → interior 2x2 removed."""
        grid = np.full((4, 4), 3)
        s = Shape(grid, (0, 0))
        result = hollow(s)
        expected = np.array(
            [[3, 3, 3, 3], [3, 0, 0, 3], [3, 0, 0, 3], [3, 3, 3, 3]]
        )
        np.testing.assert_array_equal(result.grid, expected)

    def test_thick_wall_with_hole_becomes_outer_ring(self):
        """Thick-walled shape with a small hole → entire interior becomes hollow."""
        grid = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ])
        s = Shape(grid, (0, 0))
        result = hollow(s)
        expected = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ])
        np.testing.assert_array_equal(result.grid, expected)


class TestAddBorder:
    def test_border_added(self):
        grid = np.array([[1, 1], [1, 1]])
        s = Shape(grid, (5, 5))
        result = add_border(s)
        # Grid grows from 2x2 to 4x4
        assert result.grid.shape == (4, 4)
        # Position shifts by (-1, -1)
        assert result.position == (4, 4)
        # Original pixels preserved
        np.testing.assert_array_equal(result.grid[1:-1, 1:-1], grid)
        # Border pixels filled with primary_color=1
        expected = np.array(
            [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]]
        )
        np.testing.assert_array_equal(result.grid, expected)

    def test_single_pixel(self):
        grid = np.array([[7]])
        s = Shape(grid, (3, 3))
        result = add_border(s)
        assert result.grid.shape == (3, 3)
        assert result.position == (2, 2)
        expected = np.array([[0, 7, 0], [7, 7, 7], [0, 7, 0]])
        np.testing.assert_array_equal(result.grid, expected)

    def test_l_shape_border(self, l_shape):
        result = add_border(l_shape)
        # Grid grows by 2 in each dimension
        assert result.grid.shape == (l_shape.n_rows + 2, l_shape.n_cols + 2)
        assert result.position == (-1, -1)
        # Original non-zero pixels preserved in center
        orig_nonzero = l_shape.grid != 0
        center = result.grid[1:-1, 1:-1]
        np.testing.assert_array_equal(
            center[orig_nonzero], l_shape.grid[orig_nonzero]
        )
        # Border picks adjacent pixel's color (up/down/left/right order)
        # L-shape [[1,0],[1,0],[1,3]]: concave 0 cells extend adjacent colors
        # (0,1) → adjacent (0,0)=1 left → 1
        # (1,1) → adjacent (0,1) now 1 (up) → 1 — wait, (0,1) is also 0 in original
        # Re-trace using only ORIGINAL neighbors: only 4-neighbors with
        # non-zero values in the original grid count.
        # (0,1): up=outside, down=(1,1)=0, left=(0,0)=1, right=outside → 1
        # (1,1): up=(0,1)=0, down=(2,1)=3, left=(1,0)=1, right=outside → 3 (first non-zero in order up,down,left,right)
        expected_center = np.array([[1, 1], [1, 3], [1, 3]])
        np.testing.assert_array_equal(center, expected_center)

    def test_grid_size_growth(self, rect_shape):
        result = add_border(rect_shape)
        assert result.n_rows == rect_shape.n_rows + 2
        assert result.n_cols == rect_shape.n_cols + 2

    def test_inner_hole_preserved(self):
        """Interior holes should NOT be filled — only outer border grows."""
        grid = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        s = Shape(grid, (5, 5))
        result = add_border(s)
        # Original hole at (1,1) stays 0 in new center (2,2)
        assert result.grid[2, 2] == 0

    def test_multicolor_border_extension(self):
        """Each border pixel takes the color of its adjacent source pixel."""
        # Simple rectangle with left column 1, middle 2, right 3
        grid = np.array([[1, 2, 3], [1, 2, 3]])
        s = Shape(grid, (0, 0))
        result = add_border(s)
        # Top border should extend colors 1, 2, 3 from below
        # top row (row 0 of expanded): positions (1,2,3) match original cols
        top = result.grid[0, 1:4]
        np.testing.assert_array_equal(top, [1, 2, 3])
        # Left border (col 0) extends from adjacent col 1 (color 1)
        left_col = result.grid[1:3, 0]
        np.testing.assert_array_equal(left_col, [1, 1])
        # Right border (col 4) extends from adjacent col 3 (color 3)
        right_col = result.grid[1:3, 4]
        np.testing.assert_array_equal(right_col, [3, 3])


# ==================================================================
# Movement tests (translate — move by own size, no bounds checking)
# ==================================================================


class TestTranslateUp:
    def test_translate_up_moves_by_height(self):
        """2x2 shape at (5, 3) → (3, 3)."""
        s = Shape(np.array([[1, 1], [1, 1]]), (5, 3))
        result = translate_up(s)
        assert result.position == (3, 3)

    def test_grid_unchanged(self, rect_shape):
        result = translate_up(rect_shape)
        np.testing.assert_array_equal(result.grid, rect_shape.grid)

    def test_translate_up_can_go_negative(self):
        """No bounds checking — position can be OOB."""
        s = Shape(np.array([[1, 1]]), (0, 5))
        result = translate_up(s)
        assert result.position == (-1, 5)


class TestTranslateDown:
    def test_translate_down_moves_by_height(self):
        """2x2 shape at (5, 3) → (7, 3)."""
        s = Shape(np.array([[1, 1], [1, 1]]), (5, 3))
        result = translate_down(s)
        assert result.position == (7, 3)

    def test_grid_unchanged(self, rect_shape):
        result = translate_down(rect_shape)
        np.testing.assert_array_equal(result.grid, rect_shape.grid)

    def test_translate_down_can_exceed_world(self):
        """No bounds checking — position can exceed any world size."""
        s = Shape(np.array([[3]]), (15, 0))
        result = translate_down(s)
        assert result.position == (16, 0)


class TestTranslateLeft:
    def test_translate_left_moves_by_width(self):
        """2x2 shape at (5, 7) → (5, 5)."""
        s = Shape(np.array([[1, 1], [1, 1]]), (5, 7))
        result = translate_left(s)
        assert result.position == (5, 5)

    def test_grid_unchanged(self, rect_shape):
        result = translate_left(rect_shape)
        np.testing.assert_array_equal(result.grid, rect_shape.grid)


class TestTranslateRight:
    def test_translate_right_moves_by_width(self):
        """2x2 shape at (5, 3) → (5, 5)."""
        s = Shape(np.array([[1, 1], [1, 1]]), (5, 3))
        result = translate_right(s)
        assert result.position == (5, 5)

    def test_grid_unchanged(self, rect_shape):
        result = translate_right(rect_shape)
        np.testing.assert_array_equal(result.grid, rect_shape.grid)


class TestTranslateCompositions:
    def test_translate_up_then_left(self):
        """2x3 shape at (5, 7): translate_up then translate_left → (3, 4)."""
        s = Shape(np.array([[1, 1, 1], [1, 1, 1]]), (5, 7))
        result = translate_left(translate_up(s))
        assert result.position == (
            5 - s.n_rows,   # 5 - 2 = 3
            7 - s.n_cols,   # 7 - 3 = 4
        )

    def test_translate_up_then_down_cancels(self):
        """translate_up followed by translate_down returns to original position."""
        s = Shape(np.array([[1, 1], [1, 1]]), (5, 3))
        result = translate_down(translate_up(s))
        assert result.position == s.position

    def test_translate_left_then_right_cancels(self):
        """translate_left followed by translate_right returns to original position."""
        s = Shape(np.array([[2, 2, 2]]), (4, 6))
        result = translate_right(translate_left(s))
        assert result.position == s.position


# ==================================================================
# Relational tests
# ==================================================================


class TestSwapColors:
    def test_basic_swap(self):
        src = Shape(np.array([[1, 1], [1, 0]]), (0, 0))
        tgt = Shape(np.array([[3, 3], [3, 3]]), (5, 5))
        new_src, new_tgt = swap_colors(src, tgt)
        # Source's primary (1) → target's primary (3)
        expected_src = np.array([[3, 3], [3, 0]])
        expected_tgt = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(new_src.grid, expected_src)
        np.testing.assert_array_equal(new_tgt.grid, expected_tgt)

    def test_positions_preserved(self):
        src = Shape(np.array([[2]]), (1, 2))
        tgt = Shape(np.array([[7]]), (3, 4))
        new_src, new_tgt = swap_colors(src, tgt)
        assert new_src.position == (1, 2)
        assert new_tgt.position == (3, 4)

    def test_same_color_raises(self):
        src = Shape(np.array([[5, 5]]), (0, 0))
        tgt = Shape(np.array([[5, 5]]), (0, 0))
        with pytest.raises(ValueError, match="distinct primary colors"):
            swap_colors(src, tgt)

    def test_secondary_colors_untouched(self):
        """Secondary colours that aren't the primary should remain."""
        src = Shape(np.array([[1, 1, 9], [1, 1, 9]]), (0, 0))  # primary=1
        tgt = Shape(np.array([[3, 3]]), (0, 0))  # primary=3
        new_src, _ = swap_colors(src, tgt)
        # 1→3 but 9 stays
        expected = np.array([[3, 3, 9], [3, 3, 9]])
        np.testing.assert_array_equal(new_src.grid, expected)

    def test_double_swap_identity(self):
        src = Shape(np.array([[2, 2], [0, 2]]), (0, 0))
        tgt = Shape(np.array([[8, 8]]), (5, 5))
        new_src, new_tgt = swap_colors(src, tgt)
        orig_src, orig_tgt = swap_colors(new_src, new_tgt)
        np.testing.assert_array_equal(orig_src.grid, src.grid)
        np.testing.assert_array_equal(orig_tgt.grid, tgt.grid)


class TestAlignRow:
    def test_basic_alignment(self):
        src = Shape(np.array([[1, 1], [1, 1]]), (0, 0))  # center row = 0+1 = 1
        tgt = Shape(np.array([[2, 2], [2, 2]]), (6, 5))  # center row = 6+1 = 7
        new_src, new_tgt = align_row(src, tgt)
        # src center should move to tgt center
        new_center = new_src.position[0] + new_src.n_rows // 2
        tgt_center = tgt.position[0] + tgt.n_rows // 2
        assert new_center == tgt_center
        # Col unchanged
        assert new_src.position[1] == src.position[1]
        # Target unchanged
        assert new_tgt.position == tgt.position

    def test_already_aligned_raises(self):
        src = Shape(np.array([[1, 1]]), (5, 0))
        tgt = Shape(np.array([[2, 2]]), (5, 10))
        with pytest.raises(ValueError, match="different rows"):
            align_row(src, tgt)

    def test_grid_unchanged(self):
        src = Shape(np.array([[1, 2], [3, 4]]), (0, 0))
        tgt = Shape(np.array([[5]]), (10, 10))
        new_src, _ = align_row(src, tgt)
        np.testing.assert_array_equal(new_src.grid, src.grid)


class TestAlignCol:
    def test_basic_alignment(self):
        src = Shape(np.array([[1, 1, 1]]), (0, 0))  # center col = 0+1 = 1
        tgt = Shape(np.array([[2, 2, 2]]), (5, 8))  # center col = 8+1 = 9
        new_src, new_tgt = align_col(src, tgt)
        new_center = new_src.position[1] + new_src.n_cols // 2
        tgt_center = tgt.position[1] + tgt.n_cols // 2
        assert new_center == tgt_center
        # Row unchanged
        assert new_src.position[0] == src.position[0]
        # Target unchanged
        assert new_tgt.position == tgt.position

    def test_already_aligned_raises(self):
        src = Shape(np.array([[1]]), (0, 5))
        tgt = Shape(np.array([[2]]), (10, 5))
        with pytest.raises(ValueError, match="different cols"):
            align_col(src, tgt)

    def test_grid_unchanged(self):
        src = Shape(np.array([[7, 7]]), (0, 0))
        tgt = Shape(np.array([[3, 3, 3, 3]]), (5, 10))
        new_src, _ = align_col(src, tgt)
        np.testing.assert_array_equal(new_src.grid, src.grid)


# ==================================================================
# Registry tests
# ==================================================================


class TestRegistries:
    def test_transform_count(self):
        assert len(primitives_by_category("transform")) == 7

    def test_movement_count(self):
        assert len(primitives_by_category("movement")) == 4

    def test_movement_keys(self):
        expected = {"translate_up", "translate_down", "translate_left", "translate_right"}
        assert set(primitives_by_category("movement")) == expected

    def test_relational_count(self):
        assert len(primitives_by_category("relational")) == 3

    def test_total_count(self):
        assert len(PRIMITIVES) == 14

    def test_all_callable(self):
        for name, p in PRIMITIVES.items():
            assert callable(p.fn), f"{name} is not callable"

    def test_no_category_overlap(self):
        t = set(primitives_by_category("transform"))
        m = set(primitives_by_category("movement"))
        r = set(primitives_by_category("relational"))
        assert t.isdisjoint(m)
        assert t.isdisjoint(r)
        assert m.isdisjoint(r)

    def test_categories_partition_registry(self):
        union = (
            set(primitives_by_category("transform"))
            | set(primitives_by_category("movement"))
            | set(primitives_by_category("relational"))
        )
        assert union == set(PRIMITIVES)


# ==================================================================
# Edge cases & immutability
# ==================================================================


class TestImmutability:
    """Actions should not mutate their input shapes."""

    def test_rot90_no_mutation(self, l_shape):
        original = l_shape.grid.copy()
        rot90(l_shape)
        np.testing.assert_array_equal(l_shape.grid, original)

    def test_next_color_no_mutation(self, l_shape):
        original = l_shape.grid.copy()
        next_color(l_shape)
        np.testing.assert_array_equal(l_shape.grid, original)

    def test_hollow_no_mutation(self, square_shape):
        original = square_shape.grid.copy()
        hollow(square_shape)
        np.testing.assert_array_equal(square_shape.grid, original)

    def test_add_border_no_mutation(self, l_shape):
        original = l_shape.grid.copy()
        add_border(l_shape)
        np.testing.assert_array_equal(l_shape.grid, original)

    def test_fill_interior_no_mutation(self, hollow_ring):
        original = hollow_ring.grid.copy()
        fill_interior(hollow_ring)
        np.testing.assert_array_equal(hollow_ring.grid, original)

    def test_swap_colors_no_mutation(self):
        src = Shape(np.array([[1, 1]]), (0, 0))
        tgt = Shape(np.array([[3, 3]]), (5, 5))
        src_orig = src.grid.copy()
        tgt_orig = tgt.grid.copy()
        swap_colors(src, tgt)
        np.testing.assert_array_equal(src.grid, src_orig)
        np.testing.assert_array_equal(tgt.grid, tgt_orig)
class TestAlgebraicProperties:
    """Tests for algebraic property helper functions."""

    def test_involutions_members(self):
        assert set(involutions()) == {"flip_h", "flip_v", "swap_colors"}

    def test_idempotents_members(self):
        assert set(idempotents()) == {
            "fill_interior", "hollow", "align_row", "align_col",
        }

    def test_cyclic_orders(self):
        assert cyclic_orders() == {"rot90": 4, "next_color": 9}

    def test_canceling_pairs_members(self):
        expected = {
            ("translate_up", "translate_down"),
            ("translate_down", "translate_up"),
            ("translate_left", "translate_right"),
            ("translate_right", "translate_left"),
        }
        assert set(canceling_pairs()) == expected

    def test_no_overlap_involutions_idempotents(self):
        assert set(involutions()).isdisjoint(set(idempotents()))

    def test_canceling_pairs_not_in_idempotents(self):
        canceling_actions = {a for pair in canceling_pairs() for a in pair}
        assert canceling_actions.isdisjoint(set(idempotents()))

    def test_all_algebraic_primitives_registered(self):
        all_algebraic = (
            set(involutions())
            | set(idempotents())
            | set(cyclic_orders())
            | {a for pair in canceling_pairs() for a in pair}
        )
        for prim in all_algebraic:
            assert prim in PRIMITIVES, f"{prim} not in PRIMITIVES"


class TestUtilityFunctions:
    """Tests for category_of and primitives_by_category."""

    def test_category_transform(self):
        assert category_of("rot90") == "transform"

    def test_category_movement(self):
        assert category_of("translate_up") == "movement"

    def test_category_relational(self):
        assert category_of("swap_colors") == "relational"

    def test_category_unknown_raises(self):
        with pytest.raises(KeyError):
            category_of("nonexistent_action")

    def test_primitives_by_category_transform(self):
        actions = primitives_by_category("transform")
        assert len(actions) == 7
        assert "rot90" in actions

    def test_primitives_by_category_movement(self):
        actions = primitives_by_category("movement")
        assert len(actions) == 4
        assert "translate_left" in actions

    def test_primitives_by_category_relational(self):
        actions = primitives_by_category("relational")
        assert len(actions) == 3
        assert "align_row" in actions

    def test_primitives_by_category_invalid(self):
        with pytest.raises(ValueError):
            primitives_by_category("bogus")
