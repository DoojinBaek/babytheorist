"""Tests for the program execution engine."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from babytheorist.engine import (
    ENGINE_FAILURE_COLLISION_OR_OOB,
    ENGINE_FAILURE_NOOP,
    ENGINE_FAILURE_PRECONDITION,
    ENGINE_FAILURE_SAME_SHAPE,
    ENGINE_FAILURE_SELECTOR_EMPTY,
    apply_step, apply_program, apply_program_with_failure_reason, _rebuild_grid,
    _occupied_mask, _shape_fits,
)
from babytheorist.core import Shape
from babytheorist.core import Step, Selector, ALL, by_color, is_hollow, topmost, bottommost, leftmost, rightmost


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _solid(color, pos):
    """2x2 solid shape."""
    return Shape(np.full((2, 2), color, dtype=int), pos)


def _rect(color, rows, cols, pos):
    """Rectangle shape."""
    return Shape(np.full((rows, cols), color, dtype=int), pos)


def _hollow_shape(color, pos):
    """3x3 hollow shape (has interior hole)."""
    g = np.array([[color, color, color],
                  [color, 0,     color],
                  [color, color, color]])
    return Shape(g, pos)


# ===========================================================================
# apply_step — Object actions
# ===========================================================================

class TestApplyStepObject:

    def test_rot90_all(self):
        s = Shape(np.array([[1, 2], [3, 4]]), (3, 5))
        step = Step(ALL, "rot90")
        result = apply_step(step, [s], world_size=16)
        expected = np.rot90(s.grid, 1)
        np.testing.assert_array_equal(result[0].grid, expected)

    def test_flip_h_by_color(self):
        s1 = _solid(1, (0, 0))
        s2 = Shape(np.array([[2, 0], [0, 2]]), (4, 4))
        step = Step(by_color(2), "flip_h")
        result = apply_step(step, [s1, s2], world_size=16)
        np.testing.assert_array_equal(result[0].grid, s1.grid)
        np.testing.assert_array_equal(result[1].grid, np.flipud(s2.grid))

    def test_next_color(self):
        s = _solid(9, (0, 0))
        step = Step(ALL, "next_color")
        result = apply_step(step, [s], world_size=16)
        assert result[0].grid[0, 0] == 1

    def test_fill_interior(self):
        s = _hollow_shape(3, (2, 2))
        step = Step(ALL, "fill_interior")
        result = apply_step(step, [s], world_size=16)
        assert result[0].grid[1, 1] == 3

    def test_hollow(self):
        # 3x3 solid: hollow removes the center pixel.
        s = _rect(5, 3, 3, (0, 0))
        step = Step(ALL, "hollow")
        result = apply_step(step, [s], world_size=16)
        assert result[0].grid[1, 1] == 0
        assert result[0].grid[0, 0] == 5  # corners preserved

    def test_add_border(self):
        s = _solid(1, (5, 5))
        step = Step(ALL, "add_border")
        result = apply_step(step, [s], world_size=16)
        assert result[0].n_rows == 4
        assert result[0].n_cols == 4
        assert result[0].position == (4, 4)

    def test_selector_no_match(self):
        """Selector matches no shape → step has no effect → fails."""
        s = _solid(1, (0, 0))
        step = Step(by_color(9), "rot90")
        result = apply_step(step, [s], world_size=16)
        assert result is None


# ===========================================================================
# apply_step — Movement actions (translate by own size)
# ===========================================================================

class TestApplyStepMovement:

    def test_translate_up(self):
        s = _solid(1, (5, 3))  # 2x2
        step = Step(ALL, "translate_up")
        result = apply_step(step, [s], world_size=16)
        assert result[0].position == (3, 3)  # moved up by 2

    def test_translate_down(self):
        s = _solid(1, (5, 3))  # 2x2
        step = Step(ALL, "translate_down")
        result = apply_step(step, [s], world_size=16)
        assert result[0].position == (7, 3)  # moved down by 2

    def test_translate_left(self):
        s = _solid(1, (5, 7))  # 2x2
        step = Step(ALL, "translate_left")
        result = apply_step(step, [s], world_size=16)
        assert result[0].position == (5, 5)  # moved left by 2

    def test_translate_right(self):
        s = _solid(1, (5, 3))  # 2x2
        step = Step(ALL, "translate_right")
        result = apply_step(step, [s], world_size=16)
        assert result[0].position == (5, 5)  # moved right by 2

    def test_translate_by_shape_size(self):
        """Movement amount depends on shape dimensions."""
        s = _rect(1, 3, 5, (5, 5))  # 3 rows, 5 cols
        step_up = Step(ALL, "translate_up")
        result = apply_step(step_up, [s], world_size=16)
        assert result[0].position == (2, 5)  # moved up by 3 (n_rows)

        step_right = Step(ALL, "translate_right")
        result = apply_step(step_right, [s], world_size=16)
        assert result[0].position == (5, 10)  # moved right by 5 (n_cols)

    def test_translate_selective(self):
        s1 = _solid(1, (5, 5))
        s2 = _solid(2, (5, 10))
        step = Step(by_color(1), "translate_up")
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].position == (3, 5)  # moved
        assert result[1].position == (5, 10)  # unchanged

    def test_translate_out_of_bounds_fails(self):
        """If translate would go OOB, step fails."""
        s = _solid(1, (0, 0))
        step = Step(ALL, "translate_up")
        result = apply_step(step, [s], world_size=16)
        assert result is None

    def test_translate_collision_fails(self):
        """If translate would overlap another shape, step fails."""
        s1 = _solid(1, (3, 5))  # blocker
        s2 = _solid(2, (5, 5))  # mover wants to go up by 2 → (3,5) = collision
        step = Step(by_color(2), "translate_up")
        result = apply_step(step, [s1, s2], world_size=16)
        assert result is None

    def test_translate_canceling(self):
        """Translate up then down returns to original position."""
        s = _solid(1, (5, 5))
        prog = [Step(ALL, "translate_up"), Step(ALL, "translate_down")]
        output, _ = apply_program(prog, [s], world_size=16)
        assert output[5, 5] == 1  # back to original


# ===========================================================================
# apply_step — Relational actions
# ===========================================================================

class TestApplyStepRelational:

    def test_swap_colors(self):
        s1 = _solid(1, (0, 0))
        s2 = _solid(3, (5, 5))
        step = Step(topmost(), "swap_colors", target=bottommost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].primary_color == 3
        assert result[1].primary_color == 1

    def test_align_row(self):
        s1 = _solid(1, (0, 0))
        s2 = _solid(2, (8, 5))
        step = Step(topmost(), "align_row", target=bottommost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].position[0] == 8
        assert result[1].position == (8, 5)

    def test_align_col(self):
        s1 = _solid(1, (0, 0))
        s2 = _solid(2, (5, 10))
        step = Step(leftmost(), "align_col", target=rightmost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].position[1] == 10
        assert result[1].position == (5, 10)

    def test_relational_same_src_tgt_fails(self):
        """src and tgt resolve to the same index → step fails."""
        s1 = _solid(1, (0, 0))
        s2 = _solid(2, (5, 5))
        # topmost and leftmost both pick s1 here
        step = Step(topmost(), "swap_colors", target=leftmost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result is None

    def test_relational_requires_single_selectors(self):
        """Multi-selectors (ALL, by_color, is_hollow) rejected at Step creation."""
        with pytest.raises(ValueError):
            Step(ALL, "swap_colors", target=topmost())
        with pytest.raises(ValueError):
            Step(topmost(), "swap_colors", target=Selector("is_hollow"))
        with pytest.raises(ValueError):
            Step(by_color(3), "align_row", target=bottommost())

    def test_relational_single_shape_scene_fails(self):
        """Scene with one shape: every single-selector picks it, so src and
        tgt collapse to the same index → step fails."""
        s = _solid(1, (0, 0))
        step = Step(topmost(), "swap_colors", target=bottommost())
        result = apply_step(step, [s], world_size=16)
        assert result is None


# ===========================================================================
# _rebuild_grid
# ===========================================================================

class TestRebuildGrid:

    def test_basic(self):
        s1 = _solid(1, (0, 0))
        s2 = _solid(2, (5, 5))
        grid = _rebuild_grid([s1, s2], 10)
        assert grid is not None
        assert grid[0, 0] == 1
        assert grid[5, 5] == 2

    def test_out_of_bounds(self):
        s = _solid(1, (15, 15))
        grid = _rebuild_grid([s], 16)
        assert grid is None

    def test_null_shape(self):
        s = Shape(np.zeros((2, 2), dtype=int), (0, 0))
        grid = _rebuild_grid([s], 16)
        assert grid is None


# ===========================================================================
# apply_program — integration
# ===========================================================================

class TestApplyProgram:

    def test_empty_program(self):
        s = _solid(1, (5, 5))
        output, seq = apply_program([], [s], world_size=16)
        assert output is not None
        assert len(seq) == 0

    def test_single_step(self):
        s = _solid(1, (5, 5))
        prog = [Step(ALL, "translate_up")]
        output, seq = apply_program(prog, [s], world_size=16)
        assert output is not None
        assert output[3, 5] == 1  # moved up by 2
        assert output[5, 5] == 0
        assert len(seq) == 1

    def test_multi_step(self):
        s = _solid(1, (8, 8))
        prog = [
            Step(ALL, "translate_up"),
            Step(ALL, "translate_left"),
        ]
        output, seq = apply_program(prog, [s], world_size=16)
        assert output is not None
        assert output[6, 6] == 1  # up by 2, left by 2
        assert len(seq) == 2

    def test_selective_multi_step(self):
        s1 = _solid(1, (5, 2))
        s2 = _solid(2, (5, 10))
        prog = [
            Step(by_color(1), "translate_up"),
            Step(by_color(2), "translate_down"),
        ]
        output, seq = apply_program(prog, [s1, s2], world_size=16)
        assert output is not None
        assert output[3, 2] == 1   # s1 up by 2
        assert output[7, 10] == 2  # s2 down by 2

    def test_immutability(self):
        s = _solid(1, (5, 5))
        orig_pos = s.position
        orig_grid = s.grid.copy()
        prog = [Step(ALL, "translate_up"), Step(ALL, "next_color")]
        apply_program(prog, [s], world_size=16)
        assert s.position == orig_pos
        np.testing.assert_array_equal(s.grid, orig_grid)

    def test_out_of_bounds_step(self):
        s = _solid(1, (0, 0))
        prog = [Step(ALL, "add_border")]
        output, seq = apply_program(prog, [s], world_size=16)
        assert output is None
        assert seq[0] is None

    def test_relational_program(self):
        s1 = _solid(1, (2, 2))
        s2 = _solid(3, (8, 8))
        prog = [Step(topmost(), "swap_colors", target=bottommost())]
        output, seq = apply_program(prog, [s1, s2], world_size=16)
        assert output is not None
        assert output[2, 2] == 3
        assert output[8, 8] == 1

    def test_failure_reason_selector_empty(self):
        s = _solid(1, (5, 5))
        prog = [Step(by_color(9), "translate_up")]
        output, seq, reason = apply_program_with_failure_reason(
            prog, [s], world_size=16,
        )
        assert output is None
        assert seq == [None]
        assert reason == ENGINE_FAILURE_SELECTOR_EMPTY

    def test_failure_reason_collision_or_oob(self):
        s = _solid(1, (0, 0))
        prog = [Step(ALL, "translate_up")]
        output, seq, reason = apply_program_with_failure_reason(
            prog, [s], world_size=16,
        )
        assert output is None
        assert seq == [None]
        assert reason == ENGINE_FAILURE_COLLISION_OR_OOB

    def test_failure_reason_noop(self):
        s = _solid(1, (5, 5))
        prog = [Step(ALL, "rot90")]
        output, seq, reason = apply_program_with_failure_reason(
            prog, [s], world_size=16,
        )
        assert output is None
        assert seq == [None]
        assert reason == ENGINE_FAILURE_NOOP

    def test_failure_reason_precondition(self):
        s = _solid(1, (5, 5))
        prog = [Step(ALL, "fill_interior")]
        output, seq, reason = apply_program_with_failure_reason(
            prog, [s], world_size=16,
        )
        assert output is None
        assert seq == [None]
        assert reason == ENGINE_FAILURE_PRECONDITION

    def test_failure_reason_same_shape(self):
        s = _solid(1, (5, 5))
        prog = [Step(topmost(), "swap_colors", target=bottommost())]
        output, seq, reason = apply_program_with_failure_reason(
            prog, [s], world_size=16,
        )
        assert output is None
        assert seq == [None]
        assert reason == ENGINE_FAILURE_SAME_SHAPE


# ===========================================================================
# Collision detection
# ===========================================================================

class TestCollision:

    def test_translate_blocked_by_other_fails(self):
        """Translate into another object → step fails."""
        blocker = _solid(2, (3, 5))
        mover = _solid(1, (5, 5))  # wants to go up by 2 → (3, 5) = collision
        step = Step(by_color(1), "translate_up")
        result = apply_step(step, [blocker, mover], world_size=16)
        assert result is None

    def test_translate_no_collision_different_cols(self):
        """Objects in different columns don't block each other."""
        s1 = _solid(1, (5, 0))
        s2 = _solid(2, (5, 10))
        step = Step(ALL, "translate_up")
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].position == (3, 0)
        assert result[1].position == (3, 10)

    def test_translate_oob_fails(self):
        """Translate that goes out of bounds → step fails."""
        s = _rect(1, 3, 3, (1, 5))  # 3x3 at row 1, translate_up → row -2
        step = Step(ALL, "translate_up")
        result = apply_step(step, [s], world_size=16)
        assert result is None

    def test_align_row_blocked(self):
        """Align that would overlap → step fails (no silent no-op)."""
        s1 = _solid(1, (0, 5))
        s2 = _solid(2, (8, 5))  # same column!
        step = Step(topmost(), "align_row", target=bottommost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result is None

    def test_align_col_no_collision(self):
        """Align succeeds when no collision."""
        s1 = _solid(1, (0, 0))
        s2 = _solid(2, (5, 10))
        step = Step(topmost(), "align_col", target=bottommost())
        result = apply_step(step, [s1, s2], world_size=16)
        assert result[0].position == (0, 10)  # different rows, no collision


# ===========================================================================
# Step failure (primitive preconditions)
# ===========================================================================

class TestStepFailure:

    def test_fill_interior_on_solid_fails(self):
        """fill_interior on a non-hollow shape → step returns None."""
        s = _solid(1, (3, 3))  # solid 2x2, not hollow
        step = Step(ALL, "fill_interior")
        result = apply_step(step, [s], world_size=16)
        assert result is None

    def test_fill_interior_on_hollow_succeeds(self):
        """fill_interior on a hollow shape works fine."""
        g = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        s = Shape(g, (5, 5))
        step = Step(ALL, "fill_interior")
        result = apply_step(step, [s], world_size=16)
        assert result is not None
        assert result[0].grid[1, 1] == 1  # center filled

    def test_fill_interior_mixed_scene_fails(self):
        """Mixed scene: step fails if any selected shape is not hollow."""
        solid = _solid(1, (0, 0))
        hollow = Shape(np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]]), (5, 5))
        step = Step(ALL, "fill_interior")  # tries to fill both
        result = apply_step(step, [solid, hollow], world_size=16)
        assert result is None  # step fails because solid is not hollow

    def test_fill_interior_with_is_hollow_selector(self):
        """Using is_hollow selector avoids failure."""
        solid = _solid(1, (0, 0))
        hollow = Shape(np.array([[2, 2, 2], [2, 0, 2], [2, 2, 2]]), (5, 5))
        step = Step(is_hollow(), "fill_interior")  # only selects hollow ones
        result = apply_step(step, [solid, hollow], world_size=16)
        assert result is not None
        assert result[0].position == (0, 0)  # solid unchanged
        assert result[1].grid[1, 1] == 2  # hollow's center filled

    def test_program_with_failing_step_aborts(self):
        """Program aborts if any step fails."""
        s = _solid(1, (0, 0))
        prog = [
            Step(ALL, "rot90"),
            Step(ALL, "fill_interior"),  # will fail
            Step(ALL, "rot90"),
        ]
        output, seq = apply_program(prog, [s], world_size=16)
        assert output is None
        assert len(seq) == 3  # same length but None for failed + remaining
        assert seq[1] is None
        assert seq[2] is None
