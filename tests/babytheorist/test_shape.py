"""Tests for babytheorist.shape.Shape."""

import numpy as np
import pytest

from babytheorist.core import Shape


class TestShapeBasics:
    def test_create_shape(self):
        grid = np.array([[1, 1], [1, 0]])
        s = Shape(grid, (5, 10))
        assert s.n_rows == 2
        assert s.n_cols == 2
        assert s.position == (5, 10)

    def test_default_position(self):
        s = Shape(np.ones((3, 3), dtype=int))
        assert s.position == (0, 0)

    def test_is_null(self):
        s = Shape(np.zeros((2, 2), dtype=int))
        assert s.is_null
        s2 = Shape(np.array([[1]]))
        assert not s2.is_null

    def test_primary_color(self):
        grid = np.array([[1, 1, 2], [1, 2, 2], [1, 1, 1]])
        s = Shape(grid)
        assert s.primary_color == 1

    def test_primary_color_empty(self):
        s = Shape(np.zeros((2, 2), dtype=int))
        assert s.primary_color == 0

    def test_is_hollow(self):
        # Hollow rectangle
        grid = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])
        s = Shape(grid)
        assert s.is_hollow

    def test_is_not_hollow(self):
        grid = np.ones((3, 3), dtype=int)
        s = Shape(grid)
        assert not s.is_hollow

    def test_move_to(self):
        s = Shape(np.ones((2, 2), dtype=int), (0, 0))
        s2 = s.move_to((5, 10))
        assert s2.position == (5, 10)
        assert s.position == (0, 0)  # original unchanged
        np.testing.assert_array_equal(s.grid, s2.grid)

    def test_place_on(self):
        world = np.zeros((5, 5), dtype=int)
        s = Shape(np.array([[3, 3], [3, 0]]), (1, 2))
        s.place_on(world)
        assert world[1, 2] == 3
        assert world[1, 3] == 3
        assert world[2, 2] == 3
        assert world[2, 3] == 0  # zero stays zero

    def test_copy_is_independent(self):
        s = Shape(np.array([[1, 2], [3, 4]]), (1, 1))
        s2 = s.copy()
        s2.grid[0, 0] = 9
        assert s.grid[0, 0] == 1  # original unchanged

    def test_grid_is_copied_on_init(self):
        grid = np.array([[1, 2], [3, 4]])
        s = Shape(grid)
        grid[0, 0] = 99
        assert s.grid[0, 0] == 1  # not affected
