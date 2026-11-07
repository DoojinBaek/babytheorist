"""Program execution engine for BabyTheorist.

Connects selectors and actions to execute Step-based programs.
Handles collision detection — objects never overlap after any step.

    apply_step(step, shapes, world_size) → list[Shape]
    apply_program(program, shapes, world_size) → (output_grid, grid_sequence)
"""

from __future__ import annotations

import numpy as np

from .core import Program, Shape, Step
from .primitives import PRIMITIVES
from .selectors import select

ENGINE_FAILURE_SELECTOR_EMPTY = "engine_selector_empty"
ENGINE_FAILURE_NOOP = "engine_noop"
ENGINE_FAILURE_COLLISION_OR_OOB = "engine_collision_or_oob"
ENGINE_FAILURE_SAME_SHAPE = "engine_same_shape"
ENGINE_FAILURE_PRECONDITION = "engine_precondition"
ENGINE_FAILURE_REBUILD = "engine_rebuild"


# ---------------------------------------------------------------------------
# Collision helpers
# ---------------------------------------------------------------------------


def _occupied_mask(
    shapes: list[Shape],
    exclude: set[int],
    world_size: int,
) -> np.ndarray:
    """Build a boolean mask of occupied cells, excluding given indices."""
    mask = np.zeros((world_size, world_size), dtype=bool)
    for i, s in enumerate(shapes):
        if i in exclude:
            continue
        r, c = s.position
        for di in range(s.n_rows):
            for dj in range(s.n_cols):
                if s.grid[di, dj] > 0:
                    wr, wc = r + di, c + dj
                    if 0 <= wr < world_size and 0 <= wc < world_size:
                        mask[wr, wc] = True
    return mask


def _shape_fits(
    shape: Shape,
    pos: tuple[int, int],
    occupied: np.ndarray,
    world_size: int,
) -> bool:
    """Check if shape at *pos* fits without overlap or out-of-bounds."""
    r, c = pos
    for i in range(shape.n_rows):
        for j in range(shape.n_cols):
            if shape.grid[i, j] > 0:
                wr, wc = r + i, c + j
                if wr < 0 or wr >= world_size or wc < 0 or wc >= world_size:
                    return False
                if occupied[wr, wc]:
                    return False
    return True


def _add_to_mask(shape: Shape, occupied: np.ndarray) -> None:
    """Mark shape's pixels as occupied (in-place)."""
    r, c = shape.position
    ws = occupied.shape[0]
    for i in range(shape.n_rows):
        for j in range(shape.n_cols):
            if shape.grid[i, j] > 0:
                wr, wc = r + i, c + j
                if 0 <= wr < ws and 0 <= wc < ws:
                    occupied[wr, wc] = True


def _remove_from_mask(shape: Shape, occupied: np.ndarray) -> None:
    """Unmark shape's pixels from occupied (in-place)."""
    r, c = shape.position
    ws = occupied.shape[0]
    for i in range(shape.n_rows):
        for j in range(shape.n_cols):
            if shape.grid[i, j] > 0:
                wr, wc = r + i, c + j
                if 0 <= wr < ws and 0 <= wc < ws:
                    occupied[wr, wc] = False


def _shape_changed(before: Shape, after: Shape) -> bool:
    """True if `after` differs from `before` in position or grid."""
    return before.position != after.position or not np.array_equal(before.grid, after.grid)


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------


def apply_step(
    step: Step,
    shapes: list[Shape],
    world_size: int,
) -> list[Shape] | None:
    """Apply one program step to a list of shapes.

    Resolves the selector to pick target shapes, then applies the primitive.
    Returns a new list (original shapes are not mutated), or None if the
    step has no observable effect. The exact failure conditions depend on
    the primitive category:

      - the selector (or, for relational, src or tgt) matches no shape
      - transform: the primitive returns the same shape unchanged
        (e.g. rot90 on a 4-fold symmetric shape, hollow on an already
        hollow shape), OR the transformed shape would collide with another
        shape or go out of bounds. `place_shapes` enforces a pixel-gap
        constraint rather than bbox separation, so bboxes may overlap
        initially even when pixels are apart — and a transform can move
        pixels into a neighbour's cells. Caught here.
      - movement: the shape would collide with another or go out of bounds
      - relational: src and tgt resolve to the same index, or align would
        collide. (align_* intentionally leaves tgt unchanged by design;
        swap_colors changes both via the distinct-color precondition.)
      - a primitive precondition raises ValueError (e.g. fill_interior
        on a non-hollow shape)
    """
    result, _reason = _apply_step_with_failure_reason(step, shapes, world_size)
    return result


def _apply_step_with_failure_reason(
    step: Step,
    shapes: list[Shape],
    world_size: int,
) -> tuple[list[Shape] | None, str | None]:
    """Apply one step and return a machine-readable failure reason."""
    result = [s.copy() for s in shapes]
    name = step.primitive
    if name not in PRIMITIVES:
        raise ValueError(f"Unknown primitive: {name!r}")
    prim = PRIMITIVES[name]
    fn = prim.fn

    try:
        if prim.category == "transform":
            indices = select(step.selector, result)
            if not indices:
                return None, ENGINE_FAILURE_SELECTOR_EMPTY
            occupied = _occupied_mask(result, set(), world_size)
            for i in indices:
                _remove_from_mask(result[i], occupied)
                before = result[i]
                after = fn(before)
                if not _shape_changed(before, after):
                    return None, ENGINE_FAILURE_NOOP
                if not _shape_fits(after, after.position, occupied, world_size):
                    return None, ENGINE_FAILURE_COLLISION_OR_OOB
                result[i] = after
                _add_to_mask(result[i], occupied)

        elif prim.category == "movement":
            indices = select(step.selector, result)
            if not indices:
                return None, ENGINE_FAILURE_SELECTOR_EMPTY
            occupied = _occupied_mask(result, set(), world_size)
            for i in indices:
                _remove_from_mask(result[i], occupied)
                candidate = fn(result[i])
                if not _shape_fits(candidate, candidate.position, occupied, world_size):
                    return None, ENGINE_FAILURE_COLLISION_OR_OOB
                result[i] = candidate
                _add_to_mask(result[i], occupied)

        else:  # relational — exactly one src, one tgt, distinct
            src_indices = select(step.selector, result)
            tgt_indices = select(step.target, result)
            if not src_indices or not tgt_indices:
                return None, ENGINE_FAILURE_SELECTOR_EMPTY
            src_idx, tgt_idx = src_indices[0], tgt_indices[0]
            if src_idx == tgt_idx:
                return None, ENGINE_FAILURE_SAME_SHAPE

            new_src, new_tgt = fn(result[src_idx], result[tgt_idx])
            if name == "swap_colors":
                result[src_idx] = new_src
                result[tgt_idx] = new_tgt
            else:  # align_row / align_col — tgt unchanged, src must fit
                occupied = _occupied_mask(result, {src_idx}, world_size)
                if not _shape_fits(new_src, new_src.position, occupied, world_size):
                    return None, ENGINE_FAILURE_COLLISION_OR_OOB
                result[src_idx] = new_src
    except ValueError:
        return None, ENGINE_FAILURE_PRECONDITION

    return result, None


# ---------------------------------------------------------------------------
# Grid rebuild
# ---------------------------------------------------------------------------


def _rebuild_grid(
    shapes: list[Shape],
    world_size: int,
) -> np.ndarray | None:
    """Place all shapes on a grid. Returns None if any shape is out of bounds."""
    grid = np.zeros((world_size, world_size), dtype=int)
    for s in shapes:
        if s.is_null:
            return None
        r, c = s.position
        for i in range(s.n_rows):
            for j in range(s.n_cols):
                if s.grid[i, j] > 0:
                    wr, wc = r + i, c + j
                    if wr < 0 or wr >= world_size or wc < 0 or wc >= world_size:
                        return None
                    grid[wr, wc] = s.grid[i, j]
    return grid


# ---------------------------------------------------------------------------
# Program execution
# ---------------------------------------------------------------------------


def apply_program(
    program: Program,
    shapes: list[Shape],
    world_size: int,
) -> tuple[np.ndarray | None, list[np.ndarray | None]]:
    """Apply a full program to shapes.

    Args:
        program: List of Steps to apply sequentially.
        shapes: Initial shapes (not mutated).
        world_size: Grid dimension (world_size x world_size).

    Returns:
        (output_grid, grid_sequence):
            output_grid: Final grid after all steps (None if invalid).
            grid_sequence: List of intermediate grids after each step.
    """
    current = [s.copy() for s in shapes]
    grid_sequence: list[np.ndarray | None] = []

    for step in program:
        current = apply_step(step, current, world_size)
        if current is None:
            # Step failed — rest of program is invalid
            remaining = len(program) - len(grid_sequence)
            grid_sequence.extend([None] * remaining)
            return None, grid_sequence
        grid = _rebuild_grid(current, world_size)
        grid_sequence.append(grid)

    output = grid_sequence[-1] if grid_sequence else _rebuild_grid(current, world_size)
    return output, grid_sequence


def apply_program_with_failure_reason(
    program: Program,
    shapes: list[Shape],
    world_size: int,
) -> tuple[np.ndarray | None, list[np.ndarray | None], str | None]:
    """Apply a program and report why execution failed, if it failed.

    The execution semantics intentionally mirror :func:`apply_program`; the
    extra reason is for generator diagnostics and audit metadata.
    """
    current = [s.copy() for s in shapes]
    grid_sequence: list[np.ndarray | None] = []
    rebuild_reason: str | None = None

    for step in program:
        current, reason = _apply_step_with_failure_reason(step, current, world_size)
        if current is None:
            remaining = len(program) - len(grid_sequence)
            grid_sequence.extend([None] * remaining)
            return None, grid_sequence, reason or ENGINE_FAILURE_PRECONDITION
        grid = _rebuild_grid(current, world_size)
        if grid is None and rebuild_reason is None:
            rebuild_reason = ENGINE_FAILURE_REBUILD
        grid_sequence.append(grid)

    output = grid_sequence[-1] if grid_sequence else _rebuild_grid(current, world_size)
    if output is None:
        return None, grid_sequence, rebuild_reason or ENGINE_FAILURE_REBUILD
    return output, grid_sequence, None


def apply_program_with_scenes(
    program: Program,
    shapes: list[Shape],
    world_size: int,
) -> tuple[np.ndarray | None, list[list[Shape]]]:
    """Apply a program and capture the shape state right before each step.

    Used by `babytheorist.evaluation` to compute per-step alternative selectors
    against the demo's actual scene at that point in execution.

    Returns:
        (output_grid, scenes_per_step):
            scenes_per_step[k] = deep-copied shape list immediately before
                step k is applied. Length equals the number of steps
                attempted (== len(program) on success; less on failure).
            output_grid: final grid; None if any step failed or the final
                scene cannot be rebuilt.
    """
    current = [s.copy() for s in shapes]
    scenes_per_step: list[list[Shape]] = []
    for step in program:
        scenes_per_step.append([s.copy() for s in current])
        current = apply_step(step, current, world_size)
        if current is None:
            return None, scenes_per_step
    return _rebuild_grid(current, world_size), scenes_per_step
