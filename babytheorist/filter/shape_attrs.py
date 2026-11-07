"""Shape attribute computation used by the scene rejection sampler.

Self-contained (only :mod:`numpy` + the :class:`D4` type). Holds its
own copy of the D4 group action on pixel grids and stabilizer code
so the ``filter/`` subpackage doesn't depend on :mod:`babytheorist.predicates`,
which will be slimmed down in Phase 11.
"""

from __future__ import annotations

import numpy as np

from babytheorist.symbolic import D4

from .d4_tables import D4_ELEMENTS, D4_IDENTITY


__all__ = [
    "apply_d4_to_grid",
    "stabilizer_of_pixel_grid",
    "stabilizer_of_silhouette",
]


def apply_d4_to_grid(grid: np.ndarray, g: D4) -> np.ndarray:
    """``g · grid`` — flip first (if ``g.flipped``), then rotate
    ``g.rot`` quarter turns CCW.

    Matches the BabyTheorist primitive convention::

        rot90  ↔ np.rot90(grid, 1)   (CCW)
        flip_h ↔ np.flipud            (s)
        flip_v ↔ np.fliplr            (r²s)
    """
    out = grid
    if g.flipped:
        out = np.flipud(out)
    if g.rot != 0:
        out = np.rot90(out, k=g.rot)
    return out


def stabilizer_of_pixel_grid(grid: np.ndarray) -> frozenset[D4]:
    """Subgroup ``{g ∈ D4 : g · grid == grid}`` for the colored grid.

    Non-square grids automatically exclude odd rotations (shape
    mismatch guard). Colors are part of the check — e.g. a silhouette-
    symmetric shape with asymmetric coloring has a smaller Stab.
    """
    result: set[D4] = set()
    for g in D4_ELEMENTS:
        if g == D4_IDENTITY:
            result.add(g)
            continue
        transformed = apply_d4_to_grid(grid, g)
        if transformed.shape == grid.shape and np.array_equal(transformed, grid):
            result.add(g)
    return frozenset(result)


def stabilizer_of_silhouette(grid: np.ndarray) -> frozenset[D4]:
    """Stab of the binary silhouette — a superset of
    :func:`stabilizer_of_pixel_grid`."""
    silhouette = (grid != 0).astype(grid.dtype)
    return stabilizer_of_pixel_grid(silhouette)
