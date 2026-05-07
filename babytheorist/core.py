"""Core data types for BabyTheorist.

Three things live here:
    - Shape: a colored 2D bounding box with a world position
    - Selector + Step: how a program addresses objects in the scene
    - Task / TaskPair: the unit a benchmark is made of

Everything else (primitives, programs, shape generation, IO) imports these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi


Grid = np.ndarray  # 2D int array, values 0-9


# === Shape =================================================================


class Shape:
    """A colored shape + its top-left position in the world grid.

    `grid` is the shape's own bounding box (0 = empty, 1-9 = color); `position`
    is where the top-left of that box sits in world coordinates.
    """

    __slots__ = ("grid", "position")

    def __init__(self, grid: np.ndarray, position: tuple[int, int] = (0, 0)):
        self.grid = np.array(grid, dtype=np.int64, copy=True)
        self.position = position

    @property
    def n_rows(self) -> int:
        return self.grid.shape[0]

    @property
    def n_cols(self) -> int:
        return self.grid.shape[1]

    @property
    def is_null(self) -> bool:
        return not bool(np.any(self.grid))

    @property
    def colors(self) -> set[int]:
        """All non-zero colors present in this shape."""
        return set(int(c) for c in np.unique(self.grid) if c > 0)

    @property
    def primary_color(self) -> int:
        """Most frequent non-zero color (base/fill). 0 if shape is null."""
        nz = self.grid[self.grid > 0]
        if nz.size == 0:
            return 0
        vals, counts = np.unique(nz, return_counts=True)
        return int(vals[counts.argmax()])

    @property
    def secondary_color(self) -> int | None:
        """Second most frequent non-zero color (border/marker), or None."""
        nz = self.grid[self.grid > 0]
        if nz.size == 0:
            return None
        vals, counts = np.unique(nz, return_counts=True)
        if len(vals) < 2:
            return None
        order = np.argsort(-counts)
        return int(vals[order[1]])

    @property
    def is_hollow(self) -> bool:
        """True if the shape has interior holes (empty pixels enclosed by filled)."""
        mask = self.grid != 0
        return bool(np.any(ndi.binary_fill_holes(mask) & ~mask))

    def move_to(self, position: tuple[int, int]) -> "Shape":
        return Shape(self.grid, position)

    def place_on(self, world: np.ndarray) -> np.ndarray:
        """Stamp non-zero pixels onto `world`.

        Behavior is intentionally identical to the original BabyTheorist generator,
        which used a per-pixel loop with NumPy's negative indexing. As a
        result, negative positions wrap around (a known quirk of the legacy
        generator) and only positive out-of-bounds raises IndexError.
        """
        r, c = self.position
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                v = self.grid[i, j]
                if v > 0:
                    world[r + i, c + j] = v
        return world

    def copy(self) -> "Shape":
        return Shape(self.grid, self.position)

    def __repr__(self) -> str:
        return f"Shape({self.n_rows}x{self.n_cols}@{self.position})"


# === Selector ==============================================================


MULTI_SELECTORS: frozenset[str] = frozenset({"ALL", "by_color", "is_hollow"})
SINGLE_SELECTORS: frozenset[str] = frozenset({
    "topmost", "bottommost", "leftmost", "rightmost", "by_index",
})
SELECTOR_TYPES: frozenset[str] = MULTI_SELECTORS | SINGLE_SELECTORS


@dataclass(frozen=True)
class Selector:
    """Picks object(s) from a scene of Shapes.

    Attributes:
        type: one of SELECTOR_TYPES (ALL, by_color, is_hollow, topmost,
              bottommost, leftmost, rightmost, by_index).
        param: color int for `by_color`; 1-based raster index for `by_index`.
    """

    type: str
    param: int | None = None

    def __post_init__(self):
        if self.type not in SELECTOR_TYPES:
            raise ValueError(
                f"unknown selector type {self.type!r}; must be one of {sorted(SELECTOR_TYPES)}"
            )
        needs_param = self.type in ("by_color", "by_index")
        if needs_param and self.param is None:
            raise ValueError(f"{self.type!r} selector requires a param")
        if not needs_param and self.param is not None:
            raise ValueError(f"{self.type!r} selector takes no param (got {self.param!r})")
        if self.type == "by_index" and self.param < 1:
            raise ValueError(f"by_index param must be >= 1, got {self.param!r}")


# Convenience constructors
ALL = Selector("ALL")


def by_color(c: int) -> Selector:
    return Selector("by_color", param=int(c))


def is_hollow() -> Selector:
    return Selector("is_hollow")


def topmost() -> Selector:
    return Selector("topmost")


def bottommost() -> Selector:
    return Selector("bottommost")


def leftmost() -> Selector:
    return Selector("leftmost")


def rightmost() -> Selector:
    return Selector("rightmost")


# `by_index(k)` has no convenience constructor; call `Selector("by_index", k)`
# directly. The generator sorts scenes by raster position at input time so
# `by_index(k)` refers to the k-th top-left object and stays stable through
# program execution.


# === Step / Program ========================================================


# Stable across the registry — duplicating these three names here keeps `core`
# free of a `primitives` import, and `primitives.validate_registry()` asserts
# the registry matches this set at import time.
RELATIONAL_PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "swap_colors", "align_row", "align_col",
})


@dataclass(frozen=True)
class Step:
    """One program step.

    Transform/Movement:  Step(selector, primitive)
    Relational:          Step(selector, primitive, target=...)
    """

    selector: Selector
    primitive: str
    target: Selector | None = None

    def __post_init__(self):
        if self.primitive in RELATIONAL_PRIMITIVE_NAMES:
            if self.target is None:
                raise ValueError(
                    f"relational primitive {self.primitive!r} requires target selector"
                )
            # Relational ops act on exactly one src and one tgt. Multi-selectors
            # (ALL, by_color, is_hollow) return 0..N matches, so they cannot
            # pin down "the" object. Require single-selectors by type.
            # Future: allow multi-selectors that happen to match exactly one
            # object at runtime (runtime-checked in engine).
            if self.selector.type not in SINGLE_SELECTORS:
                raise ValueError(
                    f"relational primitive {self.primitive!r} requires a single-select "
                    f"selector for src, got {self.selector.type!r}"
                )
            if self.target.type not in SINGLE_SELECTORS:
                raise ValueError(
                    f"relational primitive {self.primitive!r} requires a single-select "
                    f"target selector, got {self.target.type!r}"
                )
        elif self.target is not None:
            raise ValueError(
                f"non-relational primitive {self.primitive!r} cannot have a target"
            )


Program = list[Step]


# === Task / TaskPair =======================================================


@dataclass
class TaskPair:
    """A single input-output grid pair. `n_shapes` is the generator's
    ground-truth object count — post-hoc CC counting is unreliable for
    shapes like `hollow`. Defaults to 0 for pairs loaded from older JSON."""

    input_grid: Grid
    output_grid: Grid
    n_shapes: int = 0


@dataclass
class Task:
    """A BabyTheorist task: N pairs sharing one program, tagged with curriculum level.

    `valid_query_outputs` is populated only for eval-mode tasks. Eval contract:
    **`pairs[:-1]` are support pairs (K ≥ 1), `pairs[-1]` is the query** (its
    output is the canonical `y_q`). The last-slot-is-query rule holds for
    any K. `valid_query_outputs` holds every grid reachable by a program that
    agrees with the canonical program on all support pairs — i.e. the set of
    answers a model could legitimately produce on the query given only the
    support set. `n_support` is populated for adaptive eval tasks to record K,
    even when K is constant across rows. It is `None` for train-mode tasks and
    legacy fixed-K eval tasks.
    """

    pairs: list[TaskPair]
    program: Program
    level: int
    valid_query_outputs: list[Grid] | None = None
    n_support: int | None = None
