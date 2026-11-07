"""The 14 BabyTheorist primitives, registered via `@primitive(...)`.

Three categories, each with a uniform signature:

    TRANSFORM  (7) — `Shape -> Shape`. Grid changes, position preserved.
                     rot90, flip_h, flip_v, next_color, fill_interior,
                     hollow, add_border.
    MOVEMENT   (4) — `Shape -> Shape`. Position changes by the shape's own
                     size. Grid unchanged. translate_up/down/left/right.
    RELATIONAL (3) — `(Shape, Shape) -> tuple[Shape, Shape]`. Two-object
                     interaction. swap_colors, align_row, align_col.

The decorator records each primitive's algebraic metadata (category,
cancels_with, involution, idempotent, cyclic_order) into `PRIMITIVES`.
Arity is inferred from the function signature: one `Shape` parameter →
transform or movement; two `Shape` parameters → relational.

Two-layer safety:
    1. `babytheorist.program.simplify` uses the metadata to filter redundant
       programs at generation time.
    2. Primitives raise `ValueError` when preconditions are unmet
       (`fill_interior` on a non-hollow shape, `swap_colors` on equal
       primary colors, `align_*` on already-aligned shapes). The engine
       catches this and aborts the step, so redundant programs that
       slip past `simplify` still fail cleanly.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Literal, get_args

import numpy as np
import scipy.ndimage as ndi

from .core import RELATIONAL_PRIMITIVE_NAMES, Shape


CATEGORIES: tuple[str, ...] = ("transform", "movement", "relational")
Category = Literal["transform", "movement", "relational"]
if get_args(Category) != CATEGORIES:
    raise RuntimeError("Category Literal must match CATEGORIES tuple")


__all__ = [
    "CATEGORIES", "Category",
    "Primitive", "PRIMITIVES", "primitive",
    "category_of", "primitives_by_category",
    "canceling_pairs", "involutions", "idempotents", "cyclic_orders",
    "validate_registry",
    "apply_primitive",
]


# === Registry ==============================================================


TransformFn = Callable[[Shape], Shape]
RelationalFn = Callable[[Shape, Shape], tuple[Shape, Shape]]


@dataclass(frozen=True)
class Primitive:
    fn: Callable                        # Shape→Shape or (Shape,Shape)→(Shape,Shape)
    category: Category
    cancels_with: tuple[str, ...] = ()  # A then B = identity
    involution: bool = False            # A then A = identity
    idempotent: bool = False            # A then A = A
    cyclic_order: int | None = None     # A^n = identity


PRIMITIVES: dict[str, Primitive] = {}


def primitive(
    category: Category,
    *,
    cancels_with: tuple[str, ...] = (),
    involution: bool = False,
    idempotent: bool = False,
    cyclic_order: int | None = None,
):
    """Decorator: register a BabyTheorist primitive.

    Arity is inferred from the function signature:
        1 parameter  → transform or movement (`Shape -> Shape`)
        2 parameters → relational (`(Shape, Shape) -> tuple[Shape, Shape]`)

    The function's `__name__` becomes the primitive's name (the key in
    `PRIMITIVES`); the function itself is returned unchanged.
    """
    def decorator(fn: Callable) -> Callable:
        name = fn.__name__
        if name in PRIMITIVES:
            raise ValueError(f"primitive {name!r} already registered")
        if category not in CATEGORIES:
            raise ValueError(
                f"unknown category {category!r}; must be one of {CATEGORIES}"
            )

        n_params = len(inspect.signature(fn).parameters)
        if category == "relational":
            if n_params != 2:
                raise ValueError(
                    f"relational primitive {name!r} must take 2 params (got {n_params})"
                )
        else:
            if n_params != 1:
                raise ValueError(
                    f"{category} primitive {name!r} must take 1 param (got {n_params})"
                )

        if cyclic_order is not None and (
            type(cyclic_order) is not int or cyclic_order < 2
        ):
            raise ValueError(
                f"primitive {name!r}: cyclic_order must be int >= 2, got {cyclic_order!r}"
            )

        PRIMITIVES[name] = Primitive(
            fn=fn,
            category=category,
            cancels_with=cancels_with,
            involution=involution,
            idempotent=idempotent,
            cyclic_order=cyclic_order,
        )
        return fn
    return decorator


# === Transform (7) =========================================================


@primitive("transform", cyclic_order=4)
def rot90(s: Shape) -> Shape:
    """Rotate 90° counter-clockwise. Position unchanged."""
    return Shape(np.rot90(s.grid, 1), s.position)


@primitive("transform", involution=True)
def flip_h(s: Shape) -> Shape:
    """Flip upside-down (horizontal axis). Position unchanged."""
    return Shape(np.flipud(s.grid), s.position)


@primitive("transform", involution=True)
def flip_v(s: Shape) -> Shape:
    """Flip left-right (vertical axis). Position unchanged."""
    return Shape(np.fliplr(s.grid), s.position)


@primitive("transform", cyclic_order=9)
def next_color(s: Shape) -> Shape:
    """Cycle all non-zero colors: new = (old % 9) + 1. Zero stays zero."""
    grid = s.grid.copy()
    mask = grid != 0
    grid[mask] = (grid[mask] % 9) + 1
    return Shape(grid, s.position)


@primitive("transform", idempotent=True)
def fill_interior(s: Shape) -> Shape:
    """Fill interior holes with `primary_color`.

    Raises `ValueError` if the shape has no interior holes — the caller
    should route around this via an `is_hollow` selector.
    """
    if not s.is_hollow:
        raise ValueError("fill_interior requires a hollow shape (no interior holes)")
    binary = s.grid != 0
    filled = ndi.binary_fill_holes(binary)
    grid = s.grid.copy()
    grid[filled & ~binary] = s.primary_color
    return Shape(grid, s.position)


@primitive("transform", idempotent=True)
def hollow(s: Shape) -> Shape:
    """Keep only the outer contour, hollowing everything inside.

    Interior holes are closed first (treated as solid), then the outer
    contour is extracted. Colors are preserved from the original grid.
    """
    binary = s.grid != 0
    filled = ndi.binary_fill_holes(binary)
    rows, cols = filled.shape
    contour = np.zeros_like(filled, dtype=bool)

    for i in range(rows):
        for j in range(cols):
            if not filled[i, j]:
                continue
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                contour[i, j] = True
                continue
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols and not filled[ni, nj]:
                    contour[i, j] = True
                    break

    return Shape(np.where(contour, s.grid, 0), s.position)


@primitive("transform")
def add_border(s: Shape) -> Shape:
    """Add a 1-pixel outer border. Interior holes are preserved.

    Each new border pixel takes the color of its adjacent source pixel
    so multi-color shapes extend naturally. Grid grows by 2 in each
    dimension; position shifts by (-1, -1).
    """
    rows, cols = s.grid.shape
    expanded = np.zeros((rows + 2, cols + 2), dtype=np.int64)
    expanded[1:-1, 1:-1] = s.grid

    binary = expanded != 0
    outside = ~ndi.binary_fill_holes(binary)

    ex_rows, ex_cols = expanded.shape
    result = expanded.copy()
    for i in range(ex_rows):
        for j in range(ex_cols):
            if not outside[i, j]:
                continue
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < ex_rows and 0 <= nj < ex_cols and expanded[ni, nj] != 0:
                    result[i, j] = expanded[ni, nj]
                    break

    return Shape(result, (s.position[0] - 1, s.position[1] - 1))


# === Movement (4) ==========================================================


@primitive("movement", cancels_with=("translate_down",))
def translate_up(s: Shape) -> Shape:
    """Move shape up by its own height."""
    r, c = s.position
    return s.move_to((r - s.n_rows, c))


@primitive("movement", cancels_with=("translate_up",))
def translate_down(s: Shape) -> Shape:
    """Move shape down by its own height."""
    r, c = s.position
    return s.move_to((r + s.n_rows, c))


@primitive("movement", cancels_with=("translate_right",))
def translate_left(s: Shape) -> Shape:
    """Move shape left by its own width."""
    r, c = s.position
    return s.move_to((r, c - s.n_cols))


@primitive("movement", cancels_with=("translate_left",))
def translate_right(s: Shape) -> Shape:
    """Move shape right by its own width."""
    r, c = s.position
    return s.move_to((r, c + s.n_cols))


# === Relational (3) ========================================================


@primitive("relational", involution=True)
def swap_colors(src: Shape, tgt: Shape) -> tuple[Shape, Shape]:
    """Swap primary colors between `src` and `tgt`.

    Pixels of `src` at its primary color take `tgt`'s primary color, and
    vice versa. Other colors are untouched. Raises `ValueError` if the
    two primary colors match (the op would be a no-op).
    """
    a, b = src.primary_color, tgt.primary_color
    if a == b:
        raise ValueError(f"swap_colors requires distinct primary colors (both are {a})")

    src_grid = src.grid.copy()
    tgt_grid = tgt.grid.copy()
    src_grid[src.grid == a] = b
    tgt_grid[tgt.grid == b] = a
    return Shape(src_grid, src.position), Shape(tgt_grid, tgt.position)


@primitive("relational", idempotent=True)
def align_row(src: Shape, tgt: Shape) -> tuple[Shape, Shape]:
    """Move `src` so its top-left row matches `tgt`'s. `tgt` unchanged.

    Raises `ValueError` if already aligned.
    """
    if src.position[0] == tgt.position[0]:
        raise ValueError(f"align_row requires different rows (both at {src.position[0]})")
    return src.move_to((tgt.position[0], src.position[1])), tgt


@primitive("relational", idempotent=True)
def align_col(src: Shape, tgt: Shape) -> tuple[Shape, Shape]:
    """Move `src` so its top-left col matches `tgt`'s. `tgt` unchanged.

    Raises `ValueError` if already aligned.
    """
    if src.position[1] == tgt.position[1]:
        raise ValueError(f"align_col requires different cols (both at {src.position[1]})")
    return src.move_to((src.position[0], tgt.position[1])), tgt


# === Metadata helpers ======================================================


def category_of(name: str) -> Category:
    return PRIMITIVES[name].category


def primitives_by_category(category: Category) -> list[str]:
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    return [n for n, p in PRIMITIVES.items() if p.category == category]


def canceling_pairs(primitives: list[str] | None = None) -> list[tuple[str, str]]:
    """Return all `(a, b)` such that running `a` then `b` is identity.

    If `primitives` is given, restrict to pairs where both names are in it.
    """
    universe = set(PRIMITIVES) if primitives is None else set(primitives)
    pairs: list[tuple[str, str]] = []
    for a, p in PRIMITIVES.items():
        if a not in universe:
            continue
        for b in p.cancels_with:
            if b in universe:
                pairs.append((a, b))
    return pairs


def involutions() -> list[str]:
    return [n for n, p in PRIMITIVES.items() if p.involution]


def idempotents() -> list[str]:
    return [n for n, p in PRIMITIVES.items() if p.idempotent]


def cyclic_orders() -> dict[str, int]:
    return {n: p.cyclic_order for n, p in PRIMITIVES.items() if p.cyclic_order is not None}


# === Registry self-check ==================================================


def validate_registry() -> None:
    """Check `PRIMITIVES` is internally consistent.

    Run at module import (below) and should be re-run after any runtime
    `@primitive` addition. Cheap — walks the registry once.
    """
    names = set(PRIMITIVES)

    # 1. cancels_with symmetry and target-existence
    for name, p in PRIMITIVES.items():
        for other in p.cancels_with:
            if other not in names:
                raise ValueError(
                    f"primitive {name!r} cancels_with {other!r}, which is not registered"
                )
            if name not in PRIMITIVES[other].cancels_with:
                raise ValueError(
                    f"cancels_with not symmetric: {name!r}↔{other!r}"
                )

    # 2. Category <-> arity consistency (defensive — decorator already checks)
    for name, p in PRIMITIVES.items():
        n_params = len(inspect.signature(p.fn).parameters)
        expected = 2 if p.category == "relational" else 1
        if n_params != expected:
            raise ValueError(
                f"primitive {name!r}: category {p.category!r} expects {expected} "
                f"params, got {n_params}"
            )

    # 3. Relational registry must match the hardcoded set in core.py
    relational_registered = {n for n, p in PRIMITIVES.items() if p.category == "relational"}
    if relational_registered != set(RELATIONAL_PRIMITIVE_NAMES):
        raise ValueError(
            f"relational primitive mismatch: registry={sorted(relational_registered)}, "
            f"core.RELATIONAL_PRIMITIVE_NAMES={sorted(RELATIONAL_PRIMITIVE_NAMES)}"
        )


validate_registry()


# === Apply a single primitive ==============================================


def apply_primitive(name: str, *shapes: Shape) -> Shape | tuple[Shape, Shape]:
    """Apply a primitive by name. Dispatches on arity.

    Use this for one-off invocations. The scene-level executor lives in
    `babytheorist.engine.apply_step` / `apply_program`.
    """
    p = PRIMITIVES[name]
    if p.category == "relational":
        if len(shapes) != 2:
            raise ValueError(f"{name!r} is relational; expected 2 shapes, got {len(shapes)}")
        return p.fn(*shapes)
    if len(shapes) != 1:
        raise ValueError(f"{name!r} is {p.category}; expected 1 shape, got {len(shapes)}")
    return p.fn(shapes[0])
