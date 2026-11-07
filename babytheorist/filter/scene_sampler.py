"""Constraint-aware scene sampler.

Given an :class:`InputConstraint`, draw shapes from a
:class:`ShapeFactory` and place them on a world grid while pruning
placements that violate:

- per-shape constraints (allowed stabilizers, forbidden primaries,
  required grid_shape);
- cross-shape constraints (row / col / primary coincidence forbids
  — all empty under the Phase 6 D4-only vocab, so they're wired
  here for correctness but aren't exercised until later phases).

Shape primary color and hollow detection match the runtime
:class:`babytheorist.core.Shape` semantics used by selectors and primitives.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.ndimage as ndi

from babytheorist.core import Shape
from babytheorist.shapes import PATTERN_TYPES, ShapeFactory, hollow
from babytheorist.symbolic import LinExpr

from .constraint import InputConstraint
from .d4_tables import D4_IDENTITY, D4_SUBGROUPS
from .shape_attrs import stabilizer_of_pixel_grid


__all__ = [
    "sample_scene",
    "primary_color_of",
    "grid_shape_of",
]


_ALL_STABILIZERS = frozenset(D4_SUBGROUPS)
_TRIVIAL_STABILIZER = frozenset({D4_IDENTITY})
_CONSTRUCTIVE_SHAPE_ATTEMPTS = 32
_SOLID_COLOR_MODES = frozenset({"solid", "bordered", "patterned"})


def primary_color_of(shape: Shape) -> int:
    """Runtime primary color used by the engine's ``by_color`` selector."""
    return shape.primary_color


def grid_shape_of(shape: Shape) -> str:
    """Runtime grid-shape predicate used by ``is_hollow`` and primitives."""
    return "hollow" if shape.is_hollow else "solid"


def _shape_satisfies(shape: Shape, shape_constraint) -> bool:
    """Per-shape constraint check (attributes only — placement-
    independent)."""
    if shape_constraint.require_non_square:
        if shape.grid.shape[0] == shape.grid.shape[1]:
            return False
    if shape_constraint.min_bbox_dim > 1:
        if min(shape.grid.shape) < shape_constraint.min_bbox_dim:
            return False

    primary: int | None = None
    if shape_constraint.required_primary_color is not None:
        primary = primary_color_of(shape)
        if primary != shape_constraint.required_primary_color:
            return False
    if shape_constraint.forbidden_primary_colors:
        if primary is None:
            primary = primary_color_of(shape)
        if primary in shape_constraint.forbidden_primary_colors:
            return False

    if shape_constraint.required_grid_shape is not None:
        if grid_shape_of(shape) != shape_constraint.required_grid_shape:
            return False

    if shape_constraint.allowed_stabilizers != _ALL_STABILIZERS:
        stab = stabilizer_of_pixel_grid(shape.grid)
        if stab not in shape_constraint.allowed_stabilizers:
            return False
    return True


def _factory_accepts_shape(factory: ShapeFactory, shape: Shape) -> bool:
    max_span = getattr(factory, "max_bbox_span", None)
    return max_span is None or max(shape.n_rows, shape.n_cols) <= int(max_span)


def _constructive_shape_satisfies(shape: Shape, shape_constraint) -> bool:
    if not _shape_satisfies(shape, shape_constraint):
        return False
    if (
        shape_constraint.allowed_stabilizers == _ALL_STABILIZERS
        or shape_constraint.min_bbox_dim <= 1
    ):
        return True

    # In hard vocabularies, add_border can erase a shape's visible
    # asymmetry. If the program already needs a restricted stabilizer, the
    # one-step bordered image must satisfy the same restriction; otherwise a
    # prior flip/rotation can become droppable after add_border.
    from babytheorist.primitives import add_border

    bordered = add_border(shape)
    return stabilizer_of_pixel_grid(bordered.grid) in shape_constraint.allowed_stabilizers


def _choose_size(
    size_range: tuple[int, int], rng: np.random.Generator,
) -> int:
    lo, hi = size_range
    return lo if hi == lo else int(rng.integers(lo, hi))


def _generators_for_constraint(factory: ShapeFactory, shape_constraint) -> tuple:
    """Narrow generator choices using hard grid-shape requirements."""
    if shape_constraint.required_grid_shape == "hollow":
        return (hollow,)
    if shape_constraint.required_grid_shape == "solid":
        solid_gens = tuple(
            g for g in factory.generators if getattr(g, "__name__", "") != "hollow"
        )
        return solid_gens or factory.generators
    return factory.generators


def _recolor_primary(
    shape: Shape,
    shape_constraint,
    palette: Sequence[int],
    rng: np.random.Generator,
) -> Shape:
    """Adjust primary color constraints directly, then let checks verify."""
    required = shape_constraint.required_primary_color
    forbidden = shape_constraint.forbidden_primary_colors
    if required is None and not forbidden:
        return shape

    primary = shape.primary_color
    if primary == 0:
        return shape

    target: int | None = None
    if required is not None:
        if required == primary:
            return shape
        target = int(required)
    elif primary in forbidden:
        candidates = [int(c) for c in palette if int(c) not in forbidden]
        if not candidates:
            return shape
        target = candidates[int(rng.integers(0, len(candidates)))]
        if target == primary:
            return shape
    else:
        return shape

    grid = shape.grid.copy()
    if target in shape.colors:
        grid[shape.grid == target] = primary
    grid[shape.grid == primary] = target
    return Shape(grid, shape.position)


def _primary_color_for_constraint(
    shape_constraint,
    palette: Sequence[int],
    rng: np.random.Generator,
) -> int | None:
    if shape_constraint.required_primary_color is not None:
        color = int(shape_constraint.required_primary_color)
        if color in shape_constraint.forbidden_primary_colors:
            return None
        if color not in palette:
            return None
        return color

    candidates = [
        int(c) for c in palette
        if int(c) not in shape_constraint.forbidden_primary_colors
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _secondary_marker_color(
    primary: int,
    palette: Sequence[int],
    rng: np.random.Generator,
) -> int | None:
    candidates = [int(c) for c in palette if int(c) != primary]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _generator_name(gen) -> str:
    return str(getattr(gen, "__name__", ""))


def _solid_color_mode_for_constraint(
    factory: ShapeFactory,
    palette: Sequence[int],
    rng: np.random.Generator,
) -> str:
    probs = getattr(factory, "_probs", None)
    if probs is None or len(probs) != len(factory.generators):
        probs = (1.0 / len(factory.generators),) * len(factory.generators)

    options: list[tuple[str, float]] = []
    for gen, prob in zip(factory.generators, probs):
        mode = _generator_name(gen)
        if mode not in _SOLID_COLOR_MODES:
            continue
        if mode in {"bordered", "patterned"} and len(palette) < 2:
            continue
        options.append((mode, float(prob)))

    if not options:
        return "solid"
    weights = np.asarray([w for _mode, w in options], dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        weights = np.ones(len(options), dtype=np.float64)
    weights /= weights.sum()
    return options[int(rng.choice(len(options), p=weights))][0]


def _construct_dims(
    shape_constraint,
    *,
    hollow_shape: bool,
    rng: np.random.Generator,
    size_range: tuple[int, int],
) -> tuple[int, int]:
    min_dim = max(int(shape_constraint.min_bbox_dim), 3 if hollow_shape else 2)
    lo, hi = size_range
    target = lo if hi == lo else int(rng.integers(lo, hi))
    if hollow_shape:
        side_budget = max(target // 4 + min_dim, min_dim + 1)
        h = min_dim + int(rng.integers(0, max(2, side_budget - min_dim + 1)))
        w = side_budget + int(rng.integers(0, 3))
    else:
        target = max(target, 2 * min_dim - 1)
        center = max(min_dim, target // 2)
        jitter = int(rng.integers(-2, 3))
        h = max(min_dim, center + jitter)
        w = max(min_dim, target - h + 1)
    if h == w:
        w += 1
    # D4 + translate shortcuts can appear at small integer coincidences such
    # as w = 2h (translate before/after a rot90 becomes equivalent). These are
    # dimension preconditions, so avoid them constructively.
    if h == 2 * w or w == 2 * h:
        w += 1
    if h == w:
        w += 1
    return h, w


def _random_manhattan_path(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Random shortest path between two cells inside a fixed bbox."""
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r < tr:
            moves.append((r + 1, c))
        elif r > tr:
            moves.append((r - 1, c))
        if c < tc:
            moves.append((r, c + 1))
        elif c > tc:
            moves.append((r, c - 1))
        r, c = moves[int(rng.integers(0, len(moves)))]
        path.append((r, c))
    return path


def _connected_spanning_pixels(
    h: int,
    w: int,
    *,
    rng: np.random.Generator,
    size_range: tuple[int, int],
) -> set[tuple[int, int]]:
    """Draw a connected random silhouette that touches all bbox sides.

    This conditions generation on the geometry constraints directly:
    touching every side makes the sampled bbox exactly ``h x w``. Extra
    pixels are grown from a random frontier instead of using a fixed
    silhouette template.
    """
    anchors = [
        (0, int(rng.integers(0, w))),
        (h - 1, int(rng.integers(0, w))),
        (int(rng.integers(0, h)), 0),
        (int(rng.integers(0, h)), w - 1),
    ]
    rng.shuffle(anchors)

    pixels: set[tuple[int, int]] = set()
    current = anchors[0]
    pixels.add(current)
    for target in anchors[1:]:
        pixels.update(_random_manhattan_path(current, target, rng=rng))
        current = target

    lo, hi = size_range
    size_budget = lo if hi == lo else int(rng.integers(lo, hi))
    min_cells = len(pixels)
    max_cells = h * w
    extra_budget = max(size_budget, min_cells + max(1, min(h, w) // 2))
    target_cells = min(max_cells, extra_budget + int(rng.integers(0, max(2, min(h, w)))))

    while len(pixels) < target_cells:
        frontier: set[tuple[int, int]] = set()
        for r, c in pixels:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in pixels:
                    frontier.add((nr, nc))
        if not frontier:
            break
        candidates = sorted(frontier)
        pixels.add(candidates[int(rng.integers(0, len(candidates)))])

    return pixels


def _pixels_to_solid_grid(
    pixels: set[tuple[int, int]],
    *,
    h: int,
    w: int,
    primary: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=np.int64)
    for r, c in pixels:
        grid[r, c] = primary

    mask = grid != 0
    filled = ndi.binary_fill_holes(mask)
    grid[filled & ~mask] = primary
    return grid


def _ensure_primary_majority(
    grid: np.ndarray,
    *,
    primary: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = grid.copy()
    colors = [int(c) for c in np.unique(out) if int(c) > 0]
    if len(colors) < 2:
        return out

    while Shape(out).primary_color != primary:
        current = Shape(out).primary_color
        coords = np.argwhere(out == current)
        if len(coords) == 0:
            break
        r, c = coords[int(rng.integers(0, len(coords)))]
        out[r, c] = primary
    return out


def _apply_bordered_coloring(
    base_grid: np.ndarray,
    *,
    primary: int,
    secondary: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if secondary is None:
        return base_grid
    mask = base_grid != 0
    if not np.any(mask):
        return base_grid

    boundary = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not mask[nr, nc]:
                    boundary[r, c] = True
                    break

    interior = mask & ~boundary
    if not np.any(interior):
        return base_grid

    boundary_count = int(np.count_nonzero(boundary))
    interior_count = int(np.count_nonzero(interior))
    boundary_color = primary if boundary_count >= interior_count else secondary
    interior_color = secondary if boundary_color == primary else primary

    out = np.zeros_like(base_grid)
    out[boundary] = boundary_color
    out[interior] = interior_color
    return _ensure_primary_majority(out, primary=primary, rng=rng)


def _apply_patterned_coloring(
    base_grid: np.ndarray,
    *,
    primary: int,
    secondary: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if secondary is None:
        return base_grid
    mask = base_grid != 0
    if not np.any(mask):
        return base_grid

    pattern = str(rng.choice(PATTERN_TYPES))
    rows, cols = mask.shape
    group = np.zeros(mask.shape, dtype=bool)
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c]:
                continue
            if pattern == "striped_h":
                group[r, c] = (r % 2 == 0)
            elif pattern == "striped_v":
                group[r, c] = (c % 2 == 0)
            elif pattern == "split_h":
                group[r, c] = (r < rows / 2)
            elif pattern == "split_v":
                group[r, c] = (c < cols / 2)
            else:
                group[r, c] = ((r + c) % 2 == 0)

    a = mask & group
    b = mask & ~group
    primary_group = a if np.count_nonzero(a) >= np.count_nonzero(b) else b
    secondary_group = mask & ~primary_group

    out = np.zeros_like(base_grid)
    out[primary_group] = primary
    out[secondary_group] = secondary
    return _ensure_primary_majority(out, primary=primary, rng=rng)


def _apply_solid_coloring_mode(
    base_grid: np.ndarray,
    *,
    mode: str,
    primary: int,
    secondary: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "bordered":
        return _apply_bordered_coloring(
            base_grid, primary=primary, secondary=secondary, rng=rng,
        )
    if mode == "patterned":
        return _apply_patterned_coloring(
            base_grid, primary=primary, secondary=secondary, rng=rng,
        )
    return base_grid


def _pixels_to_hollow_grid(
    inner_pixels: set[tuple[int, int]],
    *,
    h: int,
    w: int,
    primary: int,
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=np.int64)
    shifted_inner = {(r + 1, c + 1) for r, c in inner_pixels}
    for r, c in shifted_inner:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in shifted_inner:
                grid[nr, nc] = primary
    return grid


def _add_asymmetric_marker(
    grid: np.ndarray,
    *,
    primary: int,
    marker: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if marker is None:
        return grid
    coords = np.argwhere(grid == primary)
    if len(coords) == 0:
        return grid
    for idx in rng.permutation(len(coords)):
        candidate = grid.copy()
        r, c = coords[int(idx)]
        candidate[r, c] = marker
        if stabilizer_of_pixel_grid(candidate) == _TRIVIAL_STABILIZER:
            return candidate
    r, c = coords[int(rng.integers(0, len(coords)))]
    out = grid.copy()
    out[r, c] = marker
    return out


def _maybe_add_stabilizer_marker(
    grid: np.ndarray,
    shape_constraint,
    *,
    primary: int,
    marker: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if shape_constraint.allowed_stabilizers == _ALL_STABILIZERS:
        return grid
    if stabilizer_of_pixel_grid(grid) in shape_constraint.allowed_stabilizers:
        return grid
    return _add_asymmetric_marker(grid, primary=primary, marker=marker, rng=rng)


def _construct_solid_shape(
    shape_constraint,
    primary: int,
    marker: int | None,
    rng: np.random.Generator,
    size_range: tuple[int, int],
    color_mode: str,
) -> Shape | None:
    fallback_grid: np.ndarray | None = None
    for _ in range(_CONSTRUCTIVE_SHAPE_ATTEMPTS):
        h, w = _construct_dims(
            shape_constraint, hollow_shape=False, rng=rng, size_range=size_range,
        )
        pixels = _connected_spanning_pixels(h, w, rng=rng, size_range=size_range)
        base_grid = _pixels_to_solid_grid(pixels, h=h, w=w, primary=primary)
        base_shape = Shape(base_grid)
        if not _constructive_shape_satisfies(base_shape, shape_constraint):
            fallback_grid = base_grid
            continue
        grid = _apply_solid_coloring_mode(
            base_grid,
            mode=color_mode,
            primary=primary,
            secondary=marker,
            rng=rng,
        )
        shape = Shape(grid)
        if _constructive_shape_satisfies(shape, shape_constraint):
            return shape
        fallback_grid = base_grid

    if fallback_grid is None:
        return None
    grid = _maybe_add_stabilizer_marker(
        fallback_grid, shape_constraint, primary=primary, marker=marker, rng=rng,
    )
    return Shape(grid)


def _construct_hollow_shape(
    shape_constraint,
    primary: int,
    marker: int | None,
    rng: np.random.Generator,
    size_range: tuple[int, int],
) -> Shape | None:
    fallback_grid: np.ndarray | None = None
    for _ in range(_CONSTRUCTIVE_SHAPE_ATTEMPTS):
        h, w = _construct_dims(
            shape_constraint, hollow_shape=True, rng=rng, size_range=size_range,
        )
        inner_pixels = _connected_spanning_pixels(
            h - 2, w - 2, rng=rng, size_range=size_range,
        )
        grid = _pixels_to_hollow_grid(inner_pixels, h=h, w=w, primary=primary)
        shape = Shape(grid)
        if _constructive_shape_satisfies(shape, shape_constraint):
            return shape
        fallback_grid = grid

    if fallback_grid is None:
        return None
    grid = _maybe_add_stabilizer_marker(
        fallback_grid, shape_constraint, primary=primary, marker=marker, rng=rng,
    )
    return Shape(grid)


def _construct_shape_for_constraint(
    shape_constraint,
    *,
    factory: ShapeFactory,
    palette: tuple[int, ...],
    rng: np.random.Generator,
) -> Shape | None:
    """Directly build a shape for hard preconditions when possible.

    The fallback below is still needed for unusual constraints, but hard-mode
    generation often asks for a non-square, min-size, trivial-stabilizer solid
    shape. Building that shape directly avoids thousands of random-walk
    rejection checks per accepted scene.
    """
    if not getattr(factory, "allow_constructive_shapes", False):
        return None
    needs_geometry = (
        shape_constraint.require_non_square
        or shape_constraint.min_bbox_dim > 1
        or shape_constraint.allowed_stabilizers != _ALL_STABILIZERS
    )
    if not needs_geometry:
        return None
    if shape_constraint.allowed_stabilizers != _ALL_STABILIZERS:
        if _TRIVIAL_STABILIZER not in shape_constraint.allowed_stabilizers:
            return None

    primary = _primary_color_for_constraint(shape_constraint, palette, rng)
    if primary is None:
        return None

    if shape_constraint.required_grid_shape == "hollow":
        shape = _construct_hollow_shape(
            shape_constraint,
            primary,
            _secondary_marker_color(primary, palette, rng),
            rng,
            factory.size_range,
        )
    elif shape_constraint.required_grid_shape in (None, "solid"):
        shape = _construct_solid_shape(
            shape_constraint,
            primary,
            _secondary_marker_color(primary, palette, rng),
            rng,
            factory.size_range,
            _solid_color_mode_for_constraint(factory, palette, rng),
        )
    else:
        return None

    if shape is None:
        return None
    if not _factory_accepts_shape(factory, shape):
        return None
    return shape if _constructive_shape_satisfies(shape, shape_constraint) else None


def _sample_shape_for_constraint(
    shape_constraint,
    *,
    factory: ShapeFactory,
    palette: tuple[int, ...],
    rng: np.random.Generator,
    max_attempts: int,
    placement_predicate=None,
) -> Shape | None:
    constructed = _construct_shape_for_constraint(
        shape_constraint, factory=factory, palette=palette, rng=rng,
    )
    if constructed is not None:
        if placement_predicate is None or placement_predicate(constructed):
            return constructed

    generators = _generators_for_constraint(factory, shape_constraint)
    def predicate(shape: Shape) -> bool:
        if not _shape_satisfies(shape, shape_constraint):
            return False
        return placement_predicate is None or placement_predicate(shape)

    try:
        return factory.sample_from(
            generators,
            rng,
            palette,
            max_attempts=max_attempts,
            transform=lambda shape: _recolor_primary(
                shape, shape_constraint, palette, rng,
            ),
            predicate=predicate,
        )
    except RuntimeError:
        return None


def _eval_linexpr(expr: LinExpr, shapes: list[Shape]) -> int | None:
    values: dict[str, int] = {}
    for i, shape in enumerate(shapes):
        values[f"r_{i}"] = shape.position[0]
        values[f"c_{i}"] = shape.position[1]
        values[f"h_{i}"] = shape.n_rows
        values[f"w_{i}"] = shape.n_cols
    total = expr.const
    for var, coeff in expr.coeffs:
        if var not in values:
            return None
        total += coeff * values[var]
    return total


def _eval_linexpr_partial(expr: LinExpr, shapes: list[Shape | None]) -> int | None:
    values: dict[str, int] = {}
    for i, shape in enumerate(shapes):
        if shape is None:
            continue
        values[f"r_{i}"] = shape.position[0]
        values[f"c_{i}"] = shape.position[1]
        values[f"h_{i}"] = shape.n_rows
        values[f"w_{i}"] = shape.n_cols
    total = expr.const
    for var, coeff in expr.coeffs:
        if var not in values:
            return None
        total += coeff * values[var]
    return total


def _linexpr_shape_indices(expr: LinExpr) -> frozenset[int]:
    indices: set[int] = set()
    for var, _coeff in expr.coeffs:
        try:
            indices.add(int(var.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return frozenset(indices)


def _box_bound_deps(bound) -> frozenset[int]:
    deps = {bound.shape_idx}
    for expr in (bound.row, bound.col, bound.height, bound.width):
        deps.update(_linexpr_shape_indices(expr))
    return frozenset(deps)


def _box_bound_satisfies(bound, shapes: list[Shape], world_size: int) -> bool:
    row = _eval_linexpr(bound.row, shapes)
    col = _eval_linexpr(bound.col, shapes)
    height = _eval_linexpr(bound.height, shapes)
    width = _eval_linexpr(bound.width, shapes)
    if None in (row, col, height, width):
        return False
    return 0 <= row and 0 <= col and row + height <= world_size \
        and col + width <= world_size


def _box_bound_satisfies_partial(
    bound, positioned: list[Shape | None], world_size: int,
) -> bool | None:
    deps = _box_bound_deps(bound)
    if not all(0 <= i < len(positioned) and positioned[i] is not None for i in deps):
        return None
    row = _eval_linexpr_partial(bound.row, positioned)
    col = _eval_linexpr_partial(bound.col, positioned)
    height = _eval_linexpr_partial(bound.height, positioned)
    width = _eval_linexpr_partial(bound.width, positioned)
    if None in (row, col, height, width):
        return False
    return 0 <= row and 0 <= col and row + height <= world_size \
        and col + width <= world_size


def _cross_shape_satisfies(
    shapes: list[Shape], constraint: InputConstraint, *, world_size: int,
) -> bool:
    """Cross-shape forbids evaluated on the placed scene."""
    if constraint.row_forbids:
        for f in constraint.row_forbids:
            r_i = shapes[f.shape_i].position[0]
            r_j = shapes[f.shape_j].position[0]
            if r_i + f.offset == r_j:
                return False
    if constraint.col_forbids:
        for f in constraint.col_forbids:
            c_i = shapes[f.shape_i].position[1]
            c_j = shapes[f.shape_j].position[1]
            if c_i + f.offset == c_j:
                return False
    if constraint.primary_forbids:
        primaries = [primary_color_of(s) for s in shapes]
        for f in constraint.primary_forbids:
            if (primaries[f.shape_i] + f.offset) % 9 == primaries[f.shape_j]:
                return False
    if constraint.linear_forbids:
        for f in constraint.linear_forbids:
            left = _eval_linexpr(f.left, shapes)
            right = _eval_linexpr(f.right, shapes)
            if left is not None and right is not None and left == right:
                return False
    if constraint.box_bounds:
        for bound in constraint.box_bounds:
            if not _box_bound_satisfies(bound, shapes, world_size):
                return False
    if constraint.positional_orderings:
        for o in constraint.positional_orderings:
            axis_idx = 0 if o.axis == "row" else 1
            lt_val = shapes[o.lt].position[axis_idx]
            gt_val = shapes[o.gt].position[axis_idx]
            if not (lt_val < gt_val):
                return False
    if constraint.raster_orderings:
        for o in constraint.raster_orderings:
            if not (shapes[o.before].position < shapes[o.after].position):
                return False
    return True


def _partial_cross_shape_satisfies(
    positioned: list[Shape | None], constraint: InputConstraint, *, world_size: int,
) -> bool:
    """Check only cross-shape constraints whose inputs are already placed."""
    if constraint.row_forbids:
        for f in constraint.row_forbids:
            a = positioned[f.shape_i]
            b = positioned[f.shape_j]
            if a is not None and b is not None:
                if a.position[0] + f.offset == b.position[0]:
                    return False
    if constraint.col_forbids:
        for f in constraint.col_forbids:
            a = positioned[f.shape_i]
            b = positioned[f.shape_j]
            if a is not None and b is not None:
                if a.position[1] + f.offset == b.position[1]:
                    return False
    if constraint.primary_forbids:
        for f in constraint.primary_forbids:
            a = positioned[f.shape_i]
            b = positioned[f.shape_j]
            if a is not None and b is not None:
                if (primary_color_of(a) + f.offset) % 9 == primary_color_of(b):
                    return False
    if constraint.linear_forbids:
        for f in constraint.linear_forbids:
            deps = _linexpr_shape_indices(f.left) | _linexpr_shape_indices(f.right)
            if all(0 <= i < len(positioned) and positioned[i] is not None
                   for i in deps):
                left = _eval_linexpr_partial(f.left, positioned)
                right = _eval_linexpr_partial(f.right, positioned)
                if left is not None and right is not None and left == right:
                    return False
    if constraint.box_bounds:
        for bound in constraint.box_bounds:
            ok = _box_bound_satisfies_partial(bound, positioned, world_size)
            if ok is False:
                return False
    if constraint.positional_orderings:
        for o in constraint.positional_orderings:
            a = positioned[o.lt]
            b = positioned[o.gt]
            if a is not None and b is not None:
                axis_idx = 0 if o.axis == "row" else 1
                if not (a.position[axis_idx] < b.position[axis_idx]):
                    return False
    if constraint.raster_orderings:
        for o in constraint.raster_orderings:
            a = positioned[o.before]
            b = positioned[o.after]
            if a is not None and b is not None:
                if not (a.position < b.position):
                    return False
    return True


def _candidate_positions(
    shape: Shape, world_size: int, gap: int,
) -> list[tuple[int, int]]:
    r_max = world_size - shape.n_rows - gap + 1
    c_max = world_size - shape.n_cols - gap + 1
    if r_max <= gap or c_max <= gap:
        return []
    return [(r, c) for r in range(gap, r_max) for c in range(gap, c_max)]


def _position_satisfies_self_box_bounds(
    shape: Shape,
    shape_idx: int,
    position: tuple[int, int],
    constraint: InputConstraint,
    world_size: int,
) -> bool:
    positioned: list[Shape | None] = [None] * constraint.n_shapes
    positioned[shape_idx] = shape.move_to(position)
    for bound in constraint.box_bounds:
        deps = _box_bound_deps(bound)
        if deps - {shape_idx}:
            continue
        ok = _box_bound_satisfies_partial(bound, positioned, world_size)
        if ok is False:
            return False
    return True


def _shape_has_feasible_self_box_position(
    shape: Shape,
    shape_idx: int,
    constraint: InputConstraint,
    *,
    world_size: int,
    gap: int,
) -> bool:
    if not constraint.box_bounds:
        return True
    if not any(_box_bound_deps(bound) <= {shape_idx}
               for bound in constraint.box_bounds):
        return True
    return any(
        _position_satisfies_self_box_bounds(
            shape, shape_idx, pos, constraint, world_size,
        )
        for pos in _candidate_positions(shape, world_size, gap)
    )


def _fits_occupied(
    shape: Shape,
    position: tuple[int, int],
    occupied: np.ndarray,
    world_size: int,
) -> bool:
    r, c = position
    for i in range(shape.n_rows):
        for j in range(shape.n_cols):
            if shape.grid[i, j] == 0:
                continue
            wr, wc = r + i, c + j
            if wr >= world_size or wc >= world_size or occupied[wr, wc]:
                return False
    return True


def _mark_occupied(shape: Shape, occupied: np.ndarray, gap: int) -> list[tuple[int, int]]:
    changed: list[tuple[int, int]] = []
    r, c = shape.position
    world_size = occupied.shape[0]
    for i in range(shape.n_rows):
        for j in range(shape.n_cols):
            if shape.grid[i, j] == 0:
                continue
            for di in range(-gap, gap + 1):
                for dj in range(-gap, gap + 1):
                    wr, wc = r + i + di, c + j + dj
                    if 0 <= wr < world_size and 0 <= wc < world_size:
                        if not occupied[wr, wc]:
                            occupied[wr, wc] = True
                            changed.append((wr, wc))
    return changed


def _unmark_occupied(occupied: np.ndarray, changed: list[tuple[int, int]]) -> None:
    for wr, wc in reversed(changed):
        occupied[wr, wc] = False


def _render_world(shapes: list[Shape], world_size: int) -> np.ndarray:
    world = np.zeros((world_size, world_size), dtype=np.int64)
    for shape in shapes:
        shape.place_on(world)
    return world


def _place_shapes_for_constraint(
    shapes: list[Shape],
    constraint: InputConstraint,
    *,
    world_size: int,
    gap: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> tuple[np.ndarray, list[Shape]] | None:
    """Place shapes while pruning with already-known cross constraints."""
    positioned: list[Shape | None] = [None] * len(shapes)
    occupied = np.zeros((world_size, world_size), dtype=bool)
    attempts = 0

    order = list(range(len(shapes)))
    order.sort(
        key=lambda i: sum(
            1
            for o in constraint.positional_orderings
            if o.lt == i or o.gt == i
        ) + sum(
            1
            for o in constraint.raster_orderings
            if o.before == i or o.after == i
        ),
        reverse=True,
    )

    candidates_by_idx: dict[int, list[tuple[int, int]]] = {}
    for idx in order:
        candidates = [
            pos
            for pos in _candidate_positions(shapes[idx], world_size, gap)
            if _position_satisfies_self_box_bounds(
                shapes[idx], idx, pos, constraint, world_size,
            )
        ]
        if not candidates:
            return None
        rng.shuffle(candidates)
        candidates_by_idx[idx] = candidates

    def dfs(depth: int) -> bool:
        nonlocal attempts
        if attempts >= max_attempts:
            return False
        if depth == len(order):
            return True
        idx = order[depth]
        shape = shapes[idx]
        for pos in candidates_by_idx[idx]:
            attempts += 1
            if attempts > max_attempts:
                return False
            if not _fits_occupied(shape, pos, occupied, world_size):
                continue
            placed = shape.move_to(pos)
            positioned[idx] = placed
            if not _partial_cross_shape_satisfies(
                positioned, constraint, world_size=world_size,
            ):
                positioned[idx] = None
                continue
            changed = _mark_occupied(placed, occupied, gap)
            if dfs(depth + 1):
                return True
            _unmark_occupied(occupied, changed)
            positioned[idx] = None
        return False

    if not dfs(0):
        return None
    placed_shapes = [s for s in positioned if s is not None]
    if len(placed_shapes) != len(shapes):
        return None
    if not _cross_shape_satisfies(
        placed_shapes, constraint, world_size=world_size,
    ):
        return None
    return _render_world(placed_shapes, world_size), placed_shapes


def sample_scene(
    constraint: InputConstraint,
    *,
    factory: ShapeFactory,
    palette: tuple[int, ...],
    world_size: int,
    rng: np.random.Generator,
    gap: int = 2,
    max_total_retries: int = 300,
    max_place_attempts: int = 300,
) -> Optional[tuple[np.ndarray, list[Shape]]]:
    """Sample a scene satisfying ``constraint``.

    Returns ``(world_grid, positioned_shapes)`` or ``None`` if the
    retry budget is exhausted — caller should then resample the
    program.

    The inner loop draws one shape at a time using the hard per-shape
    constraints where possible. Once all shapes pass the attribute checks,
    placement uses a small constraint-aware backtracking search so
    row/col/linear/positional constraints prune bad positions immediately.
    """
    if constraint.is_infeasible():
        return None

    n = constraint.n_shapes
    for _ in range(max_total_retries):
        # 1. Draw shapes that individually pass their per-shape constraint.
        shapes: list[Shape] = []
        for i in range(n):
            s = _sample_shape_for_constraint(
                constraint.shapes[i],
                factory=factory,
                palette=palette,
                rng=rng,
                max_attempts=max_total_retries,
                placement_predicate=lambda shape, i=i: (
                    _shape_has_feasible_self_box_position(
                        shape, i, constraint, world_size=world_size, gap=gap,
                    )
                ),
            )
            if s is None:
                break  # per-shape retry exhausted — restart outer loop
            shapes.append(s)
        if len(shapes) != n:
            continue

        # 2. Place while pruning with cross-shape constraints.
        result = _place_shapes_for_constraint(
            shapes, constraint, world_size=world_size, gap=gap, rng=rng,
            max_attempts=max_place_attempts,
        )
        if result is None:
            continue
        world, positioned = result

        # 3. Cross-shape constraints on placed scene.
        if not _cross_shape_satisfies(
            positioned, constraint, world_size=world_size,
        ):
            continue

        return world, positioned

    return None
