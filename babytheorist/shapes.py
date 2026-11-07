"""Procedural shape generation and placement.

A shape *generator* is a free function ``(rng, size, palette) -> Shape`` that
samples a bounding-box Shape at position (0, 0). Generators are stateless
and composable — adding a new one means defining a function and passing it
to `ShapeFactory`:

    from babytheorist.shapes import ShapeFactory, DEFAULT_GENERATORS, solid

    def striped_triangle(rng, size, palette): ...

    factory = ShapeFactory(generators=DEFAULT_GENERATORS + (striped_triangle,))

The four built-in generators cover the coloring modes used by BabyTheorist tasks:

    solid     — single-color random walk
    bordered  — inner walk + 1-pixel wrap (2 colors)
    hollow    — only the wrap, inner discarded (1 color)
    patterned — random walk + 2-color pattern
                (striped_h/v, split_h/v, checkerboard)

`place_shapes` places a list of shapes onto a world grid with a gap
constraint (against walls and between shapes). It is *all-or-nothing*:
if any shape cannot be placed after `max_attempts` tries, the whole call
returns `None` so callers can retry with a fresh shape set.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .core import Shape


PATTERN_TYPES: tuple[str, ...] = (
    "striped_h", "striped_v", "split_h", "split_v", "checkerboard",
)
Generator = Callable[[np.random.Generator, int, Sequence[int]], Shape]
_DEFAULT_COMPACTNESS = 8.0


# === Silhouette: biased random walk =======================================


def random_walk(
    target_size: int,
    rng: np.random.Generator,
    compactness: float = 8.0,
) -> set[tuple[int, int]]:
    """Generate a 4-connected pixel set of size `target_size` via biased walk.

    Starts at (0, 0) and grows by adding 4-adjacent pixels. Candidates with
    more filled neighbors are preferred; higher `compactness` → rounder shapes.
    Returned coordinates are origin-relative (may include negatives).
    """
    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")

    filled = {(0, 0)}
    while len(filled) < target_size:
        candidate_map: dict[tuple[int, int], int] = {}
        for r, c in filled:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb not in filled:
                    candidate_map[nb] = candidate_map.get(nb, 0) + 1

        candidates = list(candidate_map.keys())
        counts = np.array([candidate_map[c] for c in candidates], dtype=float)
        weights = counts ** compactness
        weights /= weights.sum()
        idx = rng.choice(len(candidates), p=weights)
        filled.add(candidates[idx])
    return filled


def _pixels_to_grid(pixels: set[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _ in pixels]
    cols = [c for _, c in pixels]
    min_r, min_c = min(rows), min(cols)
    h = max(rows) - min_r + 1
    w = max(cols) - min_c + 1
    grid = np.zeros((h, w), dtype=np.int64)
    for r, c in pixels:
        grid[r - min_r, c - min_c] = color
    return grid


def _dual_pixels_to_grid(
    pixels_a: set[tuple[int, int]], color_a: int,
    pixels_b: set[tuple[int, int]], color_b: int,
) -> np.ndarray:
    all_pixels = pixels_a | pixels_b
    rows = [r for r, _ in all_pixels]
    cols = [c for _, c in all_pixels]
    min_r, min_c = min(rows), min(cols)
    h = max(rows) - min_r + 1
    w = max(cols) - min_c + 1
    grid = np.zeros((h, w), dtype=np.int64)
    for r, c in pixels_a:
        grid[r - min_r, c - min_c] = color_a
    for r, c in pixels_b:
        grid[r - min_r, c - min_c] = color_b
    return grid


def _wrap_border(inner: set[tuple[int, int]]) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for r, c in inner:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if nb not in inner:
                border.add(nb)
    return border


def _colorize_patterned(
    grid: np.ndarray, color_a: int, color_b: int, pattern: str,
) -> np.ndarray:
    rows, cols = grid.shape
    result = grid.copy()
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                continue
            if pattern == "striped_h":
                result[i, j] = color_a if i % 2 == 0 else color_b
            elif pattern == "striped_v":
                result[i, j] = color_a if j % 2 == 0 else color_b
            elif pattern == "split_h":
                result[i, j] = color_a if i < rows / 2 else color_b
            elif pattern == "split_v":
                result[i, j] = color_a if j < cols / 2 else color_b
            elif pattern == "checkerboard":
                result[i, j] = color_a if (i + j) % 2 == 0 else color_b
            else:
                raise ValueError(f"unknown pattern {pattern!r}")
    return result


def _pick_colors(rng: np.random.Generator, palette: Sequence[int], k: int) -> list[int]:
    if len(palette) < k:
        raise ValueError(f"palette needs >= {k} colors, got {len(palette)}")
    if len(palette) == k:
        return [int(c) for c in palette]
    idx = rng.choice(len(palette), size=k, replace=False)
    return [int(palette[i]) for i in idx]


def _border_inner_size(
    size: int,
    rng: np.random.Generator,
    *,
    random_inner_size: bool,
) -> int:
    if random_inner_size:
        # Match the v2 generator introduced in 9791e3c: bordered/hollow
        # objects receive a large target budget but sample a smaller random
        # inner walk, yielding compact outlines instead of chunky blobs.
        return int(rng.integers(1, max(3, size // 2 + 1)))
    return max(1, size // 2)


# === Coloring-mode generators =============================================


def solid(
    rng: np.random.Generator, size: int, palette: Sequence[int],
    *, compactness: float = _DEFAULT_COMPACTNESS,
) -> Shape:
    """Single-color random walk."""
    [color] = _pick_colors(rng, palette, 1)
    pixels = random_walk(size, rng, compactness=compactness)
    return Shape(_pixels_to_grid(pixels, color))


def bordered(
    rng: np.random.Generator, size: int, palette: Sequence[int],
    *, compactness: float = _DEFAULT_COMPACTNESS,
    random_inner_size: bool = False,
) -> Shape:
    """Inner walk wrapped in a 1-pixel border of a second color.

    The default inner silhouette is a walk of `max(1, size // 2)` pixels.
    `random_inner_size=True` restores the v2 sampler from commit 9791e3c.
    """
    border_color, fill_color = _pick_colors(rng, palette, 2)
    inner = random_walk(
        _border_inner_size(size, rng, random_inner_size=random_inner_size),
        rng,
        compactness=compactness,
    )
    border = _wrap_border(inner)
    return Shape(_dual_pixels_to_grid(inner, fill_color, border, border_color))


def hollow(
    rng: np.random.Generator, size: int, palette: Sequence[int],
    *, compactness: float = _DEFAULT_COMPACTNESS,
    random_inner_size: bool = False,
) -> Shape:
    """Only the 1-pixel wrap of an inner walk — interior is empty."""
    [color] = _pick_colors(rng, palette, 1)
    inner = random_walk(
        _border_inner_size(size, rng, random_inner_size=random_inner_size),
        rng,
        compactness=compactness,
    )
    border = _wrap_border(inner)
    return Shape(_pixels_to_grid(border, color))


def patterned(
    rng: np.random.Generator, size: int, palette: Sequence[int],
    pattern: str | None = None, *, compactness: float = _DEFAULT_COMPACTNESS,
) -> Shape:
    """Random walk filled with a 2-color pattern."""
    color_a, color_b = _pick_colors(rng, palette, 2)
    pat = pattern if pattern is not None else str(rng.choice(PATTERN_TYPES))
    pixels = random_walk(size, rng, compactness=compactness)
    base = _pixels_to_grid(pixels, 1)
    return Shape(_colorize_patterned(base, color_a, color_b, pat))


DEFAULT_GENERATORS: tuple[Generator, ...] = (solid, bordered, hollow, patterned)
_COMPACTNESS_GENERATORS = frozenset({solid, bordered, hollow, patterned})
_BORDERED_HOLLOW_GENERATORS = frozenset({bordered, hollow})


# === ShapeFactory =========================================================


class ShapeFactory:
    """Samples shapes by picking a generator and a size per call.

    Args:
        generators: tuple of `(rng, size, palette) -> Shape` functions.
        weights:    per-generator sampling weights; defaults to uniform.
                    Must match `len(generators)` if given.
        size_range: `(lo, hi)` inclusive-exclusive bounds for the pixel budget
                    passed to each generator.
        compactness: Random-walk gravity factor. Higher values prefer candidate
                     pixels with more already-filled neighbors.
        bordered_hollow_min_size: Optional minimum target size before sampling
                                  bordered/hollow shapes. `9791e3c` used 9.
        bordered_hollow_random_inner_size: If true, bordered/hollow generators
                                           sample the inner walk size using the
                                           v2 rule from `9791e3c`.
        max_bbox_span: Optional maximum tight bounding-box height/width for a
                       sampled shape. Oversized samples are rejected and
                       resampled.
        max_sample_attempts: Maximum per-shape resampling attempts before
                             raising.
        allow_constructive_shapes: Allow the filter scene sampler to bypass
                                   the categorical generators with direct
                                   constraint-specific shape construction.
    """

    def __init__(
        self,
        generators: tuple[Generator, ...] = DEFAULT_GENERATORS,
        weights: tuple[float, ...] | None = None,
        size_range: tuple[int, int] = (4, 16),
        compactness: float = _DEFAULT_COMPACTNESS,
        bordered_hollow_min_size: int | None = None,
        bordered_hollow_random_inner_size: bool = False,
        max_bbox_span: int | None = None,
        max_sample_attempts: int = 64,
        allow_constructive_shapes: bool = False,
    ):
        if not generators:
            raise ValueError("ShapeFactory requires at least one generator")
        if weights is not None:
            if len(weights) != len(generators):
                raise ValueError(
                    f"weights length {len(weights)} != generators length {len(generators)}"
                )
            if any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError(f"invalid weights {weights!r}")
            self._probs = tuple(w / sum(weights) for w in weights)
        else:
            self._probs = (1.0 / len(generators),) * len(generators)

        if size_range[0] < 1 or size_range[1] < size_range[0]:
            raise ValueError(f"invalid size_range {size_range!r}")
        if compactness < 0:
            raise ValueError(f"compactness must be >= 0, got {compactness}")
        if bordered_hollow_min_size is not None and bordered_hollow_min_size < 1:
            raise ValueError(
                "bordered_hollow_min_size must be >= 1 or None, "
                f"got {bordered_hollow_min_size}"
            )
        if max_bbox_span is not None and max_bbox_span < 1:
            raise ValueError(
                f"max_bbox_span must be >= 1 or None, got {max_bbox_span}"
            )
        if max_sample_attempts < 1:
            raise ValueError(
                f"max_sample_attempts must be >= 1, got {max_sample_attempts}"
            )

        self.generators = tuple(generators)
        self.size_range = size_range
        self.compactness = float(compactness)
        self.bordered_hollow_min_size = bordered_hollow_min_size
        self.bordered_hollow_random_inner_size = bool(
            bordered_hollow_random_inner_size
        )
        self.max_bbox_span = max_bbox_span
        self.max_sample_attempts = int(max_sample_attempts)
        self.allow_constructive_shapes = bool(allow_constructive_shapes)

    def _sample_with_generator(
        self,
        gen: Generator,
        rng: np.random.Generator,
        palette: Sequence[int],
    ) -> Shape:
        lo, hi = self.size_range
        size = lo if hi == lo else int(rng.integers(lo, hi))
        if (
            self.bordered_hollow_min_size is not None
            and gen in _BORDERED_HOLLOW_GENERATORS
        ):
            size = max(size, int(self.bordered_hollow_min_size))
        if gen in _COMPACTNESS_GENERATORS:
            kwargs = {"compactness": self.compactness}
            if gen in _BORDERED_HOLLOW_GENERATORS:
                kwargs["random_inner_size"] = (
                    self.bordered_hollow_random_inner_size
                )
            return gen(rng, size, palette, **kwargs)
        return gen(rng, size, palette)

    def _accepts_shape(self, shape: Shape) -> bool:
        return (
            self.max_bbox_span is None
            or max(shape.n_rows, shape.n_cols) <= self.max_bbox_span
        )

    def sample_from(
        self,
        generators: Sequence[Generator],
        rng: np.random.Generator,
        palette: Sequence[int],
        *,
        max_attempts: int | None = None,
        transform: Callable[[Shape], Shape] | None = None,
        predicate: Callable[[Shape], bool] | None = None,
    ) -> Shape:
        if not generators:
            raise ValueError("generators must be non-empty")
        attempts = (
            self.max_sample_attempts
            if max_attempts is None
            else int(max_attempts)
        )
        if attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {attempts}")
        generators = tuple(generators)
        for _ in range(attempts):
            gen = generators[int(rng.integers(0, len(generators)))]
            shape = self._sample_with_generator(gen, rng, palette)
            if not self._accepts_shape(shape):
                continue
            if transform is not None:
                shape = transform(shape)
            if self._accepts_shape(shape) and (
                predicate is None or predicate(shape)
            ):
                return shape
        raise RuntimeError(
            "failed to sample acceptable shape after "
            f"{attempts} attempts"
        )

    def sample(self, rng: np.random.Generator, palette: Sequence[int]) -> Shape:
        for _ in range(self.max_sample_attempts):
            gen_idx = int(rng.choice(len(self.generators), p=list(self._probs)))
            shape = self._sample_with_generator(
                self.generators[gen_idx], rng, palette,
            )
            if self._accepts_shape(shape):
                return shape
        raise RuntimeError(
            "failed to sample shape within max_bbox_span="
            f"{self.max_bbox_span} after {self.max_sample_attempts} attempts"
        )


# === Placement ============================================================


def place_shapes(
    shapes: list[Shape],
    world_size: int = 16,
    gap: int = 2,
    rng: np.random.Generator | None = None,
    max_attempts: int = 300,
) -> tuple[np.ndarray, list[Shape]] | None:
    """Place shapes on a `world_size × world_size` grid.

    Shapes are kept at least `gap` empty pixels from each other and from the
    world edges. All-or-nothing: if any shape cannot be placed in
    `max_attempts`, returns `None` so the caller can retry with a fresh set.

    Returns `(world_grid, placed_shapes)` on success. `placed_shapes` are
    new Shapes with assigned positions; the input list is unchanged.
    """
    if rng is None:
        rng = np.random.default_rng()

    world = np.zeros((world_size, world_size), dtype=np.int64)
    occupied = np.zeros((world_size, world_size), dtype=bool)
    placed: list[Shape] = []

    for s in shapes:
        r_max = world_size - s.n_rows - gap + 1
        c_max = world_size - s.n_cols - gap + 1
        if r_max <= gap or c_max <= gap:
            return None  # shape too large to fit with gap

        placed_ok = False
        for _ in range(max_attempts):
            r = int(rng.integers(gap, r_max))
            c = int(rng.integers(gap, c_max))

            ok = True
            for i in range(s.n_rows):
                if not ok:
                    break
                for j in range(s.n_cols):
                    if s.grid[i, j] == 0:
                        continue
                    wr, wc = r + i, c + j
                    if wr >= world_size or wc >= world_size or occupied[wr, wc]:
                        ok = False
                        break

            if not ok:
                continue

            positioned = s.move_to((r, c))
            positioned.place_on(world)
            for i in range(s.n_rows):
                for j in range(s.n_cols):
                    if s.grid[i, j] == 0:
                        continue
                    for di in range(-gap, gap + 1):
                        for dj in range(-gap, gap + 1):
                            wr, wc = r + i + di, c + j + dj
                            if 0 <= wr < world_size and 0 <= wc < world_size:
                                occupied[wr, wc] = True
            placed.append(positioned)
            placed_ok = True
            break

        if not placed_ok:
            return None

    return world, placed
