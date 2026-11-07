"""Shape generation gallery — one column per coloring mode, rows for object count.

Columns (8): solid, bordered, hollow, striped_h, striped_v, split_h, split_v,
checkerboard.
Rows (4): 1, 2, 3, 4 objects per scene.

Usage:
    python scripts/demo_shapes.py
    # saves scripts/output/shapes_gallery.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from babytheorist.shapes import (
    PATTERN_TYPES, bordered, hollow, patterned, place_shapes, solid,
)


WORLD = 16
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

ARC_COLORS: tuple[str, ...] = (
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
)

N_OBJECTS_PER_ROW: tuple[int, ...] = (1, 2, 3, 4)

# (column label, generator callable taking (rng, size, palette) → Shape)
COLUMNS: list[tuple[str, callable]] = [
    ("solid", lambda rng, size, palette: solid(rng, size, palette)),
    ("bordered", lambda rng, size, palette: bordered(rng, size, palette)),
    ("hollow", lambda rng, size, palette: hollow(rng, size, palette)),
    *[
        (pat, (lambda p: lambda rng, size, palette: patterned(rng, size, palette, pattern=p))(pat))
        for pat in PATTERN_TYPES
    ],
]


def _draw(ax, grid: np.ndarray) -> None:
    """Pixel-perfect grid via pcolormesh — edges align exactly with cells."""
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(ARC_COLORS)
    h, w = grid.shape
    ax.pcolormesh(
        np.arange(w + 1), np.arange(h + 1), grid,
        cmap=cmap, vmin=0, vmax=9,
        edgecolors="#333333", linewidth=0.25, antialiased=False,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _scene(n: int, generator, rng: np.random.Generator) -> np.ndarray:
    palette = tuple(int(c) for c in rng.choice(range(1, 10), size=6, replace=False))
    for _ in range(30):
        shapes = [generator(rng, int(rng.integers(5, 12)), palette) for _ in range(n)]
        placed = place_shapes(shapes, world_size=WORLD, gap=1, rng=rng)
        if placed is not None:
            return placed[0]
    return np.zeros((WORLD, WORLD), dtype=np.int64)


def main() -> None:
    rng = np.random.default_rng(7)
    n_rows = len(N_OBJECTS_PER_ROW)
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.0), squeeze=False)

    for r, n in enumerate(N_OBJECTS_PER_ROW):
        for c, (_, gen) in enumerate(COLUMNS):
            _draw(axes[r, c], _scene(n, gen, rng))

    # Column headers
    for c, (label, _) in enumerate(COLUMNS):
        axes[-1, c].set_xlabel(label, fontsize=11, family="monospace")

    # Row labels
    for r, n in enumerate(N_OBJECTS_PER_ROW):
        axes[r, 0].set_ylabel(f"{n} obj", fontsize=11, family="monospace", rotation=90)

    fig.suptitle("Shape Generation Gallery", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = OUT_DIR / "shapes_gallery.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
