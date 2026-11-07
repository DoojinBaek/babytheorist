"""Selector demo gallery — 5 scenes × 7 selectors.

Layout:
    rows: 5 randomly-generated scenes, 3 shapes each, 15x15 world.
    cols: original + 7 selectors (ALL, by_color(3), is_hollow,
          topmost, bottommost, leftmost, rightmost)

Selected shapes are drawn in full color; unselected are dimmed (alpha=0.25).
A color legend is rendered at the bottom of the figure.

Usage:
    python scripts/demo_selectors.py
    # saves scripts/output/selectors_demo.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from babytheorist.core import ALL, Selector, by_color, is_hollow
from babytheorist.selectors import select
from babytheorist.shapes import ShapeFactory, place_shapes


WORLD = 15
N_SHAPES = 3
N_SCENES = 5
SEED = 42
PALETTE = (1, 2, 3, 4, 5)
BY_COLOR_PARAM = 3

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

ARC_COLORS: tuple[str, ...] = (
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
)
COLOR_NAMES: tuple[str, ...] = (
    "0:bg", "1:blue", "2:red", "3:green", "4:yellow",
    "5:grey", "6:magenta", "7:orange", "8:cyan", "9:maroon",
)

SELECTORS: list[tuple[str, Selector]] = [
    ("ALL", ALL),
    (f"by_color({BY_COLOR_PARAM})", by_color(BY_COLOR_PARAM)),
    ("is_hollow", is_hollow()),
    ("topmost", Selector("topmost")),
    ("bottommost", Selector("bottommost")),
    ("leftmost", Selector("leftmost")),
    ("rightmost", Selector("rightmost")),
]


def _generate_scene(rng, factory):
    for _ in range(50):
        shapes = [factory.sample(rng, PALETTE) for _ in range(N_SHAPES)]
        placed = place_shapes(shapes, world_size=WORLD, gap=1, rng=rng)
        if placed is not None:
            _, positioned = placed
            return positioned
    raise RuntimeError("failed to place shapes after 50 retries")


def _draw_cell(ax, shapes, selected_idx):
    """Draw shapes; non-selected ones with alpha=0.25."""
    cmap = mcolors.ListedColormap(ARC_COLORS)

    dim_layer = np.zeros((WORLD, WORLD), dtype=np.int64)
    full_layer = np.zeros((WORLD, WORLD), dtype=np.int64)
    for i, s in enumerate(shapes):
        target = full_layer if i in selected_idx else dim_layer
        s.place_on(target)

    # Background (all zeros, drawn as black to match palette index 0)
    ax.imshow(np.zeros((WORLD, WORLD), dtype=np.int64),
              cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    # Dim layer (mask zeros so background shows through)
    dim_masked = np.ma.masked_where(dim_layer == 0, dim_layer)
    ax.imshow(dim_masked, cmap=cmap, vmin=0, vmax=9,
              interpolation="nearest", alpha=0.25)
    # Selected layer at full opacity
    full_masked = np.ma.masked_where(full_layer == 0, full_layer)
    ax.imshow(full_masked, cmap=cmap, vmin=0, vmax=9,
              interpolation="nearest", alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, WORLD, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, WORLD, 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=0.3)
    ax.tick_params(which="minor", length=0)


def main() -> None:
    rng = np.random.default_rng(SEED)
    factory = ShapeFactory()

    scenes = [_generate_scene(rng, factory) for _ in range(N_SCENES)]

    n_rows = N_SCENES
    n_cols = 1 + len(SELECTORS)  # original + 7
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.0, n_rows * 2.0),
        squeeze=False,
    )

    # Col 0: original (everything full color)
    for r, shapes in enumerate(scenes):
        _draw_cell(axes[r, 0], shapes, set(range(len(shapes))))

    # Cols 1..7: selectors
    for c, (label, sel) in enumerate(SELECTORS, start=1):
        for r, shapes in enumerate(scenes):
            indices = set(select(sel, shapes))
            _draw_cell(axes[r, c], shapes, indices)

    # Column headers (top of first row)
    axes[0, 0].set_title("original", fontsize=14, family="monospace", fontweight="bold")
    for c, (label, _) in enumerate(SELECTORS, start=1):
        axes[0, c].set_title(label, fontsize=14, family="monospace", fontweight="bold")

    # Row labels (left side)
    for r in range(N_SCENES):
        axes[r, 0].set_ylabel(f"scene {r + 1}", fontsize=11, family="monospace", rotation=90)

    # Color legend at bottom, right-aligned
    legend_handles = [
        mpatches.Patch(color=ARC_COLORS[i], label=COLOR_NAMES[i])
        for i in range(1, 10)
    ]
    fig.legend(
        handles=legend_handles, loc="lower right", ncol=9,
        fontsize=10, frameon=False, bbox_to_anchor=(1.0, -0.01),
    )

    fig.suptitle("Selector Types", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    out_path = OUT_DIR / "selectors_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
