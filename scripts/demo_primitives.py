"""Visual demo of the 14 BabyTheorist primitives — one input → output pair each.

Each panel shows a random scene on the left and the scene after one
`Step(selector, primitive)` on the right, with an arrow between the two
grids and the primitive+selector centered above the pair.  For relational
primitives two selectors (src → tgt) are used.  Scenes are re-rolled until
every *selected* object actually changes, so the demo never shows a silent
no-op.

Usage:
    python scripts/demo_primitives.py
    # saves scripts/output/primitives_demo.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

from babytheorist.core import (
    ALL, Selector, Shape, Step,
    by_color, is_hollow, topmost, bottommost, leftmost, rightmost,
)
from babytheorist.engine import _rebuild_grid, apply_step
from babytheorist.primitives import category_of
from babytheorist.selectors import select
from babytheorist.shapes import ShapeFactory, place_shapes


WORLD = 16
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

ARC_COLORS: tuple[str, ...] = (
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
)

# One "interesting" selector per primitive (keeps the demo visually varied).
PRIMITIVE_SELECTORS: dict[str, Selector] = {
    "rot90": ALL,
    "flip_h": by_color(1),
    "flip_v": ALL,
    "next_color": ALL,
    "fill_interior": is_hollow(),
    "hollow": by_color(2),
    "add_border": ALL,
    "translate_up": ALL,
    "translate_down": ALL,
    "translate_left": ALL,
    "translate_right": ALL,
    "swap_colors": topmost(),
    "align_row": leftmost(),
    "align_col": topmost(),
}
RELATIONAL_TARGETS: dict[str, Selector] = {
    "swap_colors": bottommost(),
    "align_row": rightmost(),
    "align_col": bottommost(),
}

PRIMITIVE_ORDER: list[str] = [
    "rot90", "flip_h", "flip_v", "fill_interior",
    "hollow", "add_border", "next_color", "translate_up",
    "translate_down", "translate_left", "translate_right", "swap_colors",
    "align_row", "align_col",
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


def _random_scene(rng: np.random.Generator, n: int = 3) -> list[Shape]:
    factory = ShapeFactory(size_range=(6, 13))
    palette = tuple(int(c) for c in rng.choice(range(1, 10), size=6, replace=False))
    for _ in range(30):
        shapes = [factory.sample(rng, palette) for _ in range(n)]
        placed = place_shapes(shapes, world_size=WORLD, gap=1, rng=rng)
        if placed is not None:
            return placed[1]
    return []


def _label_selector(sel: Selector) -> str:
    m = {"ALL": "ALL", "by_color": f"by_color {sel.param}", "is_hollow": "is_hollow",
         "topmost": "topmost", "bottommost": "bottommost",
         "leftmost": "leftmost", "rightmost": "rightmost"}
    return m[sel.type]


def _block_title(name: str) -> str:
    sel = PRIMITIVE_SELECTORS[name]
    if category_of(name) == "relational":
        tgt = RELATIONAL_TARGETS[name]
        return f"{name}\n({_label_selector(sel)} ↔ {_label_selector(tgt)})"
    return f"{name}\n({_label_selector(sel)})"


def _demo_pair(name: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a scene where every selected shape visibly changes."""
    rng = np.random.default_rng(seed)
    sel = PRIMITIVE_SELECTORS[name]
    target = RELATIONAL_TARGETS.get(name)
    step = Step(sel, name, target=target)

    for _ in range(80):
        shapes = _random_scene(rng)
        if len(shapes) < (2 if target is not None else 1):
            continue
        selected = select(sel, shapes)
        if not selected:
            continue
        out_shapes = apply_step(step, shapes, WORLD)
        if out_shapes is None:
            continue
        if not all(
            not np.array_equal(out_shapes[i].grid, shapes[i].grid)
            or out_shapes[i].position != shapes[i].position
            for i in selected
        ):
            continue
        in_grid = _rebuild_grid(shapes, WORLD)
        out_grid = _rebuild_grid(out_shapes, WORLD)
        if in_grid is None or out_grid is None:
            continue
        return in_grid, out_grid
    return None


def main() -> None:
    n_cols = 4
    n_rows = (len(PRIMITIVE_ORDER) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(n_cols * 4.0, n_rows * 2.6))
    outer = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        wspace=0.28, hspace=0.70,
        top=0.93, bottom=0.04, left=0.03, right=0.97,
    )

    for idx, name in enumerate(PRIMITIVE_ORDER):
        r, c = divmod(idx, n_cols)
        pair = _demo_pair(name, seed=idx * 13 + 1)
        if pair is None:
            continue
        in_grid, out_grid = pair

        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[r, c], wspace=0.18,
        )
        ax_in = fig.add_subplot(inner[0, 0])
        ax_out = fig.add_subplot(inner[0, 1])
        _draw(ax_in, in_grid)
        _draw(ax_out, out_grid)

        # Centered title above the pair
        pos = outer[r, c].get_position(fig)
        cx = (pos.x0 + pos.x1) / 2
        fig.text(
            cx, pos.y1 + 0.012, _block_title(name),
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            family="monospace",
        )

        # Arrow between grids, at vertical center of the pair
        p_in = ax_in.get_position()
        p_out = ax_out.get_position()
        arrow_x = (p_in.x1 + p_out.x0) / 2
        arrow_y = (p_in.y0 + p_in.y1) / 2
        fig.text(arrow_x, arrow_y, "→", ha="center", va="center",
                 fontsize=18, fontweight="bold", color="#666666")

    fig.suptitle("BabyTheorist Primitives", fontsize=16, fontweight="bold", y=0.985)
    out_path = OUT_DIR / "primitives_demo.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
