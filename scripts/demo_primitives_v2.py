"""Primitive demo gallery — 4 categories, multiple samples per primitive.

Each primitive is shown with N=4 (input, output) pairs using diverse selectors.
Four figures are produced:
    - Transform-A (footprint-preserving): rot90, flip_h, flip_v, next_color
    - Transform-B (footprint-changing):   fill_interior, hollow, add_border
    - Movement:                            translate_up/down/left/right
    - Relational:                          swap_colors, align_row, align_col

In the input panel, selected shapes (src for relational; src+tgt) are drawn
in full color while unselected shapes are dimmed (alpha=0.25). The output
panel shows the post-step scene at full color.

Usage:
    python scripts/demo_primitives_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from babytheorist.core import ALL, Selector, Step, by_color, is_hollow
from babytheorist.engine import apply_step
from babytheorist.selectors import select
from babytheorist.shapes import ShapeFactory, place_shapes


WORLD = 15
N_SHAPES = 3
N_SAMPLES = 4
SEED = 42
PALETTE = (1, 2, 3, 4, 5)
MAX_SCENE_RETRIES = 300

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


def _sel(t: str) -> Selector:
    return Selector(t)


# Category specs.
# For non-relational primitives: list[Selector]
# For relational primitives:     list[(src_selector, tgt_selector)]
CATEGORIES: list[tuple[str, list[tuple[str, list]]]] = [
    ("Transform-A (footprint-preserving)", [
        ("rot90",      [ALL, by_color(3), _sel("topmost"),    is_hollow()]),
        ("flip_h",     [ALL, by_color(2), _sel("bottommost"), _sel("leftmost")]),
        ("flip_v",     [ALL, by_color(4), _sel("rightmost"),  is_hollow()]),
        ("next_color", [ALL, by_color(1), _sel("topmost"),    _sel("leftmost")]),
    ]),
    ("Transform-B (footprint-changing)", [
        ("fill_interior", [is_hollow(), is_hollow(), is_hollow(), is_hollow()]),
        ("hollow",        [ALL, by_color(2), _sel("topmost"),    _sel("leftmost")]),
        ("add_border",    [ALL, by_color(3), _sel("bottommost"), _sel("rightmost")]),
    ]),
    ("Movement", [
        ("translate_up",    [ALL, by_color(1), _sel("topmost"),    is_hollow()]),
        ("translate_down",  [ALL, by_color(2), _sel("bottommost"), _sel("leftmost")]),
        ("translate_left",  [ALL, by_color(3), _sel("leftmost"),   _sel("topmost")]),
        ("translate_right", [ALL, by_color(4), _sel("rightmost"),  is_hollow()]),
    ]),
    ("Relational", [
        ("swap_colors", [
            (_sel("topmost"),    _sel("bottommost")),
            (_sel("leftmost"),   _sel("rightmost")),
            (_sel("topmost"),    _sel("rightmost")),
            (_sel("bottommost"), _sel("leftmost")),
        ]),
        ("align_row", [
            (_sel("topmost"),    _sel("bottommost")),
            (_sel("bottommost"), _sel("topmost")),
            (_sel("leftmost"),   _sel("rightmost")),
            (_sel("rightmost"),  _sel("leftmost")),
        ]),
        ("align_col", [
            (_sel("leftmost"),   _sel("rightmost")),
            (_sel("rightmost"),  _sel("leftmost")),
            (_sel("topmost"),    _sel("bottommost")),
            (_sel("bottommost"), _sel("topmost")),
        ]),
    ]),
]


def _selector_label(spec) -> str:
    def name(s: Selector) -> str:
        return f"{s.type}({s.param})" if s.param is not None else s.type
    if isinstance(spec, tuple):
        return f"{name(spec[0])} → {name(spec[1])}"
    return name(spec)


def _gen_scene(rng, factory):
    for _ in range(50):
        shapes = [factory.sample(rng, PALETTE) for _ in range(N_SHAPES)]
        placed = place_shapes(shapes, world_size=WORLD, gap=1, rng=rng)
        if placed is not None:
            return placed[1]
    return None


def _scenes_differ(in_shapes, out_shapes) -> bool:
    """True if the program produced a visible change."""
    if len(in_shapes) != len(out_shapes):
        return True
    for a, b in zip(in_shapes, out_shapes):
        if a.position != b.position:
            return True
        if a.grid.shape != b.grid.shape:
            return True
        if not np.array_equal(a.grid, b.grid):
            return True
    return False


def _try_sample(rng, factory, primitive, sel_spec):
    """Generate (in_shapes, out_shapes, hl_idx) for a primitive+selector combo.

    Retries until the selector matches at least one shape, the step succeeds,
    and the output visibly differs from the input. Returns None on exhaustion.
    """
    if isinstance(sel_spec, tuple):
        selector, target = sel_spec
    else:
        selector, target = sel_spec, None

    for _ in range(MAX_SCENE_RETRIES):
        in_shapes = _gen_scene(rng, factory)
        if in_shapes is None:
            continue

        src_idx = select(selector, in_shapes)
        if not src_idx:
            continue
        if target is not None:
            tgt_idx = select(target, in_shapes)
            if not tgt_idx:
                continue
            tgt_set = {tgt_idx[0]}
        else:
            tgt_set = set()

        step = Step(selector, primitive, target)
        out_shapes = apply_step(step, in_shapes, WORLD)
        if out_shapes is None:
            continue
        if not _scenes_differ(in_shapes, out_shapes):
            continue

        return in_shapes, out_shapes, set(src_idx) | tgt_set

    return None


def _draw_cell(ax, shapes, highlighted_idx):
    cmap = mcolors.ListedColormap(ARC_COLORS)

    dim_layer = np.zeros((WORLD, WORLD), dtype=np.int64)
    full_layer = np.zeros((WORLD, WORLD), dtype=np.int64)
    for i, s in enumerate(shapes):
        if s.is_null:
            continue
        layer = full_layer if i in highlighted_idx else dim_layer
        try:
            s.place_on(layer)
        except Exception:
            pass

    ax.imshow(np.zeros((WORLD, WORLD), dtype=np.int64),
              cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    dim_masked = np.ma.masked_where(dim_layer == 0, dim_layer)
    ax.imshow(dim_masked, cmap=cmap, vmin=0, vmax=9,
              interpolation="nearest", alpha=0.25)
    full_masked = np.ma.masked_where(full_layer == 0, full_layer)
    ax.imshow(full_masked, cmap=cmap, vmin=0, vmax=9,
              interpolation="nearest", alpha=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-0.5, WORLD, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, WORLD, 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=0.3)
    ax.tick_params(which="minor", length=0)


def _empty_cell(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=9, color="grey", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def make_category_figure(name: str, specs, rng, factory, out_path: Path):
    n_rows = len(specs)
    n_cols = N_SAMPLES * 2  # input + output per sample

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.6, n_rows * 2.0),
        squeeze=False,
    )

    for r, (primitive, selectors) in enumerate(specs):
        for s_idx, sel_spec in enumerate(selectors):
            in_ax = axes[r, s_idx * 2]
            out_ax = axes[r, s_idx * 2 + 1]

            sample = _try_sample(rng, factory, primitive, sel_spec)
            if sample is None:
                _empty_cell(in_ax, "no valid\nscene")
                _empty_cell(out_ax, "—")
            else:
                in_shapes, out_shapes, hl_idx = sample
                _draw_cell(in_ax, in_shapes, hl_idx)
                _draw_cell(out_ax, out_shapes, set(range(len(out_shapes))))

            in_ax.set_title(_selector_label(sel_spec), fontsize=9, family="monospace")
            out_ax.set_title("→ output", fontsize=9, family="monospace", color="#666")

        axes[r, 0].set_ylabel(
            primitive, fontsize=13, family="monospace",
            fontweight="bold", rotation=90,
        )

    # Color legend at bottom, right-aligned
    legend_handles = [
        mpatches.Patch(color=ARC_COLORS[i], label=COLOR_NAMES[i])
        for i in range(1, 10)
    ]
    fig.legend(
        handles=legend_handles, loc="lower right", ncol=9,
        fontsize=10, frameon=False, bbox_to_anchor=(1.0, -0.01),
    )

    fig.suptitle(name, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    factory = ShapeFactory()

    for name, specs in CATEGORIES:
        slug = name.split()[0].lower().replace("-", "_")
        out_path = OUT_DIR / f"primitives_{slug}.png"
        make_category_figure(name, specs, rng, factory, out_path)


if __name__ == "__main__":
    main()
