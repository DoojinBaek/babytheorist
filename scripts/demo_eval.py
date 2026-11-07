"""Eval-transfer task gallery — demo, query, and valid query outputs Y_q.

Assumes **P=2 (K=1 support pair)** — the current BabyTheorist generator output. For
K>1 data this script needs extending (render all K support pairs per row).

Each row is one eval task from `data/lv{level}_eval.h5`:
    [demo x | demo y | query x_q | Y_q[0] | Y_q[1] | ... | Y_q[K]]

The canonical y_q (= pairs[1].output_grid, the first element of Y_q by
construction) is highlighted with a ★ in its title. Extra Y_q columns
past the max shown are collapsed into a "+N more" note in the row label.

The program is printed to the left of each row in monospace.

Usage:
    python scripts/demo_eval.py             # writes scripts/output/eval_demo_lv3.png
    LEVEL=2 python scripts/demo_eval.py     # other levels (needs data file)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


LEVEL = int(os.environ.get("LEVEL", "3"))
WORLD = 15
MAX_YQ_SHOWN = 6  # cap Y_q columns to keep figure width sane
N_ROWS = 6        # tasks per figure
SEED = 0          # picks are deterministic given a seed over available tasks

IN_PATH = Path(__file__).parent.parent / "data" / f"lv{LEVEL}_eval.h5"
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


def _render_program(program: list[dict]) -> str:
    def sel(s: dict) -> str:
        return f"{s['type']}({s['param']})" if s.get("param") is not None else s["type"]
    def step(s: dict) -> str:
        base = f"{sel(s['selector'])}.{s['primitive']}"
        return f"{base}(→{sel(s['target'])})" if s.get("target") else base
    return " → ".join(step(s) for s in program) if program else "(empty)"


def _draw(ax, grid: np.ndarray, title: str | None = None) -> None:
    cmap = mcolors.ListedColormap(ARC_COLORS)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    if title is not None:
        ax.set_title(title, fontsize=11, family="monospace", fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="#444444", linewidth=0.3)
    ax.tick_params(which="minor", length=0)


def _load_eval(path: Path) -> dict:
    from env import io_h5
    with h5py.File(path, "r") as f:
        n, P = f["inputs"].shape[:2]
        # Guard: this script's rendering is hardcoded to K=1 (single support
        # pair). Fail loud on ragged / multi-support data rather than silently
        # drawing the wrong slots.
        has_n_support = "n_support" in f
        if P != 2 or has_n_support:
            raise ValueError(
                f"{path}: demo_eval.py supports only P=2 (K=1) eval files; "
                f"got P={P}, has n_support={has_n_support}. "
                "Extend this script to render all K support pairs for K>1 data."
            )
        rows = [io_h5.read_eval_row(f, i, P) for i in range(n)]
        return {
            "inputs": f["inputs"][:],
            "outputs": f["outputs"][:],
            "programs": [io_h5.read_program(f, i) for i in range(n)],
            # Y_q already canonicalized (Y_q[0] == y_q) by io_h5.
            "Y_q": [list(r["Y_q"]) for r in rows],
        }


def _pick_tasks(data: dict, n: int, seed: int) -> list[int]:
    """Pick a diverse spread of tasks by |Y_q|: at least one with |Y_q|=1
    (no ambiguity), then increasing ambiguity up to the max available."""
    sizes = [len(y) for y in data["Y_q"]]
    rng = np.random.default_rng(seed)

    # Bucket by |Y_q|. Sample ladder of sizes.
    buckets: dict[int, list[int]] = {}
    for i, s in enumerate(sizes):
        buckets.setdefault(s, []).append(i)
    target_sizes = sorted(buckets)
    if len(target_sizes) > n:
        # Pick n roughly uniformly across the ladder (always include min and max)
        step = len(target_sizes) / n
        chosen_sizes = [target_sizes[min(int(i * step), len(target_sizes) - 1)] for i in range(n)]
    else:
        chosen_sizes = target_sizes

    picks: list[int] = []
    for s in chosen_sizes:
        picks.append(int(rng.choice(buckets[s])))
    # Top up from any remaining tasks if under n
    while len(picks) < n:
        spare = [i for i in range(len(sizes)) if i not in picks]
        picks.append(int(rng.choice(spare)))
    return picks[:n]


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(
            f"missing dataset: {IN_PATH}. Generate an eval HDF5 file first."
        )

    data = _load_eval(IN_PATH)
    picks = _pick_tasks(data, n=N_ROWS, seed=SEED)
    # Sort descending by |Y_q| so row 0 has the most Y_q entries and the
    # column headers "Y_q[0]..Y_q[K]" get rendered on a fully-filled row.
    picks.sort(key=lambda i: -len(data["Y_q"][i]))

    n_cols = 3 + MAX_YQ_SHOWN  # demo x, demo y, query x, then Y_q columns
    fig = plt.figure(figsize=(n_cols * 2.0, N_ROWS * 2.45))
    outer = gridspec.GridSpec(N_ROWS, 1, figure=fig, hspace=0.28)

    for r, idx in enumerate(picks):
        inp = data["inputs"][idx]    # (2, H, W)
        out = data["outputs"][idx]   # (2, H, W)
        Y_q = list(data["Y_q"][idx])  # canonical already at index 0 (via io_h5)
        program = data["programs"][idx]

        # Nested grid: image row (all cols) + text row (spans all cols).
        inner = gridspec.GridSpecFromSubplotSpec(
            2, n_cols, subplot_spec=outer[r],
            height_ratios=[8, 1], hspace=0.05, wspace=0.08,
        )
        image_axes = [fig.add_subplot(inner[0, c]) for c in range(n_cols)]
        text_ax = fig.add_subplot(inner[1, :])

        # Left panel (demo + query) — column headers on first row only
        _draw(image_axes[0], inp[0], "demo x" if r == 0 else None)
        _draw(image_axes[1], out[0], "demo y" if r == 0 else None)
        _draw(image_axes[2], inp[1], "query x_q" if r == 0 else None)

        # Y_q columns (up to MAX_YQ_SHOWN); canonical (j==0) gets ★ marker
        k = min(len(Y_q), MAX_YQ_SHOWN)
        for j in range(k):
            if r == 0:
                header = f"Y_q[{j}]" + (" ★" if j == 0 else "")
            else:
                header = "★" if j == 0 else None
            _draw(image_axes[3 + j], Y_q[j], header)

        # Blank unused Y_q columns
        for j in range(k, MAX_YQ_SHOWN):
            image_axes[3 + j].axis("off")

        # Text row: program below the image row, centered
        prog_text = _render_program(program)
        total = len(data["Y_q"][idx])
        tag = f"|Y_q| = {total}"
        if total > MAX_YQ_SHOWN:
            tag += f"  (showing {MAX_YQ_SHOWN})"
        text_ax.text(
            0.5, 0.5,
            f"{prog_text}    ·    {tag}",
            ha="center", va="center",
            fontsize=11, family="monospace",
            transform=text_ax.transAxes,
        )
        text_ax.axis("off")

    # Color legend at bottom
    legend_handles = [
        mpatches.Patch(color=ARC_COLORS[i], label=COLOR_NAMES[i])
        for i in range(1, 10)
    ]
    fig.legend(
        handles=legend_handles, loc="lower right", ncol=9,
        fontsize=10, frameon=False, bbox_to_anchor=(1.0, -0.01),
    )

    fig.suptitle(
        f"Level {LEVEL} eval — ★ marks canonical y_q "
        f"(= pairs[1].output)  ·  {N_ROWS} tasks sorted by |Y_q|",
        fontsize=14, fontweight="bold",
    )
    # No tight_layout here — nested GridSpec already controls spacing.
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.03, right=0.97)

    out_path = OUT_DIR / f"eval_demo_lv{LEVEL}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
