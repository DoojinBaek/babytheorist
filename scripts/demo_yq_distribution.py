"""Plot the distribution of |Y_q| per eval task, for levels 1..3.

|Y_q| is the number of grids in `valid_query_outputs` — i.e. how many
outputs are consistent with the demo under selector-ambiguity. `|Y_q|=1`
means the program is fully disambiguated by the demo; larger values mean
the model has multiple valid hypotheses.

Usage:
    python scripts/demo_yq_distribution.py
    # saves scripts/output/eval_yq_distribution.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np


LEVELS = [1, 2, 3]
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

BAR_COLOR = "#0074D9"


def _load_sizes(level: int) -> list[int]:
    path = DATA_DIR / f"lv{level}_eval.h5"
    with h5py.File(path, "r") as f:
        return [len(json.loads(s)) for s in f["valid_query_outputs"].asstr()]


def main() -> None:
    all_sizes = {lv: _load_sizes(lv) for lv in LEVELS}
    global_max = max(max(s) for s in all_sizes.values())
    # Shared x-axis so level comparison is visual.
    xmax = global_max + 1

    fig, axes = plt.subplots(
        len(LEVELS), 1,
        figsize=(6.5, 3.0 * len(LEVELS)),
        squeeze=False,
    )

    for ax, lv in zip(axes[:, 0], LEVELS):
        sizes = all_sizes[lv]
        counter = Counter(sizes)
        xs = np.arange(1, xmax + 1)
        ys = np.array([counter.get(int(x), 0) for x in xs])
        ax.bar(xs, ys, color=BAR_COLOR, edgecolor="white", linewidth=0.3)

        mean = float(np.mean(sizes))
        median = int(np.median(sizes))
        max_val = int(np.max(sizes))
        frac1 = counter.get(1, 0) / len(sizes)

        ax.axvline(mean, color="#FF4136", linewidth=1.4, linestyle="--",
                   label=f"mean = {mean:.2f}")
        ax.set_title(
            f"Level {lv}   n={len(sizes)}\n"
            f"mean={mean:.2f}  median={median}  max={max_val}  |Y_q|=1 frac={frac1:.1%}",
            fontsize=11, family="monospace", fontweight="bold",
        )
        ax.set_xlabel("|Y_q|", fontsize=11)
        ax.set_ylabel("# tasks", fontsize=11)
        ax.set_xlim(0.5, xmax + 0.5)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=9, frameon=False)

    fig.suptitle(
        "|Y_q| distribution across eval tasks (log-scale y)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT_DIR / "eval_yq_distribution.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
