"""Matplotlib visualization for BabyTheorist tasks.

Optional dependency: matplotlib. Imports are deferred to call time so the
core `babytheorist/` library can be installed in a matplotlib-free environment.

    from utils.viz import render_task, render_dataset

    render_task(tasks[0], path="task.png")
    render_dataset(tasks[:10], path="preview.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from babytheorist.core import Selector, Step, Task

if TYPE_CHECKING:
    import matplotlib.figure


def _render_program(program: list[Step]) -> str:
    """Human-readable one-liner for a Step-based program."""
    def sel(s: Selector) -> str:
        return f"{s.type}({s.param})" if s.param is not None else s.type

    def step(s: Step) -> str:
        base = f"{sel(s.selector)}.{s.primitive}"
        return f"{base}(→{sel(s.target)})" if s.target is not None else base

    return " → ".join(step(s) for s in program) if program else "(empty)"


# ARC standard 10-color palette (index 0 = background / empty cell).
ARC_COLORS: tuple[str, ...] = (
    "#000000",  # 0  black     (empty)
    "#0074D9",  # 1  blue
    "#FF4136",  # 2  red
    "#2ECC40",  # 3  green
    "#FFDC00",  # 4  yellow
    "#AAAAAA",  # 5  grey
    "#F012BE",  # 6  magenta
    "#FF851B",  # 7  orange
    "#7FDBFF",  # 8  cyan
    "#870C25",  # 9  maroon
)


def _draw_grid(ax, grid: np.ndarray, title: str) -> None:
    import matplotlib.colors as mcolors

    cmap = mcolors.ListedColormap(ARC_COLORS)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)


def render_task(
    task: Task,
    path: str | Path | None = None,
) -> "matplotlib.figure.Figure | None":
    """Render a single task as a 2-column figure (one row per pair).

    If `path` is given, save and close the figure (returns None). Otherwise
    return the figure for further inspection. Raises ValueError if the task
    has no pairs.
    """
    import matplotlib.pyplot as plt

    if not task.pairs:
        raise ValueError("render_task: task has no pairs")
    n = len(task.pairs)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.4 * n), squeeze=False)
    for i, pair in enumerate(task.pairs):
        _draw_grid(axes[i, 0], pair.input_grid, f"input {i}")
        _draw_grid(axes[i, 1], pair.output_grid, f"output {i}")

    program = _render_program(task.program)
    fig.suptitle(
        f"level {task.level}   depth {len(task.program)}\n{program}",
        fontsize=10,
    )
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


def render_dataset(
    tasks: list[Task],
    path: str | Path | None = None,
    max_tasks: int = 12,
) -> "matplotlib.figure.Figure | None":
    """Render up to `max_tasks` tasks side-by-side, one row per task.

    Each row shows the *first* (input, output) pair of that task plus the
    program annotation. Useful for getting a quick visual sense of a dataset.
    """
    import matplotlib.pyplot as plt

    tasks = [t for t in list(tasks)[:max_tasks] if t.pairs]
    if not tasks:
        return None

    n = len(tasks)
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.2 * n), squeeze=False)
    for i, task in enumerate(tasks):
        pair = task.pairs[0]
        program = _render_program(task.program)
        _draw_grid(axes[i, 0], pair.input_grid, f"L{task.level} d{len(task.program)}  input")
        _draw_grid(axes[i, 1], pair.output_grid, program)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
