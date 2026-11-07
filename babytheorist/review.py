"""Save, load, and pretty-print BabyTheorist tasks.

Two serialization formats, dispatched by `save`/`load` on file suffix:
    `.json`        — human-readable, round-trippable, default for BabyTheorist internals.
    `.h5`/`.hdf5`  — task-row HDF5 export (`(N, P, H, W)`) intended for
                     downstream loaders. Write-only; see `_save_h5`.
ASCII is the rendering format (no external plot dependency).

Step serialization schema:
    {"selector": {"type": "by_color", "param": 3},
     "primitive": "rot90",
     "target": null}             # relational: nested selector dict
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .core import Program, Selector, Step, Task, TaskPair


# === Selector / Step codec ================================================


def _selector_to_dict(s: Selector) -> dict:
    d: dict = {"type": s.type}
    if s.param is not None:
        d["param"] = s.param
    return d


def _selector_from_dict(d: dict) -> Selector:
    return Selector(type=d["type"], param=d.get("param"))


def _step_to_dict(s: Step) -> dict:
    return {
        "selector": _selector_to_dict(s.selector),
        "primitive": s.primitive,
        "target": _selector_to_dict(s.target) if s.target is not None else None,
    }


def _step_from_dict(d: dict) -> Step:
    tgt = d.get("target")
    return Step(
        selector=_selector_from_dict(d["selector"]),
        primitive=d["primitive"],
        target=_selector_from_dict(tgt) if tgt is not None else None,
    )


def _program_to_list(program: Program) -> list[dict]:
    return [_step_to_dict(s) for s in program]


def _program_from_list(items: list[dict]) -> Program:
    return [_step_from_dict(d) for d in items]


def _render_selector(s: Selector) -> str:
    return f"{s.type}({s.param})" if s.param is not None else s.type


def _render_step(s: Step) -> str:
    base = f"{_render_selector(s.selector)}.{s.primitive}"
    return f"{base}(→{_render_selector(s.target)})" if s.target is not None else base


# === Task codec ===========================================================


def task_to_dict(task: Task) -> dict:
    out: dict = {
        "level": task.level,
        "program": _program_to_list(task.program),
        "pairs": [
            {
                "input": p.input_grid.tolist(),
                "output": p.output_grid.tolist(),
                "n_shapes": p.n_shapes,
            }
            for p in task.pairs
        ],
    }
    if task.valid_query_outputs is not None:
        out["valid_query_outputs"] = [g.tolist() for g in task.valid_query_outputs]
    if task.n_support is not None:
        out["n_support"] = task.n_support
    return out


def task_from_dict(d: dict) -> Task:
    vqo = d.get("valid_query_outputs")
    return Task(
        level=d["level"],
        program=_program_from_list(d["program"]),
        pairs=[
            TaskPair(
                input_grid=np.array(p["input"], dtype=np.int64),
                output_grid=np.array(p["output"], dtype=np.int64),
                n_shapes=p.get("n_shapes", 0),
            )
            for p in d["pairs"]
        ],
        valid_query_outputs=(
            [np.array(g, dtype=np.int64) for g in vqo] if vqo is not None else None
        ),
        n_support=d.get("n_support"),
    )


_H5_SUFFIXES = (".h5", ".hdf5")


def save(tasks: Task | list[Task], path: str | Path,
         extra_info: dict | None = None) -> None:
    """Write tasks to `path` — HDF5 if `.h5`/`.hdf5`, else JSON.

    `extra_info` is merged into the HDF5 sidecar `.info.json` or written
    under the top-level JSON `info` key.
    """
    items = [tasks] if isinstance(tasks, Task) else list(tasks)
    if Path(path).suffix.lower() in _H5_SUFFIXES:
        _save_h5(items, path, extra_info=extra_info)
        return
    payload = {"tasks": [task_to_dict(t) for t in items]}
    if extra_info:
        payload["info"] = extra_info
    Path(path).write_text(json.dumps(payload))


def load(path: str | Path) -> list[Task]:
    """Read a JSON file written by `save`. HDF5 is write-only; use JSON for round-trip."""
    if Path(path).suffix.lower() in _H5_SUFFIXES:
        raise NotImplementedError("load: HDF5 is a write-only export; use JSON")
    payload = json.loads(Path(path).read_text())
    return [task_from_dict(t) for t in payload["tasks"]]


# === HDF5 export (task-row layout, write-only) ============================
# One row per task; each row holds `P = n_pairs` pairs sharing a program:
#
#     inputs, outputs        (N, P, H, W)  int8   grid values 0..9
#     grid_sizes             (N, 2)        int32  (h, w), uniform per file
#     n_shapes               (N, P)        int32  generator ground-truth
#                                                   (0 in padding slots)
#     depths                 (N,)          int32  sampled program depth
#     n_support              (N,)          int32  optional eval support count
#     transformation_suites  (N,)          vlen   JSON-serialized BabyTheorist program
#
# `N = n_tasks`, `P = n_pairs`, guaranteed uniform because `generate_task`
# returns None on any partial failure. Use `P=1` for training (single
# (x, y) per row) and `P>=2` for transfer eval. Eval layout contract:
# support occupies slots `[0, K_i)` and query occupies slot `K_i`. Current
# writer may emit variable K, in which case `P` is the max padded row width.
# Ragged support is supported by readers via an **optional `n_support` (N,)
# int** dataset carrying per-row K_i; padding may fill slots `(K_i + 1, P)`
# and is ignored. In padded slots, inputs/outputs are zero grids and n_shapes
# is 0. When writer adds ragged support, include `n_support`. Key layout is
# BabyTheorist-native, not COGITAO-compatible.


def _save_h5(tasks: list[Task], path: str | Path,
             extra_info: dict | None = None) -> None:
    try:
        import h5py
    except ImportError as e:
        raise ImportError("save: HDF5 export requires h5py (pip install h5py)") from e

    if not tasks or any(not t.pairs for t in tasks):
        raise ValueError("save: HDF5 requires non-empty tasks with pairs")
    n_support_values = [t.n_support for t in tasks]
    has_n_support = any(k is not None for k in n_support_values)
    if has_n_support:
        if any(k is None for k in n_support_values):
            raise ValueError("save: HDF5 requires n_support on every ragged eval task")
        for t in tasks:
            if t.n_support is None or t.n_support < 1:
                raise ValueError(f"save: invalid n_support={t.n_support}")
            if len(t.pairs) != t.n_support + 1:
                raise ValueError(
                    "save: eval task with n_support must have "
                    "support pairs plus one query"
                )
        pair_count = max(int(k) for k in n_support_values) + 1
    else:
        pair_counts = {len(t.pairs) for t in tasks}
        if len(pair_counts) != 1:
            raise ValueError(f"save: HDF5 requires uniform n_pairs per task, got {pair_counts}")
        pair_count = next(iter(pair_counts))
    shapes = {g.shape for t in tasks for p in t.pairs for g in (p.input_grid, p.output_grid)}
    if len(shapes) != 1:
        raise ValueError(f"save: HDF5 requires uniform grid shape, got {shapes}")
    if any(p.n_shapes <= 0 for t in tasks for p in t.pairs):
        raise ValueError("save: HDF5 requires ground-truth n_shapes (>0) on every pair")

    h, w = next(iter(shapes))
    n_tasks = len(tasks)
    inputs = np.zeros((n_tasks, pair_count, h, w), dtype=np.int8)
    outputs = np.zeros((n_tasks, pair_count, h, w), dtype=np.int8)
    n_shapes = np.zeros((n_tasks, pair_count), dtype=np.int32)
    for i, t in enumerate(tasks):
        for j, p in enumerate(t.pairs):
            inputs[i, j] = p.input_grid
            outputs[i, j] = p.output_grid
            n_shapes[i, j] = p.n_shapes
    depths = np.array([len(t.program) for t in tasks], dtype=np.int32)
    suites = [json.dumps(_program_to_list(t.program)) for t in tasks]

    has_vqo = any(t.valid_query_outputs is not None for t in tasks)
    if has_vqo:
        # Variable-length per task (|Y_q| differs). Store as JSON-encoded
        # list of 2D int arrays, one per task. Same vlen-string pattern as
        # transformation_suites. Empty list = {} valid outputs (should not
        # normally occur; caller drops such tasks).
        vqo_blobs = [
            json.dumps([g.tolist() for g in (t.valid_query_outputs or [])])
            for t in tasks
        ]

    with h5py.File(path, "w") as f:
        # Grids are small int8 arrays. Leaving these datasets contiguous avoids
        # HDF5's auto-chunked spatial blocks, which are very slow for the
        # benchmark's common row-wise and batch-wise access patterns.
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("grid_sizes", data=np.tile([h, w], (n_tasks, 1)).astype(np.int32))
        f.create_dataset("n_shapes", data=n_shapes)
        f.create_dataset("depths", data=depths)
        if has_n_support:
            f.create_dataset(
                "n_support",
                data=np.array(n_support_values, dtype=np.int32),
            )
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=h5py.string_dtype())
        if has_vqo:
            f.create_dataset("valid_query_outputs",
                             data=np.array(vqo_blobs, dtype=object),
                             dtype=h5py.string_dtype())

    # Sidecar metadata: dataset stats + empirical primitive frequency across
    # all program steps. Optional `extra_info` (e.g. dominant programs from
    # the sampler pool) is merged in. Same basename, `.info.json` suffix.
    counts = Counter(s.primitive for t in tasks for s in t.program)
    depth_counts = Counter(int(d) for d in depths)
    info = {
        "level": tasks[0].level,
        "n_tasks": n_tasks,
        "n_pairs": pair_count,
        "grid": [int(h), int(w)],
        "depth_counts": {str(d): c for d, c in sorted(depth_counts.items())},
        "depth_probs_empirical": {
            str(d): c / n_tasks for d, c in sorted(depth_counts.items())
        },
        "primitive_counts": dict(counts.most_common()),
        "h5_layout": {
            "inputs_outputs": "contiguous_uncompressed",
        },
    }
    if has_n_support:
        support_counts = Counter(int(k) for k in n_support_values)
        info.update({
            "support_counts": {
                str(k): c for k, c in sorted(support_counts.items())
            },
            "support_probs_empirical": {
                str(k): c / n_tasks for k, c in sorted(support_counts.items())
            },
        })
    if extra_info:
        info.update(extra_info)
    Path(path).with_suffix(".info.json").write_text(json.dumps(info, indent=2))


def render_grid(grid: np.ndarray) -> str:
    """ASCII render: '.' for 0, digit for non-zero colors."""
    return "\n".join(
        "".join("." if int(c) == 0 else str(int(c)) for c in row) for row in grid
    )


def show(task: Task) -> None:
    """Pretty-print a task to stdout."""
    print(f"Task(level={task.level}, depth={len(task.program)})")
    rendered = " → ".join(_render_step(s) for s in task.program) if task.program else "<empty>"
    print(f"  program: {rendered}")
    for i, p in enumerate(task.pairs):
        print(f"  --- pair {i} ---")
        print("  input:")
        for row in render_grid(p.input_grid).split("\n"):
            print(f"    {row}")
        print("  output:")
        for row in render_grid(p.output_grid).split("\n"):
            print(f"    {row}")
