"""HDF5 export path regression tests."""

from __future__ import annotations

import json
import sys

import h5py
import numpy as np
import pytest

from babytheorist.generation import LevelConfig, TaskConfig, generate_dataset
from babytheorist.core import Task, TaskPair
from babytheorist.review import _save_h5, load, save


def _tiny_dataset(tmp_path):
    lv = LevelConfig(
        level=2, primitives=["rot90", "flip_h", "next_color"], fixed_depth=2,
    )
    tasks = generate_dataset(lv, TaskConfig(mode="eval", n_shapes_range=(2, 3)),
                             n_tasks=4, seed=0)
    return tasks


def test_save_h5_schema_and_n_shapes(tmp_path):
    tasks = _tiny_dataset(tmp_path)  # 4 eval tasks, adaptive support + query
    path = tmp_path / "out.h5"
    save(tasks, path)

    with h5py.File(path, "r") as f:
        # Eval mode → valid_query_outputs dataset is present.
        assert set(f.keys()) == {
            "inputs", "outputs", "grid_sizes", "n_shapes", "depths",
            "n_support", "transformation_suites", "valid_query_outputs",
        }
        n_tasks = len(tasks)
        n_pairs = max(t.n_support for t in tasks) + 1
        assert f["inputs"].shape == (n_tasks, n_pairs, 20, 20)
        assert f["inputs"].dtype == np.int8
        assert f["inputs"].chunks is None
        assert f["outputs"].chunks is None
        assert f["inputs"].compression is None
        assert f["outputs"].compression is None
        assert f["grid_sizes"].shape == (n_tasks, 2)
        assert f["n_shapes"].shape == (n_tasks, n_pairs)
        assert f["depths"].shape == (n_tasks,)
        assert np.issubdtype(f["depths"].dtype, np.integer)
        assert f["n_support"].shape == (n_tasks,)
        assert f["n_support"][:].tolist() == [t.n_support for t in tasks]
        assert f["transformation_suites"].shape == (n_tasks,)
        # Per-pair counts round-trip, all sampled from (2, 3).
        exported = f["n_shapes"][:].tolist()
        for i, t in enumerate(tasks):
            expected = [p.n_shapes for p in t.pairs]
            assert exported[i][:len(expected)] == expected
        assert all(2 <= c <= 3 for t in tasks for p in t.pairs for c in [p.n_shapes])
        # transformation_suites decodes to the same JSON program (one per task).
        suite0 = f["transformation_suites"].asstr()[0]
        assert json.loads(suite0)  # parses as list of Step dicts
        decoded_depths = [
            len(json.loads(s)) for s in f["transformation_suites"].asstr()
        ]
        assert f["depths"][:].tolist() == decoded_depths
        # Eval's valid_query_outputs decodes to a non-empty list of grids.
        # With selectors=("ALL",) alternatives collapse to the canonical
        # step only, so |Y_q|=1 for every task.
        assert f["valid_query_outputs"].shape == (n_tasks,)
        for s in f["valid_query_outputs"].asstr():
            grids = json.loads(s)
            assert len(grids) == 1
            assert np.array(grids[0]).shape == (20, 20)
    info = json.loads(path.with_suffix(".info.json").read_text())
    assert info["depth_counts"] == {"2": n_tasks}
    assert info["depth_probs_empirical"] == {"2": 1.0}
    assert info["h5_layout"] == {"inputs_outputs": "contiguous_uncompressed"}


def test_load_h5_is_rejected(tmp_path):
    tasks = _tiny_dataset(tmp_path)
    path = tmp_path / "out.h5"
    save(tasks, path)
    with pytest.raises(NotImplementedError, match="write-only"):
        load(path)


def test_save_h5_rejects_missing_n_shapes(tmp_path):
    # A hand-built task (e.g. loaded from legacy JSON) has n_shapes=0.
    grid = np.zeros((20, 20), dtype=np.int64)
    task = Task(
        level=1,
        program=[],
        pairs=[TaskPair(input_grid=grid, output_grid=grid)],
    )
    with pytest.raises(ValueError, match="n_shapes"):
        save([task], tmp_path / "x.h5")


def test_save_h5_rejects_mismatched_output_shape(tmp_path):
    inp = np.zeros((20, 20), dtype=np.int64)
    out = np.zeros((15, 15), dtype=np.int64)
    task = Task(
        level=1,
        program=[],
        pairs=[TaskPair(input_grid=inp, output_grid=out, n_shapes=2)],
    )
    with pytest.raises(ValueError, match="uniform"):
        save([task], tmp_path / "x.h5")


def test_save_h5_rejects_empty_tasks(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        save([], tmp_path / "x.h5")


def test_save_h5_rejects_ragged_n_pairs(tmp_path):
    inp = np.zeros((20, 20), dtype=np.int64)
    pair = TaskPair(input_grid=inp, output_grid=inp, n_shapes=1)
    tasks = [
        Task(level=1, program=[], pairs=[pair]),
        Task(level=1, program=[], pairs=[pair, pair]),
    ]
    with pytest.raises(ValueError, match="uniform n_pairs"):
        save(tasks, tmp_path / "x.h5")


def test_save_h5_writes_ragged_eval_with_n_support(tmp_path):
    g0 = np.zeros((4, 4), dtype=np.int64)
    g1 = np.ones((4, 4), dtype=np.int64)
    g2 = np.full((4, 4), 2, dtype=np.int64)
    t0 = Task(
        level=1,
        program=[],
        pairs=[
            TaskPair(input_grid=g0, output_grid=g1, n_shapes=1),
            TaskPair(input_grid=g1, output_grid=g2, n_shapes=1),
        ],
        valid_query_outputs=[g2],
        n_support=1,
    )
    t1 = Task(
        level=1,
        program=[],
        pairs=[
            TaskPair(input_grid=g0, output_grid=g1, n_shapes=1),
            TaskPair(input_grid=g1, output_grid=g2, n_shapes=1),
            TaskPair(input_grid=g2, output_grid=g0, n_shapes=1),
        ],
        valid_query_outputs=[g0],
        n_support=2,
    )
    path = tmp_path / "ragged.h5"
    save([t0, t1], path)

    with h5py.File(path, "r") as f:
        assert f["inputs"].shape == (2, 3, 4, 4)
        assert f["n_support"][:].tolist() == [1, 2]
        assert np.array_equal(f["outputs"][0, 1], g2)
        assert np.array_equal(f["outputs"][1, 2], g0)
        assert f["n_shapes"][0, 2] == 0  # padding slot


def test_save_h5_missing_h5py_raises_friendly(tmp_path, monkeypatch):
    """When h5py is absent, `_save_h5` should raise a clear ImportError."""
    monkeypatch.setitem(sys.modules, "h5py", None)
    inp = np.zeros((20, 20), dtype=np.int64)
    task = Task(
        level=1, program=[],
        pairs=[TaskPair(input_grid=inp, output_grid=inp, n_shapes=1)],
    )
    with pytest.raises(ImportError, match="requires h5py"):
        _save_h5([task], tmp_path / "x.h5")
