"""Tests for pair-level uniqueness (dedup) added to generate.py.

Covers:
    - `_pair_hash` determinism + sensitivity
    - `_accept_unique` intra-task and cross-task collision behaviour
    - `_hashes_of_file` for both JSON and HDF5 formats
    - `load_excluded_pairs` directory expand, .info.json filter, self-skip
    - End-to-end: chained generate_filter_dataset runs are pair-disjoint
    - End-to-end via CLI main(): `--exclude DIR` enforces cross-file uniqueness
"""

from __future__ import annotations

import sys; sys.path.insert(0, ".")

import numpy as np

from babytheorist.generation.cli import main
from babytheorist.generation.config import LevelConfig, TaskConfig
from babytheorist.generation.filter_dataset import generate_filter_dataset
from babytheorist.generation.rejection import (
    _accept_unique, _hashes_of_file, _pair_hash, load_excluded_pairs,
)
from babytheorist.core import Task, TaskPair
from babytheorist.review import save


# ---------------------------------------------------------------------------
# Hash function

def test_pair_hash_deterministic():
    a = np.array([[1, 2], [3, 0]], dtype=np.int8)
    b = np.array([[0, 1], [2, 3]], dtype=np.int8)
    assert _pair_hash(a, b) == _pair_hash(a, b)


def test_pair_hash_input_output_swap_differs():
    a = np.array([[1, 2], [3, 0]], dtype=np.int8)
    b = np.array([[0, 1], [2, 3]], dtype=np.int8)
    assert _pair_hash(a, b) != _pair_hash(b, a)


def test_pair_hash_dtype_invariant():
    """int8 vs int64 of the same logical grid → same hash."""
    a = np.array([[1, 2], [3, 0]], dtype=np.int8)
    b = np.array([[0, 1], [2, 3]], dtype=np.int8)
    assert _pair_hash(a.astype(np.int64), b.astype(np.int64)) == _pair_hash(a, b)


def test_pair_hash_shape_fold_prevents_byte_alias():
    """(3,4) vs (4,3) zeros must not alias."""
    z34 = np.zeros((3, 4), dtype=np.int8)
    z43 = np.zeros((4, 3), dtype=np.int8)
    assert _pair_hash(z34, z34) != _pair_hash(z43, z43)


# ---------------------------------------------------------------------------
# Accept unique

def _make_task(pairs_data):
    return Task(
        pairs=[TaskPair(input_grid=i, output_grid=o, n_shapes=1) for i, o in pairs_data],
        program=[],
        level=1,
    )


def test_accept_unique_first_then_reject_same():
    a = np.array([[1, 2]], dtype=np.int8)
    b = np.array([[3, 4]], dtype=np.int8)
    seen: set[bytes] = set()
    t = _make_task([(a, b)])
    assert _accept_unique(t, seen) is True
    assert len(seen) == 1
    assert _accept_unique(t, seen) is False  # identical task — rejected
    assert len(seen) == 1


def test_accept_unique_intra_task_duplicate_rejected():
    a = np.array([[1, 2]], dtype=np.int8)
    b = np.array([[3, 4]], dtype=np.int8)
    t = _make_task([(a, b), (a, b)])
    seen: set[bytes] = set()
    assert _accept_unique(t, seen) is False  # same pair twice in one task
    assert seen == set()  # not mutated on reject


def test_accept_unique_partial_overlap_rejected():
    a = np.array([[1, 2]], dtype=np.int8); b = np.array([[3, 4]], dtype=np.int8)
    c = np.array([[5, 6]], dtype=np.int8); d = np.array([[7, 8]], dtype=np.int8)
    seen = {_pair_hash(a, b)}
    t = _make_task([(a, b), (c, d)])  # one new pair, one already seen
    assert _accept_unique(t, seen) is False
    assert seen == {_pair_hash(a, b)}  # unchanged


# ---------------------------------------------------------------------------
# File hashing

def _make_dataset(tmp_path, level=1, n_tasks=3, mode="train", seed=0, name="d"):
    lv = LevelConfig(level=level, primitives=["rot90", "flip_h"], fixed_depth=1)
    tasks = generate_filter_dataset(
        lv, TaskConfig(mode=mode, n_shapes_range=(2, 2), world_size=10),
        n_tasks=n_tasks, seed=seed,
    )
    return tasks


def test_hashes_of_file_json_roundtrip(tmp_path):
    tasks = _make_dataset(tmp_path, n_tasks=3, mode="train", seed=1)
    p = tmp_path / "data.json"
    save(tasks, p)
    hashes = _hashes_of_file(p)
    expected = {
        _pair_hash(pair.input_grid, pair.output_grid)
        for t in tasks for pair in t.pairs
    }
    assert hashes == expected
    assert len(hashes) == sum(len(t.pairs) for t in tasks)


def test_hashes_of_file_hdf5_roundtrip(tmp_path):
    tasks = _make_dataset(tmp_path, n_tasks=3, mode="train", seed=2)
    p = tmp_path / "data.h5"
    save(tasks, p)
    hashes = _hashes_of_file(p)
    expected = {
        _pair_hash(pair.input_grid, pair.output_grid)
        for t in tasks for pair in t.pairs
    }
    assert hashes == expected


# ---------------------------------------------------------------------------
# load_excluded_pairs

def test_load_excluded_pairs_directory_expand_skips_info_json(tmp_path):
    tasks_a = _make_dataset(tmp_path, n_tasks=2, seed=10)
    tasks_b = _make_dataset(tmp_path, n_tasks=2, seed=11)
    save(tasks_a, tmp_path / "a.h5")  # writes a.info.json sidecar too
    save(tasks_b, tmp_path / "b.h5")  # writes b.info.json sidecar too
    seen = load_excluded_pairs([tmp_path])  # directory expand
    expected = {
        _pair_hash(pair.input_grid, pair.output_grid)
        for ts in (tasks_a, tasks_b) for t in ts for pair in t.pairs
    }
    assert seen == expected


def test_load_excluded_pairs_self_skip(tmp_path):
    tasks_a = _make_dataset(tmp_path, n_tasks=2, seed=20)
    tasks_b = _make_dataset(tmp_path, n_tasks=2, seed=21)
    save(tasks_a, tmp_path / "a.h5")
    save(tasks_b, tmp_path / "b.h5")
    seen = load_excluded_pairs([tmp_path], out_path=tmp_path / "a.h5")
    expected_b = {
        _pair_hash(pair.input_grid, pair.output_grid)
        for t in tasks_b for pair in t.pairs
    }
    assert seen == expected_b  # only b's pairs; a (==out_path) skipped


def test_load_excluded_pairs_explicit_files(tmp_path):
    tasks = _make_dataset(tmp_path, n_tasks=2, seed=30)
    save(tasks, tmp_path / "x.h5")
    seen = load_excluded_pairs([tmp_path / "x.h5"])  # explicit file
    expected = {
        _pair_hash(pair.input_grid, pair.output_grid)
        for t in tasks for pair in t.pairs
    }
    assert seen == expected


def test_load_excluded_pairs_empty():
    assert load_excluded_pairs([]) == set()


# ---------------------------------------------------------------------------
# End-to-end: chained generation is pair-disjoint

def test_generate_with_seen_pair_hashes_dedups_cross_run(tmp_path):
    lv = LevelConfig(level=1, primitives=["rot90", "flip_h"], fixed_depth=1)
    cfg = TaskConfig(mode="train", n_shapes_range=(2, 2), world_size=10)
    seen: set[bytes] = set()
    first = generate_filter_dataset(lv, cfg, n_tasks=4, seed=100, seen_pair_hashes=seen)
    second = generate_filter_dataset(lv, cfg, n_tasks=4, seed=101, seen_pair_hashes=seen)

    first_hashes = {_pair_hash(p.input_grid, p.output_grid) for t in first for p in t.pairs}
    second_hashes = {_pair_hash(p.input_grid, p.output_grid) for t in second for p in t.pairs}
    assert first_hashes & second_hashes == set()  # zero collisions across runs


def test_main_exclude_directory_enforces_cross_file_uniqueness(tmp_path):
    """Round-trip via the CLI: --exclude DIR must prevent any pair overlap
    between the two output files. Uses a wider search space than the unit
    tests so the second run can still produce some non-colliding tasks."""
    out_a = tmp_path / "a.h5"
    out_b = tmp_path / "b.h5"
    common = [
        "--level", "2", "--n", "3", "--mode", "train",
        "--world-size", "12", "--n-shapes", "2",
        "--shape-min", "4", "--shape-max", "7",
        "--benchmark-mode", "single_easy", "--backend", "filter",
        "--pool-seed", "200",
    ]
    assert main(common + ["--seed", "200", "--out", str(out_a)]) == 0
    assert main(common + ["--seed", "201", "--exclude", str(tmp_path), "--out", str(out_b)]) == 0

    a_hashes = _hashes_of_file(out_a)
    b_hashes = _hashes_of_file(out_b)
    assert a_hashes & b_hashes == set()  # zero overlap
