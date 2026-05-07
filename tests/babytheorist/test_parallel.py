"""Parallel generation regression tests."""

from __future__ import annotations

import numpy as np
import pytest
import time

import babytheorist.generation.filter_dataset as filter_dataset_module
from babytheorist.generation.config import LevelConfig, TaskConfig
from babytheorist.generation.filter_dataset import (
    _FilterAttemptTimedOut, _generate_filter_attempt_with_timeout,
    generate_filter_dataset,
)
from babytheorist.generation.rejection import generate_dataset


def _task_signature(tasks):
    """Bytes blob covering every pair grid + the program primitive sequence,
    so two task lists are equal as a multiset of (program, grids)."""
    parts = []
    for t in tasks:
        parts.append(",".join(s.primitive for s in t.program).encode())
        for p in t.pairs:
            parts.append(np.ascontiguousarray(p.input_grid, dtype=np.int8).tobytes())
            parts.append(np.ascontiguousarray(p.output_grid, dtype=np.int8).tobytes())
    return b"|".join(parts)


def test_parallel_generation_is_reproducible():
    """Same `(seed, n_workers)` across two runs must give the same task set."""
    lv = LevelConfig(
        level=2, primitives=["rot90", "flip_h", "next_color"], fixed_depth=2,
    )
    cfg = TaskConfig(mode="train", n_shapes_range=(2, 2))
    a = generate_dataset(lv, cfg, n_tasks=20, seed=42, n_workers=4)
    b = generate_dataset(lv, cfg, n_tasks=20, seed=42, n_workers=4)
    assert _task_signature(a) == _task_signature(b)


def test_parallel_generation_is_workers_invariant():
    """Different `n_workers` with the same seed must give the same task set."""
    lv = LevelConfig(
        level=2, primitives=["rot90", "flip_h", "next_color"], fixed_depth=2,
    )
    cfg = TaskConfig(mode="train", n_shapes_range=(2, 2))
    a = generate_dataset(lv, cfg, n_tasks=40, seed=42, n_workers=2)
    b = generate_dataset(lv, cfg, n_tasks=40, seed=42, n_workers=8)
    assert _task_signature(a) == _task_signature(b)


def test_generate_dataset_rejects_zero_workers():
    lv = LevelConfig(level=1, primitives=["rot90", "flip_h"], fixed_depth=1)
    with pytest.raises(ValueError, match="n_workers"):
        generate_dataset(lv, n_tasks=2, seed=0, n_workers=0)


def test_parallel_filter_generation_is_reproducible():
    lv = LevelConfig(
        level=1,
        primitives=["rot90", "flip_h", "translate_up", "translate_down"],
        selectors=("ALL", "by_index"),
        fixed_depth=2,
        palette=(1, 2, 3, 4, 5),
    )
    cfg = TaskConfig(
        mode="train", n_shapes_range=(2, 2), world_size=24,
        shape_size_range=(6, 12),
    )
    a = generate_filter_dataset(lv, cfg, n_tasks=6, seed=42, n_workers=2)
    b = generate_filter_dataset(lv, cfg, n_tasks=6, seed=42, n_workers=2)
    assert _task_signature(a) == _task_signature(b)


def test_parallel_filter_generation_is_workers_invariant():
    lv = LevelConfig(
        level=1,
        primitives=["rot90", "flip_h", "translate_up", "translate_down"],
        selectors=("ALL", "by_index"),
        fixed_depth=2,
        palette=(1, 2, 3, 4, 5),
    )
    cfg = TaskConfig(
        mode="train", n_shapes_range=(2, 2), world_size=24,
        shape_size_range=(6, 12),
    )
    a = generate_filter_dataset(lv, cfg, n_tasks=8, seed=42, n_workers=1)
    b = generate_filter_dataset(lv, cfg, n_tasks=8, seed=42, n_workers=4)
    assert _task_signature(a) == _task_signature(b)


def test_unordered_filter_attempt_scheduler_produces_tasks():
    lv = LevelConfig(
        level=1,
        primitives=["rot90", "flip_h", "translate_up", "translate_down"],
        selectors=("ALL", "by_index"),
        fixed_depth=2,
        palette=(1, 2, 3, 4, 5),
    )
    cfg = TaskConfig(
        mode="train", n_shapes_range=(2, 2), world_size=24,
        shape_size_range=(6, 12),
    )
    tasks = generate_filter_dataset(
        lv, cfg, n_tasks=6, seed=42, n_workers=4,
        attempt_scheduler="unordered",
    )
    assert len(tasks) == 6


def test_generate_filter_dataset_rejects_zero_workers():
    lv = LevelConfig(level=1, primitives=["rot90", "flip_h"], fixed_depth=1)
    with pytest.raises(ValueError, match="n_workers"):
        generate_filter_dataset(lv, n_tasks=2, seed=0, n_workers=0)


def test_generate_filter_dataset_rejects_unknown_attempt_scheduler():
    lv = LevelConfig(level=1, primitives=["rot90", "flip_h"], fixed_depth=1)
    with pytest.raises(ValueError, match="attempt_scheduler"):
        generate_filter_dataset(
            lv, n_tasks=2, seed=0, n_workers=2,
            attempt_scheduler="not-a-scheduler",
        )


def test_generate_filter_dataset_rejects_invalid_attempt_timeout():
    lv = LevelConfig(level=1, primitives=["rot90", "flip_h"], fixed_depth=1)
    with pytest.raises(ValueError, match="attempt_timeout_sec"):
        generate_filter_dataset(
            lv, n_tasks=2, seed=0, n_workers=1,
            attempt_timeout_sec=0,
        )


def test_filter_attempt_timeout_returns_marker(monkeypatch):
    def slow_attempt(*args, **kwargs):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(
        filter_dataset_module, "_generate_filter_attempt", slow_attempt,
    )

    result = _generate_filter_attempt_with_timeout(
        None, None, 0, 0.0, attempt_timeout_sec=0.01,
    )

    assert isinstance(result, _FilterAttemptTimedOut)
