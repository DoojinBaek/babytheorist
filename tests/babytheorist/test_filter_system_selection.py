from __future__ import annotations

import json

import numpy as np

import babytheorist.generation.filter_pairs as filter_pairs_module
from babytheorist.generation.cli import main
from babytheorist.generation.config import (
    LevelConfig, TaskConfig, _PAIR_FAILURE_IDENTITY, _PAIR_FAILURE_SCENE_NONE,
)
from babytheorist.generation.filter_dataset import (
    FilterGenerationStats, generate_filter_dataset,
)
from babytheorist.generation.filter_pairs import (
    _sample_filter_pair_with_reason,
)
from babytheorist.core import Selector, Step
from babytheorist.engine import ENGINE_FAILURE_SELECTOR_EMPTY
from babytheorist.filter.constraint import InputConstraint
from babytheorist.shapes import ShapeFactory


def test_generate_filter_dataset_uses_production_filter():
    level_cfg = LevelConfig(
        level=1,
        primitives=["rot90", "flip_h", "translate_up"],
        selectors=("ALL", "by_index"),
        fixed_depth=1,
        palette=(1, 2, 3, 4, 5),
        use_dominant=False,
    )
    task_cfg = TaskConfig(
        mode="train",
        n_shapes_range=(2, 2),
        world_size=24,
        shape_size_range=(8, 12),
    )

    stats = FilterGenerationStats()
    tasks = generate_filter_dataset(
        level_cfg,
        task_cfg,
        n_tasks=2,
        seed=0,
        stats=stats,
    )

    assert len(tasks) == 2
    assert all(len(task.program) == 1 for task in tasks)
    assert stats.produced_tasks == 2
    assert stats.sampler_calls >= 2
    assert stats.sampler_steps >= 2
    assert sum(stats.grounding_signature_attempts.values()) >= 2
    assert sum(stats.grounding_signature_successes.values()) == 2


def test_generate_filter_dataset_uses_production_filter_in_parallel():
    level_cfg = LevelConfig(
        level=1,
        primitives=["rot90", "flip_h", "translate_up"],
        selectors=("ALL", "by_index"),
        fixed_depth=1,
        palette=(1, 2, 3, 4, 5),
        use_dominant=False,
    )
    task_cfg = TaskConfig(
        mode="train",
        n_shapes_range=(2, 2),
        world_size=24,
        shape_size_range=(8, 12),
    )

    tasks = generate_filter_dataset(
        level_cfg,
        task_cfg,
        n_tasks=2,
        seed=0,
        n_workers=2,
    )

    assert len(tasks) == 2
    assert all(len(task.program) == 1 for task in tasks)


def test_filter_pair_failure_reason_identity_pair():
    level_cfg = LevelConfig(
        level=1,
        primitives=["rot90"],
        selectors=("ALL",),
        fixed_depth=1,
        palette=(1, 2, 3),
        use_dominant=False,
    )
    task_cfg = TaskConfig(
        mode="train",
        n_shapes_range=(1, 1),
        world_size=24,
        shape_size_range=(3, 6),
        max_pair_retries=20,
    )

    pair, reason = _sample_filter_pair_with_reason(
        InputConstraint.empty(1),
        [],
        ShapeFactory(size_range=(3, 6)),
        level_cfg,
        task_cfg,
        np.random.default_rng(0),
        uses_by_index=False,
        gap=2,
    )

    assert pair is None
    assert reason == _PAIR_FAILURE_IDENTITY


def test_filter_pair_failure_reason_engine_selector_empty():
    level_cfg = LevelConfig(
        level=1,
        primitives=["translate_up"],
        selectors=("by_color",),
        fixed_depth=1,
        palette=(1, 2, 3),
        use_dominant=False,
    )
    task_cfg = TaskConfig(
        mode="train",
        n_shapes_range=(1, 1),
        world_size=24,
        shape_size_range=(3, 6),
        max_pair_retries=20,
    )

    pair, reason = _sample_filter_pair_with_reason(
        InputConstraint.empty(1),
        [Step(Selector("by_color", 9), "translate_up")],
        ShapeFactory(size_range=(3, 6)),
        level_cfg,
        task_cfg,
        np.random.default_rng(0),
        uses_by_index=False,
        gap=2,
    )

    assert pair is None
    assert reason == ENGINE_FAILURE_SELECTOR_EMPTY


def test_filter_pair_stops_on_scene_none(monkeypatch):
    calls = {"sample_scene": 0}

    class FakeSceneSampler:
        @staticmethod
        def sample_scene(*args, **kwargs):
            calls["sample_scene"] += 1
            return None

    monkeypatch.setattr(
        filter_pairs_module, "_filter_scene_sampler_module",
        lambda: FakeSceneSampler,
    )

    level_cfg = LevelConfig(
        level=1,
        primitives=["rot90"],
        selectors=("ALL",),
        fixed_depth=1,
        palette=(1, 2, 3),
        use_dominant=False,
    )
    task_cfg = TaskConfig(mode="train", world_size=5, max_pair_retries=1)

    pair, reason = _sample_filter_pair_with_reason(
        InputConstraint.empty(1),
        [Step(Selector("ALL"), "rot90")],
        ShapeFactory(size_range=(3, 6)),
        level_cfg,
        task_cfg,
        np.random.default_rng(0),
        uses_by_index=False,
        gap=2,
        execution_retries=2,
    )

    assert pair is None
    assert reason == _PAIR_FAILURE_SCENE_NONE
    assert calls["sample_scene"] == 1


def test_filter_pair_reports_scene_none_without_repeating_exhausted_scene(monkeypatch):
    calls = {"sample_scene": 0}

    class FakeSceneSampler:
        @staticmethod
        def sample_scene(*args, **kwargs):
            calls["sample_scene"] += 1
            return None

    monkeypatch.setattr(
        filter_pairs_module, "_filter_scene_sampler_module",
        lambda: FakeSceneSampler,
    )

    level_cfg = LevelConfig(
        level=1,
        primitives=["rot90"],
        selectors=("ALL",),
        fixed_depth=1,
        palette=(1, 2, 3),
        use_dominant=False,
    )
    task_cfg = TaskConfig(mode="train", world_size=5, max_pair_retries=1)

    pair, reason = _sample_filter_pair_with_reason(
        InputConstraint.empty(1),
        [Step(Selector("ALL"), "rot90")],
        ShapeFactory(size_range=(3, 6)),
        level_cfg,
        task_cfg,
        np.random.default_rng(0),
        uses_by_index=False,
        gap=2,
        execution_retries=3,
    )

    assert pair is None
    assert reason == _PAIR_FAILURE_SCENE_NONE
    assert calls["sample_scene"] == 1


def test_cli_writes_filter_metadata(tmp_path):
    out = tmp_path / "filter.json"
    rc = main([
        "--level", "1",
        "--n", "1",
        "--backend", "filter",
        "--no-dominant",
        "--primitives", "rot90,flip_h,translate_up",
        "--selectors", "ALL,by_index",
        "--n-shapes", "2",
        "--shape-min", "8",
        "--shape-max", "12",
        "--world-size", "24",
        "--out", str(out),
    ])

    assert rc == 0
    info = json.loads(out.read_text())["info"]
    assert info["generator_backend"] == "filter"
    assert info["filter_system"] == "filter"
    assert "attempt_scheduler" not in info
    assert "attempt_timeout_sec" not in info
    assert info["filter_rules_version"] == "filter.rules.v3"
    assert info["filter_grounding_version"] == "filter.abstract_grounding.v2"
    assert info["filter_generation_stats"]["requested_tasks"] == 1
    assert info["filter_generation_stats"]["sampler_calls"] >= 1
    assert "tasks_per_sec" in info["filter_generation_stats"]
    assert "sampler_elapsed_sec" in info["filter_generation_stats"]
    assert "pair_elapsed_sec" in info["filter_generation_stats"]
    assert info["filter_generation_stats"]["pair_execution_retries"] == 16
    assert "pair_failure_reasons" in info["filter_generation_stats"]
    assert "grounding_signature_stats" in info["filter_generation_stats"]
    assert isinstance(info["filter_reject_stats"], dict)


def test_cli_writes_unordered_attempt_scheduler_metadata(tmp_path):
    out = tmp_path / "filter_unordered.json"
    rc = main([
        "--level", "1",
        "--n", "1",
        "--backend", "filter",
        "--attempt-scheduler", "unordered",
        "--no-dominant",
        "--primitives", "rot90,flip_h,translate_up",
        "--selectors", "ALL,by_index",
        "--n-shapes", "2",
        "--shape-min", "8",
        "--shape-max", "12",
        "--world-size", "24",
        "--out", str(out),
    ])

    assert rc == 0
    info = json.loads(out.read_text())["info"]
    assert info["attempt_scheduler"] == "unordered"


def test_cli_writes_attempt_timeout_metadata(tmp_path):
    out = tmp_path / "filter_timeout.json"
    rc = main([
        "--level", "1",
        "--n", "1",
        "--backend", "filter",
        "--attempt-timeout-sec", "10",
        "--no-dominant",
        "--primitives", "rot90,flip_h,translate_up",
        "--selectors", "ALL,by_index",
        "--n-shapes", "2",
        "--shape-min", "8",
        "--shape-max", "12",
        "--world-size", "24",
        "--out", str(out),
    ])

    assert rc == 0
    info = json.loads(out.read_text())["info"]
    assert info["attempt_timeout_sec"] == 10.0
    assert info["filter_generation_stats"]["attempt_timeouts"] == 0
