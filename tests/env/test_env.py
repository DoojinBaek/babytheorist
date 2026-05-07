"""Tests for env/env.py. See env_plan.md §9 for the 23-case table."""

from __future__ import annotations

import multiprocessing as mp
import os
import json
import pickle
import shutil
from contextlib import contextmanager

import numpy as np
import pytest

from env import (
    EvalExample,
    GenerationConfig,
    BabyTheoristEnv,
    ScoreReport,
    TrainBatch,
    TrainExample,
)


# ---- 1–2: dataset construction -------------------------------------------

def test_train_returns_train_examples(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(shapes["LEVEL"],))
    ds = env.train(level=1, n=shapes["N_TRAIN"])
    assert len(ds) == shapes["N_TRAIN"]
    ex = ds[0]
    assert isinstance(ex, TrainExample)
    assert ex.x.shape == (shapes["H"], shapes["W"]) and ex.x.dtype == np.int8
    assert ex.y.shape == (shapes["H"], shapes["W"]) and ex.y.dtype == np.int8
    assert ex.level == 1 and ex.task_id == 0


def test_eval_returns_eval_examples(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(shapes["LEVEL"],))
    ds = env.eval(level=1)
    assert len(ds) == shapes["N_EVAL"]
    ex = ds[0]
    assert isinstance(ex, EvalExample)
    # Current BabyTheorist gen writes P=2 → K=1 support pair.
    assert ex.support_x.shape == (1, shapes["H"], shapes["W"])
    assert ex.support_y.shape == (1, shapes["H"], shapes["W"])
    assert ex.query_x.shape == (shapes["H"], shapes["W"])
    assert ex.level == 1 and ex.task_id == 0


# ---- 3: label isolation on EvalExample -----------------------------------

def test_eval_example_has_no_label_fields():
    fields = {f.name for f in EvalExample.__dataclass_fields__.values()}
    assert "y_q" not in fields and "Y_q" not in fields and "program" not in fields


# ---- 4–5: scoring end-to-end ---------------------------------------------

def test_oracle_scores_perfectly(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = [(tid, env.answers(1, tid).y_q) for tid in range(shapes["N_EVAL"])]
    r = env.score_predictions(preds, level=1)
    assert isinstance(r, ScoreReport)
    assert r.n == shapes["N_EVAL"]
    assert r.correct == shapes["N_EVAL"]
    assert r.accuracy == 1.0


def test_identity_fn_runs(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    # support_y[0] has shape (H, W) — usable as a prediction.
    r = env.score(lambda sx, sy, qx: sy[0], level=1)
    assert 0.0 <= r.accuracy <= 1.0
    assert r.n == shapes["N_EVAL"]


# ---- 6–12: strict validation ---------------------------------------------

def _good_preds(env, level):
    return [(tid, env.answers(level, tid).y_q)
            for tid in range(len(env.eval(level=level)))]


def test_shape_mismatch_raises(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)
    preds[0] = (0, np.zeros((shapes["H"] + 1, shapes["W"]), dtype=np.int8))
    with pytest.raises(ValueError, match="task_id=0.*shape"):
        env.score_predictions(preds, level=1)


def test_duplicate_task_id_raises(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)
    preds[2] = (0, preds[0][1])  # duplicate tid=0
    with pytest.raises(ValueError, match="duplicate task_id=0"):
        env.score_predictions(preds, level=1)


def test_out_of_range_task_id_raises(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)
    preds[-1] = (shapes["N_EVAL"] + 99, preds[-1][1])
    with pytest.raises(ValueError, match="out-of-range"):
        env.score_predictions(preds, level=1)


def test_wrong_length_raises(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)[:-1]  # drop one
    with pytest.raises(ValueError, match="length"):
        env.score_predictions(preds, level=1)


def test_list_of_lists_coerced(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = [(tid, env.answers(1, tid).y_q.tolist())
             for tid in range(len(env.eval(level=1)))]
    r = env.score_predictions(preds, level=1)
    assert r.accuracy == 1.0


def test_float_dtype_raises(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)
    preds[0] = (0, np.zeros((shapes["H"], shapes["W"]), dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        env.score_predictions(preds, level=1)


def test_value_out_of_range_raises(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    preds = _good_preds(env, 1)
    bad = np.zeros((shapes["H"], shapes["W"]), dtype=np.int8)
    bad[0, 0] = 10
    preds[0] = (0, bad)
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        env.score_predictions(preds, level=1)


# ---- 13–15: train sizing + reproducibility -------------------------------

def test_train_n_zero(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    ds = env.train(level=1, n=0)
    assert len(ds) == 0
    assert list(ds) == []


def test_train_n_too_large(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    with pytest.raises(ValueError):
        env.train(level=1, n=shapes["N_TRAIN"] + 1)


def test_train_prefix_reproducible(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    a = [(ex.task_id, ex.x.tobytes()) for ex in env.train(level=1, n=3)]
    b = [(ex.task_id, ex.x.tobytes()) for ex in env.train(level=1, n=3)]
    assert a == b
    assert [tid for tid, _ in a] == [0, 1, 2]


def test_train_as_arrays_matches_row_examples(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    ds = env.train(level=1, n=5)

    x, y = ds.as_arrays()

    assert x.shape == (5, 4, 4)
    assert y.shape == (5, 4, 4)
    for i in range(len(ds)):
        ex = ds[i]
        assert np.array_equal(x[i], ex.x)
        assert np.array_equal(y[i], ex.y)


def test_train_iter_batches_supports_incremental_exposure(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    ds = env.train(level=1, n=6)

    batches = list(ds.iter_batches(batch_size=2, start=2, stop=6))

    assert all(isinstance(b, TrainBatch) for b in batches)
    assert [b.x.shape[0] for b in batches] == [2, 2]
    assert [b.task_id.tolist() for b in batches] == [[2, 3], [4, 5]]
    assert all(b.level == 1 for b in batches)
    for batch in batches:
        for offset, tid in enumerate(batch.task_id):
            ex = ds[int(tid)]
            assert np.array_equal(batch.x[offset], ex.x)
            assert np.array_equal(batch.y[offset], ex.y)


def test_train_fast_paths_validate_grid_values(tiny_env_dir):
    import h5py

    with h5py.File(tiny_env_dir / "lv1_train.h5", "r+") as h:
        h["inputs"][3, 0, 0, 0] = 10

    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    ds = env.train(level=1, n=6)

    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        ds.as_arrays()
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        list(ds.iter_batches(batch_size=2))


# ---- 16: pickle round-trip -----------------------------------------------

def test_pickle_roundtrip(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    ds = env.eval(level=1)
    _ = ds[0]  # open in parent
    ds2 = pickle.loads(pickle.dumps(ds))
    ex = ds2[0]
    assert isinstance(ex, EvalExample)


# ---- 17: fork-inherited handle recovery (module globals, not pickled) ----

_DS = None
_PARENT_PID = None


def _init_ds(data_dir):
    global _DS, _PARENT_PID
    _DS = BabyTheoristEnv(data_dir=data_dir, levels=(1,)).eval(level=1)
    _ = _DS[0]  # open h5 in parent so child will inherit a stale handle
    _PARENT_PID = os.getpid()


def _worker(i):
    ex = _DS[i]
    # Return enough info to prove an actual reopen happened in the child:
    # _open_count was incremented (it started at 1 after parent's _DS[0]).
    return (os.getpid(), _DS._opener_pid, _DS._open_count, ex.query_x.shape)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(),
                    reason="fork start method unavailable")
def test_fork_inherited_handle(tiny_env_dir):
    _init_ds(str(tiny_env_dir))
    parent_open_count = _DS._open_count
    assert parent_open_count == 1, "parent should have opened the file exactly once"

    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        results = pool.map(_worker, list(range(len(_DS))))
    assert len(results) == len(_DS)
    for child_pid, opener_pid, open_count, _ in results:
        assert child_pid != _PARENT_PID, "child must be a forked process"
        assert opener_pid == child_pid, "opener_pid not updated in child"
        # Strong oracle: child must have executed h5py.File(...) itself,
        # not just overwritten _opener_pid. That increments _open_count.
        assert open_count > parent_open_count, (
            f"PID-guard reopen did not fire (open_count={open_count}, "
            f"parent had {parent_open_count})"
        )


# ---- 18–19: io_h5 schema assertions --------------------------------------

def test_empty_Y_q_raises(tiny_env_dir):
    import h5py
    path = tiny_env_dir / "lv1_eval.h5"
    with h5py.File(path, "r+") as h:
        ds = h["valid_query_outputs"]
        ds[0] = json.dumps([])
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    with pytest.raises(ValueError, match="empty"):
        env.answers(1, 0)


def test_Y_q_shape_mismatch_raises(tiny_env_dir, shapes):
    import h5py
    path = tiny_env_dir / "lv1_eval.h5"
    wrong = np.zeros((shapes["H"] + 1, shapes["W"]), dtype=np.int8).tolist()
    with h5py.File(path, "r+") as h:
        ds = h["valid_query_outputs"]
        ds[0] = json.dumps([wrong])
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    with pytest.raises(ValueError, match="Y_q"):
        env.answers(1, 0)


# ---- 20–22: metadata / answers API boundaries ----------------------------

def test_metadata_eval_raises(tiny_env_dir):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    with pytest.raises(ValueError, match="eval"):
        env.metadata(split="eval", level=1, task_id=0)


def test_metadata_train_returns_program_only(tiny_env_dir):
    from env import Metadata
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    m = env.metadata(split="train", level=1, task_id=0)
    assert isinstance(m, Metadata)
    assert isinstance(m.program, list)


def test_answers_eval_only(tiny_env_dir):
    from env import Answers
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    a = env.answers(1, 0)
    assert isinstance(a, Answers)
    assert isinstance(a.Y_q, tuple)
    assert a.Y_q[0].shape == a.y_q.shape


# ---- 23: canonicalization invariant --------------------------------------

def test_Y_q_canonicalization(tiny_env_dir, shapes):
    env = BabyTheoristEnv(tiny_env_dir, levels=(1,))
    for tid in range(shapes["N_EVAL"]):
        a = env.answers(1, tid)
        assert np.array_equal(a.Y_q[0], a.y_q)


# ---- 24–26: multi-support (K>1) future-proofing --------------------------

def test_multi_support_shapes(multi_support_env_dir, shapes):
    env = BabyTheoristEnv(multi_support_env_dir, levels=(1,))
    ex = env.eval(level=1)[0]
    # K=2: two support pairs, one query.
    assert ex.support_x.shape == (2, shapes["H"], shapes["W"])
    assert ex.support_y.shape == (2, shapes["H"], shapes["W"])
    assert ex.query_x.shape == (shapes["H"], shapes["W"])


def test_multi_support_oracle(multi_support_env_dir, shapes):
    env = BabyTheoristEnv(multi_support_env_dir, levels=(1,))
    preds = [(tid, env.answers(1, tid).y_q)
             for tid in range(shapes["N_EVAL"])]
    r = env.score_predictions(preds, level=1)
    assert r.accuracy == 1.0


def test_multi_support_score_signature(multi_support_env_dir):
    env = BabyTheoristEnv(multi_support_env_dir, levels=(1,))
    # model_fn receives (K=2, H, W) support stacks; return sy[-1] as prediction.
    r = env.score(lambda sx, sy, qx: sy[-1], level=1)
    assert 0.0 <= r.accuracy <= 1.0


# ---- 27–29: variable-K (per-row ragged support) --------------------------

def test_variable_support_per_row_shapes(variable_support_env_dir):
    env = BabyTheoristEnv(variable_support_env_dir, levels=(1,))
    ds = env.eval(level=1)
    expected_K = [1, 2, 1, 3, 2]  # from fixture
    for tid, K in enumerate(expected_K):
        ex = ds[tid]
        assert ex.support_x.shape[0] == K, f"tid={tid}: K={ex.support_x.shape[0]} != {K}"
        assert ex.support_y.shape[0] == K
        assert ex.support_x.shape[1:] == ex.query_x.shape


def test_variable_support_oracle(variable_support_env_dir, shapes):
    env = BabyTheoristEnv(variable_support_env_dir, levels=(1,))
    preds = [(tid, env.answers(1, tid).y_q)
             for tid in range(shapes["N_EVAL"])]
    r = env.score_predictions(preds, level=1)
    assert r.accuracy == 1.0


def test_variable_support_out_of_range_raises(variable_support_env_dir):
    """If n_support[i] is outside [1, P-1], reader must fail loud."""
    import h5py
    path = variable_support_env_dir / "lv1_eval.h5"
    with h5py.File(path, "r+") as h:
        h["n_support"][0] = 0  # invalid: no support
    env = BabyTheoristEnv(variable_support_env_dir, levels=(1,))
    with pytest.raises(ValueError, match="n_support"):
        env.answers(1, 0)


# ---- 30–32: additional file-level schema guards --------------------------

def test_n_support_wrong_dtype_raises(variable_support_env_dir):
    """n_support must be integer. float n_support can silently truncate."""
    import h5py
    path = variable_support_env_dir / "lv1_eval.h5"
    with h5py.File(path, "r+") as h:
        vals = h["n_support"][:].astype(np.float32)
        del h["n_support"]
        h.create_dataset("n_support", data=vals)
    with pytest.raises(ValueError, match="n_support.dtype"):
        BabyTheoristEnv(variable_support_env_dir, levels=(1,))


def test_n_support_wrong_length_raises(variable_support_env_dir):
    """n_support must have length N (one entry per eval row)."""
    import h5py
    path = variable_support_env_dir / "lv1_eval.h5"
    with h5py.File(path, "r+") as h:
        vals = h["n_support"][:-1]  # drop one entry
        del h["n_support"]
        h.create_dataset("n_support", data=vals)
    with pytest.raises(ValueError, match="n_support.shape"):
        BabyTheoristEnv(variable_support_env_dir, levels=(1,))


def test_train_P_not_1_raises(tiny_env_dir):
    """Train rows are single (x, y) pairs; P != 1 must fail at init."""
    import h5py
    path = tiny_env_dir / "lv1_train.h5"
    with h5py.File(path, "r") as f:
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]
        suites = [s for s in f["transformation_suites"].asstr()]
    # Duplicate the single pair along P to make P=2 (still uniform).
    inputs2 = np.concatenate([inputs, inputs], axis=1)
    outputs2 = np.concatenate([outputs, outputs], axis=1)
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs2)
        f.create_dataset("outputs", data=outputs2)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
    with pytest.raises(ValueError, match="train P="):
        BabyTheoristEnv(tiny_env_dir, levels=(1,))


# ---- 33: shape drift guard -----------------------------------------------

def test_inputs_outputs_shape_drift_raises(tiny_env_dir):
    """If inputs/outputs diverge on P, BabyTheoristEnv must fail loud at __init__
    (before any row is read). Exercises the file-invariant check."""
    import h5py
    path = tiny_env_dir / "lv1_eval.h5"
    with h5py.File(path, "r") as f:
        inputs = f["inputs"][:]
        outputs = f["outputs"][:]
        suites = [s for s in f["transformation_suites"].asstr()]
        vqo = [s for s in f["valid_query_outputs"].asstr()]
    trimmed = outputs[:, :-1]  # drop last slot → P mismatch
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=trimmed)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
        f.create_dataset("valid_query_outputs",
                         data=np.array(vqo, dtype=object), dtype=dt)
    with pytest.raises(ValueError, match="inputs.shape.*!= outputs.shape"):
        BabyTheoristEnv(tiny_env_dir, levels=(1,))


# ---- 34-36: mode-aware benchmark directories -----------------------------

def _write_minimal_generated_file(path, split: str, n: int):
    import h5py

    P = 1 if split == "train" else 2
    inputs = np.zeros((n, P, 4, 4), dtype=np.int8)
    outputs = np.zeros((n, P, 4, 4), dtype=np.int8)
    suites = [
        json.dumps([{"selector": {"type": "by_index", "param": 1},
                     "primitive": "rot90", "target": None}])
        for _ in range(n)
    ]
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
        if split == "eval":
            vqo = [json.dumps([outputs[i, 1].tolist()]) for i in range(n)]
            f.create_dataset("valid_query_outputs",
                             data=np.array(vqo, dtype=object), dtype=dt)


def _mode_info_json(mode: str) -> str:
    return json.dumps({
        "mode": mode,
        "shape_generation": {
            "profile": "standard",
            "shape_size_range": [4, 16],
            "shape_compactness": 8.0,
            "shape_bordered_hollow_min_size": 9,
            "shape_bordered_hollow_random_inner_size": True,
            "shape_max_bbox_span": 5,
        },
    })


def test_mode_selects_subdir_and_generates_missing(tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate(out_path, level, split, mode, cfg):
        calls.append((out_path, level, split, mode, cfg))
        _write_minimal_generated_file(
            out_path, split, cfg.train_n if split == "train" else cfg.eval_n,
        )
        out_path.with_suffix(".info.json").write_text(_mode_info_json(mode))

    monkeypatch.setattr(env_mod, "_generate_dataset_file", fake_generate)

    cfg = GenerationConfig(train_n=2, eval_n=3, workers=7)
    root = tmp_path / "data"
    env = BabyTheoristEnv(root, levels=(1,), mode="single-A", generation=cfg)

    assert env.data_dir == root / env_mod._default_mode_dir_name("single_a")
    assert env.mode == "single_a"
    assert env.train_size(1) == 2
    assert len(env.eval(1)) == 3
    assert [(c[1], c[2], c[3], c[4]) for c in calls] == [
        (1, "train", "single_a", cfg),
        (1, "eval", "single_a", cfg),
    ]


def test_generate_file_uses_mode_default_n_shapes(tmp_path, monkeypatch):
    import babytheorist.generation.cli as generate_cli
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(generate_cli, "main", fake_generate_main)

    env_mod._generate_dataset_file(
        tmp_path / "single_a.h5",
        level=1,
        split="train",
        mode="single_a",
        cfg=GenerationConfig(train_n=1),
    )
    env_mod._generate_dataset_file(
        tmp_path / "multi_a.h5",
        level=1,
        split="train",
        mode="multi_a",
        cfg=GenerationConfig(train_n=1),
    )
    env_mod._generate_dataset_file(
        tmp_path / "multi_dev.h5",
        level=1,
        split="train",
        mode="multi_dev",
        cfg=GenerationConfig(train_n=1),
    )
    env_mod._generate_dataset_file(
        tmp_path / "override.h5",
        level=1,
        split="train",
        mode="single_a",
        cfg=GenerationConfig(train_n=1, n_shapes=3),
    )
    env_mod._generate_dataset_file(
        tmp_path / "multi_b.h5",
        level=1,
        split="train",
        mode="multi_b",
        cfg=GenerationConfig(train_n=1),
    )

    def n_shapes_arg(argv):
        return argv[argv.index("--n-shapes") + 1]

    def arg_after(argv, name):
        return argv[argv.index(name) + 1]

    assert [n_shapes_arg(argv) for argv in calls] == ["1", "2", "2", "3", "3"]
    assert [arg_after(argv, "--benchmark-mode") for argv in calls] == [
        "single_a", "multi_a", "multi_dev", "single_a", "multi_b",
    ]
    for argv in calls:
        assert "--primitives" not in argv
        assert "--selectors" not in argv
        assert "--depth-distribution" not in argv
        assert "--depth-gamma" not in argv
        assert "--depth-prior" not in argv
        assert "--filter-system" not in argv
        assert arg_after(argv, "--shape-profile") == "standard"
        assert "--shape-min" not in argv
        assert "--shape-max" not in argv
        assert arg_after(argv, "--pool-seed") == "42"
        assert arg_after(argv, "--min-support") == "2"
        assert arg_after(argv, "--max-support") == "5"
        assert arg_after(argv, "--attempt-scheduler") == "ordered"
        assert "--attempt-timeout-sec" not in argv
        assert arg_after(argv, "--dominant-pool-dir") == str(
            tmp_path.parent / f"dominant_pools-v{env_mod._babytheorist_code_version()}"
        )


def test_generate_file_respects_explicit_dominant_pool_dir(tmp_path, monkeypatch):
    import babytheorist.generation.cli as generate_cli
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(generate_cli, "main", fake_generate_main)

    env_mod._generate_dataset_file(
        tmp_path / "custom_pool.h5",
        level=1,
        split="train",
        mode="single_b",
        cfg=GenerationConfig(
            train_n=1,
            dominant_pool_dir=tmp_path / "my_pools",
        ),
    )

    argv = calls[0]
    assert argv[argv.index("--dominant-pool-dir") + 1] == str(
        tmp_path / "my_pools"
    )


def test_generate_file_passes_attempt_scheduler(tmp_path, monkeypatch):
    import babytheorist.generation.cli as generate_cli
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(generate_cli, "main", fake_generate_main)

    env_mod._generate_dataset_file(
        tmp_path / "unordered_scheduler.h5",
        level=1,
        split="train",
        mode="multi_dev",
        cfg=GenerationConfig(train_n=1, attempt_scheduler="unordered"),
    )

    argv = calls[0]
    assert argv[argv.index("--attempt-scheduler") + 1] == "unordered"


def test_generate_file_passes_attempt_timeout(tmp_path, monkeypatch):
    import babytheorist.generation.cli as generate_cli
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(generate_cli, "main", fake_generate_main)

    env_mod._generate_dataset_file(
        tmp_path / "attempt_timeout.h5",
        level=1,
        split="eval",
        mode="multi_dev",
        cfg=GenerationConfig(eval_n=1, attempt_timeout_sec=5.0),
    )

    argv = calls[0]
    assert argv[argv.index("--attempt-timeout-sec") + 1] == "5.0"


def test_generate_file_passes_shape_profile_and_overrides(tmp_path, monkeypatch):
    import babytheorist.generation.cli as generate_cli
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate_main(argv):
        calls.append(argv)
        return 0

    monkeypatch.setattr(generate_cli, "main", fake_generate_main)

    env_mod._generate_dataset_file(
        tmp_path / "shape_profile.h5",
        level=1,
        split="train",
        mode="multi_dev",
        cfg=GenerationConfig(
            train_n=1,
            shape_profile="standard",
            shape_min=5,
            shape_max=12,
            shape_compactness=4.0,
            shape_border_min=0,
            shape_max_bbox_span=0,
        ),
    )

    argv = calls[0]

    def arg_after(name):
        return argv[argv.index(name) + 1]

    assert arg_after("--shape-profile") == "standard"
    assert arg_after("--shape-min") == "5"
    assert arg_after("--shape-max") == "12"
    assert arg_after("--shape-compactness") == "4.0"
    assert arg_after("--shape-border-min") == "0"
    assert arg_after("--shape-max-bbox-span") == "0"


def test_generation_config_rejects_invalid_n_shapes():
    with pytest.raises(ValueError, match="n_shapes"):
        GenerationConfig(n_shapes=0)


def test_generation_config_rejects_invalid_support_bounds():
    with pytest.raises(ValueError, match="min_support"):
        GenerationConfig(min_support=0)
    with pytest.raises(ValueError, match="max_support"):
        GenerationConfig(max_support=0)
    with pytest.raises(ValueError, match="min_support=6 > max_support=5"):
        GenerationConfig(min_support=6, max_support=5)
    with pytest.raises(ValueError, match="attempt_timeout_sec"):
        GenerationConfig(attempt_timeout_sec=0)


def test_generation_config_rejects_invalid_shape_options():
    with pytest.raises(ValueError, match="shape_profile"):
        GenerationConfig(shape_profile="not_a_profile")
    with pytest.raises(ValueError, match="shape_profile"):
        GenerationConfig(shape_profile="v2_large")
    with pytest.raises(ValueError, match="shape_min"):
        GenerationConfig(shape_min=0)
    with pytest.raises(ValueError, match="shape_max=3 < shape_min=4"):
        GenerationConfig(shape_min=4, shape_max=3)
    with pytest.raises(ValueError, match="shape_compactness"):
        GenerationConfig(shape_compactness=-1.0)
    with pytest.raises(ValueError, match="shape_border_min"):
        GenerationConfig(shape_border_min=-1)
    with pytest.raises(ValueError, match="shape_max_bbox_span"):
        GenerationConfig(shape_max_bbox_span=-1)


def test_mode_reuses_existing_subdir(tiny_env_dir, tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")

    def fail_generate(*args, **kwargs):
        raise AssertionError("existing mode data should not be regenerated")

    monkeypatch.setattr(env_mod, "_generate_dataset_file", fail_generate)
    multi_b_dir = tmp_path / "data" / env_mod._default_mode_dir_name("multi_b")
    multi_b_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", multi_b_dir / f"lv1_{split}.h5")
        (multi_b_dir / f"lv1_{split}.info.json").write_text(_mode_info_json("multi_b"))

    env = BabyTheoristEnv(tmp_path / "data", levels=(1,), mode="multi-B")

    assert env.data_dir == multi_b_dir
    assert env.train_size(1) == 6


def test_mode_alias_requires_alias_subdir(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_easy")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(_mode_info_json("single_easy"))

    with pytest.raises(FileNotFoundError, match="single_a"):
        BabyTheoristEnv(
            tmp_path / "data",
            levels=(1,),
            mode="single-A",
            generate_missing=False,
        )


def test_mode_alias_reuses_existing_alias_subdir(tiny_env_dir, tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")

    def fail_generate(*args, **kwargs):
        raise AssertionError("existing alias data should not be regenerated")

    monkeypatch.setattr(env_mod, "_generate_dataset_file", fail_generate)
    single_a_dir = tmp_path / "data" / env_mod._versioned_mode_dir_name("single_a")
    single_a_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_a_dir / f"lv1_{split}.h5")
        (single_a_dir / f"lv1_{split}.info.json").write_text(_mode_info_json("single_easy"))

    env = BabyTheoristEnv(tmp_path / "data", levels=(1,), mode="single-A")

    assert env.mode == "single_a"
    assert env.data_dir == single_a_dir
    assert env.train_size(1) == 6


def test_mode_alias_generates_alias_subdir_when_missing(tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")
    calls = []

    def fake_generate(out_path, level, split, mode, cfg):
        calls.append((out_path, level, split, mode, cfg))
        _write_minimal_generated_file(
            out_path, split, cfg.train_n if split == "train" else cfg.eval_n,
        )
        out_path.with_suffix(".info.json").write_text(_mode_info_json(mode))

    monkeypatch.setattr(env_mod, "_generate_dataset_file", fake_generate)

    cfg = GenerationConfig(train_n=2, eval_n=3)
    root = tmp_path / "data"
    env = BabyTheoristEnv(root, levels=(1,), mode="single-A", generation=cfg)

    assert env.mode == "single_a"
    assert env.data_dir == root / env_mod._versioned_mode_dir_name("single_a")
    assert env.train_size(1) == 2
    assert [(c[1], c[2], c[3], c[4]) for c in calls] == [
        (1, "train", "single_a", cfg),
        (1, "eval", "single_a", cfg),
    ]


def test_mode_metadata_mismatch_raises(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text('{"mode": "multi_b"}')

    with pytest.raises(ValueError, match="mode=.*does not match"):
        BabyTheoristEnv(tmp_path / "data", levels=(1,), mode="single_easy")


def test_non_official_depth_prior_metadata_raises(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(
            '{"mode": "single_easy", "depth_distribution": "legacy_geometric"}'
        )

    with pytest.raises(ValueError, match="depth_distribution=.*does not match"):
        BabyTheoristEnv(tmp_path / "data", levels=(1,), mode="single_easy")


def test_shape_profile_metadata_mismatch_raises(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    current_shape_info = (
        '{"mode": "single_easy", "depth_distribution": "piecewise_geometric", '
        '"shape_generation": {"profile": "v2_large", '
        '"shape_size_range": [4, 16], "shape_compactness": 8.0, '
        '"shape_bordered_hollow_min_size": 9, '
        '"shape_bordered_hollow_random_inner_size": true, '
        '"shape_max_bbox_span": 5}}'
    )
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(current_shape_info)

    with pytest.raises(ValueError, match="shape_generation\\.profile=.*does not match"):
        BabyTheoristEnv(
            tmp_path / "data",
            levels=(1,),
            mode="single_easy",
            generation=GenerationConfig(shape_profile="standard"),
        )


def test_requested_shape_profile_requires_metadata(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(
            '{"mode": "single_easy", "depth_distribution": "piecewise_geometric"}'
        )

    with pytest.raises(ValueError, match="missing shape_generation metadata"):
        BabyTheoristEnv(
            tmp_path / "data",
            levels=(1,),
            mode="single_easy",
            generation=GenerationConfig(shape_profile="standard"),
        )


def test_standard_shape_overrides_require_metadata(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(
            '{"mode": "single_easy", "depth_distribution": "piecewise_geometric"}'
        )

    with pytest.raises(ValueError, match="missing shape_generation metadata"):
        BabyTheoristEnv(
            tmp_path / "data",
            levels=(1,),
            mode="single_easy",
            generation=GenerationConfig(shape_profile="standard", shape_min=5),
        )


def test_standard_shape_defaults_require_metadata(tiny_env_dir, tmp_path):
    import importlib

    env_mod = importlib.import_module("env.env")
    single_easy_dir = tmp_path / "data" / env_mod._default_mode_dir_name("single_a")
    single_easy_dir.mkdir(parents=True)
    for split in ("train", "eval"):
        shutil.copy(tiny_env_dir / f"lv1_{split}.h5", single_easy_dir / f"lv1_{split}.h5")
        (single_easy_dir / f"lv1_{split}.info.json").write_text(
            '{"mode": "single_easy", "depth_distribution": "piecewise_geometric"}'
        )

    with pytest.raises(ValueError, match="missing shape_generation metadata"):
        BabyTheoristEnv(
            tmp_path / "data",
            levels=(1,),
            mode="single_easy",
            generation=GenerationConfig(shape_profile="standard"),
        )


def test_mode_rejects_conflicting_explicit_mode_dir(tmp_path):
    with pytest.raises(ValueError, match="conflicts with requested mode"):
        BabyTheoristEnv(
            tmp_path / "data" / "multi_hard",
            levels=(1,),
            mode="single_easy",
            generate_missing=False,
        )


def test_generation_rechecks_after_lock(tiny_env_dir, tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")
    generated = []

    def fail_generate(*args, **kwargs):
        raise AssertionError("target created while waiting for lock should be reused")

    @contextmanager
    def fake_lock(target):
        split = "train" if target.name.endswith("_train.h5") else "eval"
        n = 2 if split == "train" else 3
        _write_minimal_generated_file(target, split, n)
        target.with_suffix(".info.json").write_text(_mode_info_json("single_easy"))
        generated.append(target)
        yield True

    monkeypatch.setattr(env_mod, "_generation_lock", fake_lock)
    monkeypatch.setattr(env_mod, "_generate_dataset_file", fail_generate)

    env = BabyTheoristEnv(tmp_path / "data", levels=(1,), mode="single_easy")

    assert len(generated) == 2
    assert env.train_size(1) == 2
    assert len(env.eval(1)) == 3


def test_generation_lock_reclaims_stale_pid_lock(tmp_path, monkeypatch):
    import importlib

    env_mod = importlib.import_module("env.env")
    target = tmp_path / "lv1_eval.h5"
    lock_path = target.with_name(f"{target.name}.lock")
    lock_path.write_text("123456")

    def fake_kill(pid, sig):
        assert pid == 123456
        assert sig == 0
        raise ProcessLookupError

    monkeypatch.setattr(env_mod.os, "kill", fake_kill)

    with env_mod._generation_lock(target) as should_generate:
        assert should_generate is True
        assert lock_path.exists()

    assert not lock_path.exists()


def test_generate_missing_requires_mode(tmp_path):
    with pytest.raises(ValueError, match="mode to be set"):
        BabyTheoristEnv(tmp_path / "data", levels=(1,), generate_missing=True)
