"""Fixture: tiny synthetic h5 data_dir with one level, mirroring BabyTheorist schema.

We write h5 directly (instead of going through `babytheorist/review.py`) so tests
exercise the reader end-to-end against the documented schema, independent
of the generator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


H = W = 4
LEVEL = 1
N_TRAIN = 6
N_EVAL = 5


def _write_train(path: Path):
    import h5py
    rng = np.random.default_rng(0)
    inputs = rng.integers(0, 10, size=(N_TRAIN, 1, H, W), dtype=np.int8)
    outputs = rng.integers(0, 10, size=(N_TRAIN, 1, H, W), dtype=np.int8)
    suites = [json.dumps([{"selector": {"type": "ALL", "param": None},
                           "primitive": "rot90", "target": None}])
              for _ in range(N_TRAIN)]
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)


def _write_eval(path: Path):
    import h5py
    rng = np.random.default_rng(1)
    inputs = rng.integers(0, 10, size=(N_EVAL, 2, H, W), dtype=np.int8)
    outputs = rng.integers(0, 10, size=(N_EVAL, 2, H, W), dtype=np.int8)
    suites = [json.dumps([{"selector": {"type": "ALL", "param": None},
                           "primitive": "rot90", "target": None}])
              for _ in range(N_EVAL)]
    # Y_q: canonical (= outputs[i, 1]) placed at a non-zero index so the
    # reader must re-canonicalize. Add a distractor grid as well.
    vqo = []
    for i in range(N_EVAL):
        canon = outputs[i, 1].tolist()
        distractor = (outputs[i, 1] ^ 1).clip(0, 9).tolist()
        vqo.append(json.dumps([distractor, canon]))  # canon at index 1
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
        f.create_dataset("valid_query_outputs",
                         data=np.array(vqo, dtype=object), dtype=dt)


@pytest.fixture
def tiny_env_dir(tmp_path: Path) -> Path:
    pytest.importorskip("h5py")
    _write_train(tmp_path / f"lv{LEVEL}_train.h5")
    _write_eval(tmp_path / f"lv{LEVEL}_eval.h5")
    return tmp_path


def _write_eval_multi_support(path: Path, K: int):
    """Write eval h5 with P = K+1 (K support pairs + 1 query)."""
    import h5py
    rng = np.random.default_rng(2)
    P = K + 1
    inputs = rng.integers(0, 10, size=(N_EVAL, P, H, W), dtype=np.int8)
    outputs = rng.integers(0, 10, size=(N_EVAL, P, H, W), dtype=np.int8)
    suites = [json.dumps([{"selector": {"type": "ALL", "param": None},
                           "primitive": "rot90", "target": None}])
              for _ in range(N_EVAL)]
    vqo = []
    for i in range(N_EVAL):
        canon = outputs[i, -1].tolist()
        distractor = (outputs[i, -1] ^ 1).clip(0, 9).tolist()
        vqo.append(json.dumps([distractor, canon]))
    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
        f.create_dataset("valid_query_outputs",
                         data=np.array(vqo, dtype=object), dtype=dt)


@pytest.fixture
def multi_support_env_dir(tmp_path: Path) -> Path:
    """Env with P=3 (K=2 support). Exercises multi-shot support path."""
    pytest.importorskip("h5py")
    _write_train(tmp_path / f"lv{LEVEL}_train.h5")
    _write_eval_multi_support(tmp_path / f"lv{LEVEL}_eval.h5", K=2)
    return tmp_path


def _write_eval_variable_support(path: Path, K_per_row: list[int]):
    """Write eval h5 with per-row K (ragged support). Slots (K_i, P_max) are
    padding. Query sits at slot K_i, canonical (y_q) placed there."""
    import h5py
    assert len(K_per_row) == N_EVAL
    assert min(K_per_row) >= 1
    P_max = max(K_per_row) + 1
    rng = np.random.default_rng(3)

    inputs = rng.integers(0, 10, size=(N_EVAL, P_max, H, W), dtype=np.int8)
    outputs = rng.integers(0, 10, size=(N_EVAL, P_max, H, W), dtype=np.int8)

    suites = [json.dumps([{"selector": {"type": "ALL", "param": None},
                           "primitive": "rot90", "target": None}])
              for _ in range(N_EVAL)]
    vqo = []
    for i, K in enumerate(K_per_row):
        # Canonical y_q = outputs[i, K] (query slot). Distractor + canon.
        canon = outputs[i, K].tolist()
        distractor = (outputs[i, K] ^ 1).clip(0, 9).tolist()
        vqo.append(json.dumps([distractor, canon]))

    dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as f:
        f.create_dataset("inputs", data=inputs)
        f.create_dataset("outputs", data=outputs)
        f.create_dataset("n_support", data=np.array(K_per_row, dtype=np.int32))
        f.create_dataset("transformation_suites",
                         data=np.array(suites, dtype=object), dtype=dt)
        f.create_dataset("valid_query_outputs",
                         data=np.array(vqo, dtype=object), dtype=dt)


@pytest.fixture
def variable_support_env_dir(tmp_path: Path) -> Path:
    """Env with ragged support: K varies per row (1, 2, 1, 3, 2)."""
    pytest.importorskip("h5py")
    _write_train(tmp_path / f"lv{LEVEL}_train.h5")
    _write_eval_variable_support(
        tmp_path / f"lv{LEVEL}_eval.h5",
        K_per_row=[1, 2, 1, 3, 2],
    )
    return tmp_path


@pytest.fixture
def shapes():
    return {"H": H, "W": W, "LEVEL": LEVEL, "N_TRAIN": N_TRAIN, "N_EVAL": N_EVAL}
