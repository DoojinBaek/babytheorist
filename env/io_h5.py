"""HDF5 reader for BabyTheorist datasets: train (P=1) and eval (support + query + Y_q).

Single source of truth for h5 schema reads. `babytheorist/review.py` writes the h5;
this module reads it. Other consumers (env/env.py, scripts/demo_*.py) go
through here to avoid decode drift.

Schema:
    inputs                 (N, P, H, W)  int8
    outputs                (N, P, H, W)  int8
    depths                 (N,)          int32 sampled program depth
    transformation_suites  (N,)          vlen JSON program
    valid_query_outputs    (N,)          vlen JSON list[list[list[int]]]  (eval only)
    n_support              (N,)          int, optional — per-row K (eval only)

Eval contract (row i): support occupies slots [0, K), query at slot K.
  • Uniform mode (no `n_support`):  K = P - 1 for every row.
  • Variable mode (with `n_support`): K = n_support[i] ∈ [1, P-1];
    slots (K, P) are padding and are not read.

File-invariant checks (`inputs.shape == outputs.shape`, eval `P >= 2`) are
done once by `BabyTheoristEnv.__init__`; these readers trust them. Per-row dtype /
range / shape checks run here so corrupt rows fail loud.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py


# ---- validation -----------------------------------------------------------

def assert_grid(arr: np.ndarray, ctx: str,
                expected_shape: tuple | None = None) -> np.ndarray:
    """Validate a grid (2D) or grid block (3D). Checks dtype integer, values
    in [0, 9], and either `expected_shape` (if given) or `ndim == 2`.
    Shared between h5 readers and prediction validation."""
    if expected_shape is not None:
        if arr.shape != expected_shape:
            raise ValueError(f"{ctx}: shape={arr.shape}, expected {expected_shape}")
    elif arr.ndim != 2:
        raise ValueError(f"{ctx}: ndim={arr.ndim}, expected 2")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{ctx}: dtype={arr.dtype}, expected integer")
    if arr.size and (arr.min() < 0 or arr.max() > 9):
        raise ValueError(f"{ctx}: values out of [0, 9]")
    return arr


# ---- train split ----------------------------------------------------------

def read_train_row(h: "h5py.File", i: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) for train row i. Train h5 has P=1."""
    x = assert_grid(h["inputs"][i, 0], f"train row {i}: x")
    y = assert_grid(h["outputs"][i, 0], f"train row {i}: y")
    return x, y


# ---- eval split -----------------------------------------------------------

def _row_K(h: "h5py.File", i: int, P: int) -> int:
    """Resolve per-row K: n_support[i] if present, else uniform P-1."""
    if "n_support" in h:
        K = int(h["n_support"][i])
        if not 1 <= K <= P - 1:
            raise ValueError(
                f"eval row {i}: n_support={K} out of [1, {P - 1}] (P={P})"
            )
        return K
    return P - 1


def read_eval_row(h: "h5py.File", i: int, P: int) -> dict:
    """Return {support_x, support_y, query_x, y_q, Y_q} for eval row i.

    `support_x` / `support_y` have shape (K, H, W); `query_x` / `y_q` are (H, W).
    `Y_q` is canonicalized so `Y_q[0] == y_q`. Caller supplies `P = inputs.shape[1]`
    (validated once by BabyTheoristEnv).
    """
    K = _row_K(h, i, P)
    H, W = h["inputs"].shape[2:]
    block_shape = (K, H, W)
    # Slab reads: one h5 call each for support_x/y instead of K calls.
    support_x = assert_grid(h["inputs"][i, :K], f"eval row {i}: support_x", block_shape)
    support_y = assert_grid(h["outputs"][i, :K], f"eval row {i}: support_y", block_shape)
    query_x = assert_grid(h["inputs"][i, K], f"eval row {i}: query_x")
    y_q = assert_grid(h["outputs"][i, K], f"eval row {i}: y_q")
    Y_q = _read_Y_q(h, i, expected_shape=query_x.shape, y_q=y_q)
    return {"support_x": support_x, "support_y": support_y,
            "query_x": query_x, "y_q": y_q, "Y_q": Y_q}


def read_Y_q(h: "h5py.File", i: int, P: int) -> tuple[np.ndarray, ...]:
    """Scoring fast-path: returns canonicalized Y_q for eval row i without
    reading support_x / support_y. Reads only the query slot + VQO."""
    K = _row_K(h, i, P)
    H, W = h["inputs"].shape[2:]
    y_q = assert_grid(h["outputs"][i, K], f"eval row {i}: y_q")
    return _read_Y_q(h, i, expected_shape=(H, W), y_q=y_q)


def _read_Y_q(h: "h5py.File", i: int,
              expected_shape: tuple, y_q: np.ndarray) -> tuple[np.ndarray, ...]:
    """Decode, validate, and canonicalize `valid_query_outputs[i]`."""
    raw = json.loads(h["valid_query_outputs"].asstr()[i])
    if not raw:
        raise ValueError(f"eval row {i}: valid_query_outputs is empty")
    Y_q = tuple(
        assert_grid(np.asarray(g), f"eval row {i}: Y_q[{j}]", expected_shape)
        for j, g in enumerate(raw)
    )
    canon = next((j for j, g in enumerate(Y_q) if np.array_equal(g, y_q)), None)
    if canon is None:
        raise ValueError(f"eval row {i}: canonical y_q not found in Y_q")
    if canon != 0:
        Y_q = (Y_q[canon],) + Y_q[:canon] + Y_q[canon + 1:]
    return Y_q


# ---- program (annotation) -------------------------------------------------

def read_program(h: "h5py.File", i: int) -> list[dict]:
    """Return row i's program (Step list). Same for train and eval."""
    return json.loads(h["transformation_suites"].asstr()[i])
