#!/usr/bin/env python
"""Validate BabyTheorist generated dataset directories.

Default target is the current development suites:

    data/multi_dev-v<version>
    data/single_a-v<version>
    data/multi_a-v<version>
    data/single_b-v<version>
    data/multi_b-v<version>

The default checks are schema, sidecar metadata, sampled grid values, sampled
program JSON, and sampled eval answer JSON. Use --full-grid-check when preparing
a release artifact and you want to scan every grid value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SUITES = (
    "multi_dev", "single_a", "multi_a", "single_b", "multi_b",
)
REQUIRED_COMMON = (
    "inputs",
    "outputs",
    "grid_sizes",
    "n_shapes",
    "depths",
    "transformation_suites",
)
REQUIRED_INFO_KEYS = (
    "babytheorist_version",
    "benchmark_spec_version",
    "dataset_release_version",
    "generator_backend",
    "primitive_vocabulary",
    "selector_vocabulary",
)


def _repo_version() -> str:
    ns: dict[str, object] = {}
    version_path = Path(__file__).resolve().parents[1] / "babytheorist" / "version.py"
    exec(version_path.read_text(), ns)
    return str(ns["__version__"])


def _row_sample_indices(n: int, limit: int) -> list[int]:
    if n <= 0:
        return []
    if limit <= 0 or limit >= n:
        return list(range(n))
    return sorted(set(int(i) for i in np.linspace(0, n - 1, limit)))


class Validator:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, path: Path, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def check(self, condition: bool, path: Path, msg: str) -> None:
        if not condition:
            self.fail(path, msg)


def _assert_integer_grid_values(
    v: Validator,
    path: Path,
    name: str,
    ds,
    *,
    sample_rows: int,
    full_grid_check: bool,
) -> None:
    if not np.issubdtype(ds.dtype, np.integer):
        v.fail(path, f"{name}.dtype={ds.dtype}, expected integer")
        return
    rows = range(ds.shape[0]) if full_grid_check else _row_sample_indices(ds.shape[0], sample_rows)
    for i in rows:
        arr = ds[i]
        if arr.size and (arr.min() < 0 or arr.max() > 9):
            v.fail(path, f"{name}[{i}] has values outside [0, 9]")
            return


def _check_info(
    v: Validator,
    info_path: Path,
    *,
    suite: str,
    version: str,
) -> dict:
    if not info_path.exists():
        v.fail(info_path, "missing sidecar metadata")
        return {}
    try:
        info = json.loads(info_path.read_text())
    except Exception as e:
        v.fail(info_path, f"invalid JSON ({e})")
        return {}
    for key in REQUIRED_INFO_KEYS:
        v.check(key in info, info_path, f"missing info key {key!r}")
    expected_release = f"BabyTheorist-{suite}-v{version}"
    if "babytheorist_version" in info:
        v.check(
            info["babytheorist_version"] == version,
            info_path,
            f"babytheorist_version={info['babytheorist_version']!r}, expected {version!r}",
        )
    if "benchmark_spec_version" in info:
        v.check(
            info["benchmark_spec_version"] == version,
            info_path,
            "benchmark_spec_version does not match code version "
            f"{version!r}",
        )
    if "dataset_release_version" in info:
        v.check(
            info["dataset_release_version"] == expected_release,
            info_path,
            "dataset_release_version="
            f"{info['dataset_release_version']!r}, expected {expected_release!r}",
        )
    return info


def _check_program_rows(
    v: Validator,
    path: Path,
    h,
    *,
    sample_rows: int,
) -> None:
    rows = _row_sample_indices(h["inputs"].shape[0], sample_rows)
    suites = h["transformation_suites"].asstr()
    depths = h["depths"]
    for i in rows:
        try:
            program = json.loads(suites[i])
        except Exception as e:
            v.fail(path, f"transformation_suites[{i}] invalid JSON ({e})")
            continue
        if not isinstance(program, list):
            v.fail(path, f"transformation_suites[{i}] is not a list")
            continue
        if len(program) != int(depths[i]):
            v.fail(
                path,
                f"depths[{i}]={int(depths[i])} but program has "
                f"{len(program)} steps",
            )


def _check_eval_answers(
    v: Validator,
    path: Path,
    h,
    *,
    sample_rows: int,
) -> None:
    rows = _row_sample_indices(h["inputs"].shape[0], sample_rows)
    raw = h["valid_query_outputs"].asstr()
    for i in rows:
        try:
            yq = json.loads(raw[i])
        except Exception as e:
            v.fail(path, f"valid_query_outputs[{i}] invalid JSON ({e})")
            continue
        if not isinstance(yq, list) or not yq:
            v.fail(path, f"valid_query_outputs[{i}] is empty or not a list")


def _validate_h5(
    v: Validator,
    h5_path: Path,
    split: str,
    *,
    sample_rows: int,
    full_grid_check: bool,
) -> None:
    if not h5_path.exists():
        v.fail(h5_path, "missing dataset file")
        return
    try:
        import h5py
    except ImportError:
        v.fail(h5_path, "validation requires h5py (pip install h5py)")
        return
    try:
        h = h5py.File(h5_path, "r")
    except Exception as e:
        v.fail(h5_path, f"failed to open HDF5 ({e})")
        return
    with h:
        required = set(REQUIRED_COMMON)
        if split == "eval":
            required.add("valid_query_outputs")
        for key in sorted(required):
            v.check(key in h, h5_path, f"missing dataset {key!r}")
        if any(key not in h for key in required):
            return

        inputs = h["inputs"]
        outputs = h["outputs"]
        v.check(inputs.shape == outputs.shape, h5_path, "inputs/outputs shape mismatch")
        v.check(len(inputs.shape) == 4, h5_path, f"inputs.ndim={len(inputs.shape)}, expected 4")
        if len(inputs.shape) != 4:
            return
        n, p, hgt, wid = inputs.shape
        v.check(n > 0, h5_path, "N must be positive")
        v.check(hgt == wid, h5_path, f"grid is not square: {(hgt, wid)}")
        if split == "train":
            v.check(p == 1, h5_path, f"train P={p}, expected 1")
        else:
            v.check(p >= 2, h5_path, f"eval P={p}, expected >= 2")

        v.check(h["grid_sizes"].shape == (n, 2), h5_path, "bad grid_sizes shape")
        v.check(h["n_shapes"].shape == (n, p), h5_path, "bad n_shapes shape")
        v.check(h["depths"].shape == (n,), h5_path, "bad depths shape")
        v.check(h["transformation_suites"].shape == (n,), h5_path, "bad transformation_suites shape")

        if split == "eval" and "n_support" in h:
            ns = h["n_support"][:]
            v.check(ns.shape == (n,), h5_path, "bad n_support shape")
            if ns.size and (ns.min() < 1 or ns.max() > p - 1):
                v.fail(h5_path, f"n_support values outside [1, {p - 1}]")

        _assert_integer_grid_values(
            v, h5_path, "inputs", inputs,
            sample_rows=sample_rows, full_grid_check=full_grid_check,
        )
        _assert_integer_grid_values(
            v, h5_path, "outputs", outputs,
            sample_rows=sample_rows, full_grid_check=full_grid_check,
        )
        _check_program_rows(v, h5_path, h, sample_rows=sample_rows)
        if split == "eval":
            _check_eval_answers(v, h5_path, h, sample_rows=sample_rows)


def validate_suite(
    v: Validator,
    suite_dir: Path,
    *,
    suite: str,
    version: str,
    levels: Iterable[int],
    sample_rows: int,
    full_grid_check: bool,
) -> None:
    v.check(suite_dir.exists(), suite_dir, "missing suite directory")
    v.check(
        suite_dir.name == f"{suite}-v{version}",
        suite_dir,
        f"directory name should be {suite}-v{version}",
    )
    for level in levels:
        for split in ("train", "eval"):
            h5_path = suite_dir / f"lv{level}_{split}.h5"
            _check_info(v, h5_path.with_suffix(".info.json"), suite=suite, version=version)
            _validate_h5(
                v, h5_path, split,
                sample_rows=sample_rows,
                full_grid_check=full_grid_check,
            )


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Validate BabyTheorist generated HDF5 datasets.")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--version", default=_repo_version())
    ap.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES))
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--sample-rows", type=int, default=16)
    ap.add_argument("--full-grid-check", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    v = Validator()
    root = Path(args.data_root)
    for suite in args.suites:
        suite_dir = root / f"{suite}-v{args.version}"
        validate_suite(
            v, suite_dir,
            suite=suite,
            version=args.version,
            levels=args.levels,
            sample_rows=args.sample_rows,
            full_grid_check=args.full_grid_check,
        )
    if v.errors:
        for err in v.errors:
            print(f"ERROR: {err}", file=sys.stderr)
        print(f"validation failed: {len(v.errors)} error(s)", file=sys.stderr)
        return 1
    print(
        "validation passed: "
        f"{', '.join(args.suites)} at v{args.version} "
        f"levels={','.join(str(lv) for lv in args.levels)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
