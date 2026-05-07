#!/usr/bin/env python3
"""Compare generated BabyTheorist datasets between main and the current checkout.

The script is intentionally conservative: it generates each suite in the same
level/split order as the official shell scripts and compares every HDF5 dataset
exactly. Sidecar metadata is also compared after removing volatile timing/git
fields that cannot be stable across runs.

Example smoke run:

    python scripts/compare_generation_equivalence.py --train-n 2 --eval-n 2 \
        --levels 1 --modes single_a --workers 1

Full guard requested for refactors:

    python scripts/compare_generation_equivalence.py --train-n 10000 --eval-n 10000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT))

from babytheorist.benchmark import BENCHMARK_MODES, normalize_benchmark_mode

DEFAULT_MODES = BENCHMARK_MODES
DEFAULT_LEVELS = tuple(range(1, 9))
SPLITS = ("train", "eval")
VOLATILE_INFO_KEYS = {
    "git_commit",
    "git_dirty",
    "dominant_pool_dir",
    "elapsed_sec",
    "tasks_per_sec",
    "sampler_elapsed_sec",
    "pair_elapsed_sec",
    "avg_sampler_sec_per_call",
    "avg_pair_sec_per_attempt",
}


@dataclass(frozen=True)
class Job:
    mode: str
    level: int
    split: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.mode, self.level, self.split)

    @property
    def label(self) -> str:
        return f"{self.mode} L{self.level} {self.split}"

    @property
    def split_mode(self) -> str:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        return self.split


@dataclass(frozen=True)
class GeneratedFile:
    path: Path
    elapsed_sec: float


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), f"(cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def check_output(cmd: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(cmd, cwd=cwd, text=True).strip()


def repo_root(path: Path) -> Path:
    return Path(check_output(["git", "rev-parse", "--show-toplevel"], cwd=path))


def resolve_git_ref(source_root: Path, ref: str) -> str:
    return check_output(["git", "rev-parse", f"{ref}^{{commit}}"], cwd=source_root)


def materialize_baseline_snapshot(
    source_root: Path, baseline_ref: str, baseline_commit: str, target: Path,
) -> Path:
    marker_path = target / ".babytheorist_baseline_ref"
    if target.exists():
        marker = marker_path.read_text().strip() if marker_path.exists() else None
        if marker != baseline_commit:
            raise RuntimeError(
                f"baseline snapshot {target} is for {marker!r}, "
                f"not {baseline_ref!r} ({baseline_commit}). "
                "Use a fresh --work-dir."
            )
        print(f"using existing baseline snapshot {target}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir()
    print(f"extracting {baseline_ref} into {target}", flush=True)
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", baseline_commit],
        cwd=source_root,
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        tar.extractall(target, filter="data")
    marker_path.write_text(f"{baseline_commit}\n")
    return target


def read_version(root: Path) -> str:
    ns: dict[str, object] = {}
    version_path = root / "babytheorist" / "version.py"
    if not version_path.exists():
        version_path = root / "parc" / "version.py"
    exec(version_path.read_text(), ns)
    return str(ns["__version__"])


def output_dir(root: Path, mode: str, version: str) -> Path:
    return root / "data" / f"{mode}-v{version}"


def dominant_pool_dir(root: Path, version: str) -> Path:
    return root / "data" / f"dominant_pools-v{version}"


def generation_cmd(
    python: str,
    *,
    mode: str,
    level: int,
    split: str,
    n: int,
    out_file: Path,
    out_dir: Path,
    pool_dir: Path,
    workers: int,
    max_support: int,
    seed: int,
    pool_seed: int,
    version: str,
) -> list[str]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    split_mode = split
    split_seed = seed if split == "train" else seed + 1
    return [
        python,
        "generate.py",
        "--level",
        str(level),
        "--n",
        str(n),
        "--mode",
        split_mode,
        "--world-size",
        "15",
        "--shape-profile",
        "standard",
        "--benchmark-mode",
        mode,
        "--max-support",
        str(max_support),
        "--backend",
        "filter",
        "--workers",
        str(workers),
        "--seed",
        str(split_seed),
        "--pool-seed",
        str(pool_seed),
        "--dominant-pool-dir",
        str(pool_dir),
        "--dataset-version",
        f"BabyTheorist-{mode}-v{version}",
        "--exclude",
        str(out_dir),
        "--out",
        str(out_file),
    ]


def generate_one(
    root: Path,
    out_root: Path,
    python: str,
    job: Job,
    *,
    train_n: int,
    eval_n: int,
    workers: int,
    max_support: int,
    seed: int,
    pool_seed: int,
) -> GeneratedFile:
    version = read_version(root)
    mode_out_dir = output_dir(out_root, job.mode, version)
    mode_out_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = dominant_pool_dir(out_root, version)
    n = train_n if job.split == "train" else eval_n
    out_file = mode_out_dir / f"lv{job.level}_{job.split}.h5"
    cmd = generation_cmd(
        python,
        mode=job.mode,
        level=job.level,
        split=job.split,
        n=n,
        out_file=out_file,
        out_dir=mode_out_dir,
        pool_dir=pool_dir,
        workers=workers,
        max_support=max_support,
        seed=seed,
        pool_seed=pool_seed,
        version=version,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    t0 = time.perf_counter()
    run(cmd, cwd=root, env=env)
    elapsed = time.perf_counter() - t0
    return GeneratedFile(path=out_file, elapsed_sec=elapsed)


def generated_path(root: Path, out_root: Path, job: Job) -> Path:
    version = read_version(root)
    return output_dir(out_root, job.mode, version) / f"lv{job.level}_{job.split}.h5"


def compare_h5(a_path: Path, b_path: Path) -> list[str]:
    errors: list[str] = []
    with h5py.File(a_path, "r") as a, h5py.File(b_path, "r") as b:
        a_keys = sorted(a.keys())
        b_keys = sorted(b.keys())
        if a_keys != b_keys:
            return [f"h5 keys differ: {a_keys} != {b_keys}"]
        for key in a_keys:
            da = a[key]
            db = b[key]
            if da.shape != db.shape:
                errors.append(f"{key}: shape {da.shape} != {db.shape}")
                continue
            if da.dtype != db.dtype:
                errors.append(f"{key}: dtype {da.dtype} != {db.dtype}")
                continue
            storage_errors = compare_dataset_storage(key, da, db)
            if storage_errors:
                errors.extend(storage_errors)
                continue
            if string_dtype(da) is not None or string_dtype(db) is not None:
                if string_dtype(da) != string_dtype(db):
                    errors.append(
                        f"{key}: string dtype {string_dtype(da)} != {string_dtype(db)}"
                    )
                    continue
                va = list(da.asstr()[:])
                vb = list(db.asstr()[:])
                if va != vb:
                    errors.append(f"{key}: string values differ")
                continue
            va = da[:]
            vb = db[:]
            if not np.array_equal(va, vb):
                diff = np.argwhere(va != vb)
                loc = tuple(int(x) for x in diff[0]) if diff.size else ()
                errors.append(f"{key}: values differ at {loc}")
    return errors


def string_dtype(dataset) -> object:
    return h5py.check_string_dtype(dataset.dtype)


def compare_dataset_storage(key: str, a, b) -> list[str]:
    errors: list[str] = []
    for attr in (
        "chunks",
        "compression",
        "compression_opts",
        "shuffle",
        "fletcher32",
        "scaleoffset",
        "maxshape",
    ):
        av = getattr(a, attr)
        bv = getattr(b, attr)
        if av != bv:
            errors.append(f"{key}: {attr} {av!r} != {bv!r}")
    if a.fillvalue != b.fillvalue:
        errors.append(f"{key}: fillvalue {a.fillvalue!r} != {b.fillvalue!r}")
    return errors


def sanitize_info(value):
    if isinstance(value, dict):
        return {
            key: sanitize_info(val)
            for key, val in sorted(value.items())
            if key not in VOLATILE_INFO_KEYS
        }
    if isinstance(value, list):
        return [sanitize_info(item) for item in value]
    return value


def compare_info(a_path: Path, b_path: Path) -> list[str]:
    a_info = sanitize_info(json.loads(a_path.with_suffix(".info.json").read_text()))
    b_info = sanitize_info(json.loads(b_path.with_suffix(".info.json").read_text()))
    if a_info == b_info:
        return []
    return ["sanitized .info.json differs"]


def timing_row(job: Job, n: int, baseline: GeneratedFile, candidate: GeneratedFile) -> dict:
    baseline_rate = n / baseline.elapsed_sec if baseline.elapsed_sec > 0 else None
    candidate_rate = n / candidate.elapsed_sec if candidate.elapsed_sec > 0 else None
    speed_ratio = (
        candidate.elapsed_sec / baseline.elapsed_sec
        if baseline.elapsed_sec > 0 else None
    )
    throughput_ratio = (
        candidate_rate / baseline_rate
        if baseline_rate and candidate_rate else None
    )
    return {
        "mode": job.mode,
        "level": job.level,
        "split": job.split,
        "n": n,
        "baseline_sec": baseline.elapsed_sec,
        "candidate_sec": candidate.elapsed_sec,
        "baseline_tasks_per_sec": baseline_rate,
        "candidate_tasks_per_sec": candidate_rate,
        "candidate_sec_over_baseline_sec": speed_ratio,
        "candidate_throughput_over_baseline": throughput_ratio,
        "baseline_path": str(baseline.path),
        "candidate_path": str(candidate.path),
    }


def load_timing_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "timing_report.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return list(payload.get("rows", []))


def row_key(row: dict) -> tuple[str, int, str]:
    return (str(row["mode"]), int(row["level"]), str(row["split"]))


def ordered_rows(expected: list[Job], rows_by_key: dict[tuple[str, int, str], dict]) -> list[dict]:
    return [rows_by_key[job.key] for job in expected if job.key in rows_by_key]


def write_timing_report(
    run_dir: Path,
    rows: list[dict],
    expected: list[Job] | None = None,
) -> None:
    payload = {
        "rows": rows,
        "summary": timing_summary(rows, expected),
    }
    (run_dir / "timing_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )
    lines = [
        "# BabyTheorist Generation Timing",
        "",
        "| mode | level | split | n | baseline sec | candidate sec | cand/base sec | base task/s | cand task/s |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mode} | {level} | {split} | {n} | {baseline_sec:.3f} | "
            "{candidate_sec:.3f} | {candidate_sec_over_baseline_sec:.4f} | "
            "{baseline_tasks_per_sec:.3f} | {candidate_tasks_per_sec:.3f} |".format(
                **row
            )
        )
    summary = payload["summary"]
    lines.extend([
        "",
        "## Summary",
        "",
        f"- baseline total sec: {summary['baseline_total_sec']:.3f}",
        f"- candidate total sec: {summary['candidate_total_sec']:.3f}",
        f"- candidate/baseline total sec: {summary['candidate_sec_over_baseline_sec']:.4f}",
        f"- baseline task/s: {summary['baseline_tasks_per_sec']:.3f}",
        f"- candidate task/s: {summary['candidate_tasks_per_sec']:.3f}",
    ])
    (run_dir / "timing_report.md").write_text("\n".join(lines) + "\n")


def timing_summary(rows: list[dict], expected: list[Job] | None = None) -> dict:
    baseline_total = sum(float(row["baseline_sec"]) for row in rows)
    candidate_total = sum(float(row["candidate_sec"]) for row in rows)
    n_total = sum(int(row["n"]) for row in rows)
    completed = {row_key(row) for row in rows}
    expected_keys = {job.key for job in expected} if expected is not None else completed
    missing = sorted(expected_keys - completed)
    return {
        "n_jobs": len(rows),
        "expected_jobs": len(expected_keys),
        "missing_jobs": [
            f"{mode} L{level} {split}" for mode, level, split in missing
        ],
        "n_total_per_side": n_total,
        "baseline_total_sec": baseline_total,
        "candidate_total_sec": candidate_total,
        "candidate_sec_over_baseline_sec": (
            candidate_total / baseline_total if baseline_total > 0 else None
        ),
        "baseline_tasks_per_sec": n_total / baseline_total if baseline_total > 0 else None,
        "candidate_tasks_per_sec": n_total / candidate_total if candidate_total > 0 else None,
    }


def jobs(modes: Iterable[str], levels: Iterable[int], splits: Iterable[str]) -> list[Job]:
    return [
        Job(mode=mode, level=level, split=split)
        for mode in modes
        for level in levels
        for split in splits
    ]


def parse_csv(raw: str, *, name: str, cast=str):
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(cast(item))
        except ValueError as e:
            raise ValueError(f"{name} contains invalid value {item!r}") from e
    if not values:
        raise ValueError(f"{name} must include at least one value")
    return tuple(values)


def validate_choices(name: str, values: Iterable[str], choices: tuple[str, ...]) -> None:
    invalid = [value for value in values if value not in choices]
    if invalid:
        raise ValueError(
            f"{name} values must be one of {choices}; got {tuple(invalid)!r}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate and compare BabyTheorist datasets from main vs current checkout.",
    )
    p.add_argument("--baseline-ref", default="main")
    p.add_argument("--candidate-root", type=Path, default=Path.cwd())
    p.add_argument("--work-dir", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--modes", default=",".join(DEFAULT_MODES))
    p.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--train-n", type=int, default=10_000)
    p.add_argument("--eval-n", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--max-support", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-seed", type=int, default=42)
    p.add_argument("--stop-on-diff", action="store_true")
    p.add_argument(
        "--skip-info",
        action="store_true",
        help=(
            "Compare HDF5 contents only. Use this when an intentional "
            "project rename changes sidecar metadata but must preserve raw data."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mode_list = [
            normalize_benchmark_mode(mode)
            for mode in parse_csv(args.modes, name="--modes")
        ]
        level_list = parse_csv(args.levels, name="--levels", cast=int)
        split_list = parse_csv(args.splits, name="--splits")
        validate_choices("--modes", mode_list, DEFAULT_MODES)
        validate_choices("--splits", split_list, SPLITS)
    except ValueError as e:
        parser.error(str(e))
    candidate_root = repo_root(args.candidate_root.resolve())
    baseline_commit = resolve_git_ref(candidate_root, args.baseline_ref)
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        work_dir = run_dir.parent
        if run_dir.exists() and not args.resume:
            parser.error("--run-dir already exists; pass --resume to reuse it")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{time.time_ns()}"
        work_dir = (
            args.work_dir or Path("/tmp") / "babytheorist_generation_equivalence"
        ).resolve()
        run_dir = work_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
    baseline_root = materialize_baseline_snapshot(
        candidate_root,
        args.baseline_ref,
        baseline_commit,
        work_dir / "baseline",
    )
    baseline_out = run_dir / "baseline_out"
    candidate_out = run_dir / "candidate_out"
    expected_jobs = jobs(mode_list, level_list, split_list)
    rows_by_key = {row_key(row): row for row in load_timing_rows(run_dir)}
    failures: list[str] = []
    for job in expected_jobs:
        print(f"\n=== compare {job.label} ===", flush=True)
        prior_row = rows_by_key.get(job.key)
        baseline_existing = generated_path(baseline_root, baseline_out, job)
        candidate_existing = generated_path(candidate_root, candidate_out, job)
        if prior_row and baseline_existing.exists() and candidate_existing.exists():
            errors = compare_h5(baseline_existing, candidate_existing)
            if not args.skip_info:
                errors.extend(compare_info(baseline_existing, candidate_existing))
            if not errors:
                write_timing_report(
                    run_dir, ordered_rows(expected_jobs, rows_by_key), expected_jobs,
                )
                label = "already exact" if not args.skip_info else "already HDF5-identical"
                print(f"SKIP {job.label} ({label})", flush=True)
                continue
            print(f"rerunning {job.label}; existing outputs differ", flush=True)
            rows_by_key.pop(job.key, None)

        baseline_file = generate_one(
            baseline_root,
            baseline_out,
            args.python,
            job,
            train_n=args.train_n,
            eval_n=args.eval_n,
            workers=args.workers,
            max_support=args.max_support,
            seed=args.seed,
            pool_seed=args.pool_seed,
        )
        candidate_file = generate_one(
            candidate_root,
            candidate_out,
            args.python,
            job,
            train_n=args.train_n,
            eval_n=args.eval_n,
            workers=args.workers,
            max_support=args.max_support,
            seed=args.seed,
            pool_seed=args.pool_seed,
        )
        n = args.train_n if job.split == "train" else args.eval_n
        rows_by_key[job.key] = timing_row(job, n, baseline_file, candidate_file)
        write_timing_report(
            run_dir, ordered_rows(expected_jobs, rows_by_key), expected_jobs,
        )
        print(
            "timing "
            f"baseline={baseline_file.elapsed_sec:.3f}s "
            f"candidate={candidate_file.elapsed_sec:.3f}s "
            f"candidate/baseline={candidate_file.elapsed_sec / baseline_file.elapsed_sec:.4f}",
            flush=True,
        )
        errors = compare_h5(baseline_file.path, candidate_file.path)
        if not args.skip_info:
            errors.extend(compare_info(baseline_file.path, candidate_file.path))
        if errors:
            label = job.label
            failures.extend(f"{label}: {err}" for err in errors)
            print(f"DIFF {label}", flush=True)
            for err in errors:
                print(f"  - {err}", flush=True)
            if args.stop_on_diff:
                break
        else:
            suffix = "HDF5" if args.skip_info else "exact"
            print(f"OK {job.mode} L{job.level} {job.split} ({suffix})", flush=True)

    if failures:
        print("\nFAILURES", flush=True)
        for failure in failures:
            print(f"- {failure}", flush=True)
        print(f"\nartifacts: {run_dir}", flush=True)
        return 1
    missing = [
        job.label for job in expected_jobs
        if job.key not in rows_by_key
    ]
    if missing:
        print("\nMISSING JOBS", flush=True)
        for label in missing:
            print(f"- {label}", flush=True)
        print(f"\nartifacts: {run_dir}", flush=True)
        return 1
    write_timing_report(
        run_dir, ordered_rows(expected_jobs, rows_by_key), expected_jobs,
    )
    match_kind = "HDF5 contents" if args.skip_info else "datasets"
    print(f"\nall generated {match_kind} matched exactly: {run_dir}", flush=True)
    print(f"timing report: {run_dir / 'timing_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
