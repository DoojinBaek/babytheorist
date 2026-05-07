"""Benchmark production filter generation throughput.

This is a development tool, not part of the official generator path.  It runs
``generate_filter_dataset`` for one level/configuration and prints JSON records
with task/sec and filter stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from babytheorist.benchmark import (
    benchmark_mode_default_n_shapes,
    resolve_primitive_curriculum,
)
from babytheorist.generation import LevelConfig, TaskConfig, generate_filter_dataset
from babytheorist.generation.filter_dataset import FilterGenerationStats


def _build_level_cfg(args, *, use_dominant: bool) -> LevelConfig:
    curriculum = resolve_primitive_curriculum(args.benchmark_mode, args.level)
    fixed_depth = args.fixed_depth if args.fixed_depth > 0 else None
    return LevelConfig(
        level=args.level,
        primitives=list(curriculum.effective_primitives),
        selectors=curriculum.selectors,
        fixed_depth=fixed_depth,
        palette=tuple(range(1, 10)),
        use_dominant=use_dominant,
        curriculum=curriculum if use_dominant else None,
    )


def _build_task_cfg(args) -> TaskConfig:
    n_shapes = (
        args.n_shapes
        if args.n_shapes > 0
        else benchmark_mode_default_n_shapes(args.benchmark_mode)
    )
    return TaskConfig(
        mode=args.split,
        n_shapes_range=(n_shapes, n_shapes),
        world_size=args.world_size,
        shape_size_range=(args.shape_min, args.shape_max),
        max_dataset_attempts_per_task=args.max_dataset_attempts_per_task,
        max_pair_retries=args.max_pair_retries,
        min_support=args.min_support,
        max_support=args.max_support,
    )


def _run_once(
    args,
    *,
    repeat: int,
    warmup: bool,
) -> dict[str, object]:
    level_cfg = _build_level_cfg(args, use_dominant=not args.no_dominant)
    task_cfg = _build_task_cfg(args)
    stats = FilterGenerationStats()
    t0 = time.perf_counter()
    tasks = generate_filter_dataset(
        level_cfg,
        task_cfg,
        n_tasks=args.n,
        seed=args.seed,
        pool_seed=args.pool_seed,
        n_workers=args.workers,
        stats=stats,
        dominant_pool_dir=args.dominant_pool_dir,
    )
    elapsed = time.perf_counter() - t0
    record = {
        "record_type": "run",
        "filter_system": stats.filter_system,
        "repeat": repeat,
        "warmup": warmup,
        "config": _record_config(args),
        "n_tasks": len(tasks),
        "elapsed_sec_wall": elapsed,
        "tasks_per_sec_wall": len(tasks) / elapsed if elapsed > 0.0 else None,
        "stats": stats.generation_dict(),
        "reject_stats": stats.reject_dict(),
    }
    return record


def _record_config(args) -> dict[str, object]:
    keys = (
        "benchmark_mode",
        "dominant_pool_dir",
        "fixed_depth",
        "level",
        "max_dataset_attempts_per_task",
        "max_pair_retries",
        "max_support",
        "min_support",
        "n",
        "n_shapes",
        "no_dominant",
        "pool_seed",
        "repeats",
        "seed",
        "shape_max",
        "shape_min",
        "split",
        "warmups",
        "workers",
        "world_size",
    )
    return {key: getattr(args, key) for key in keys}


def _summary(args, records: list[dict[str, object]]) -> dict[str, object]:
    by_system: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record.get("tasks_per_sec_wall")
        if isinstance(value, (int, float)):
            by_system[str(record["filter_system"])].append(float(value))
    systems = {}
    for system, values in sorted(by_system.items()):
        systems[system] = {
            "runs": len(values),
            "tasks_per_sec_mean": sum(values) / len(values),
            "tasks_per_sec_min": min(values),
            "tasks_per_sec_max": max(values),
        }
    return {
        "config": _record_config(args),
        "record_type": "summary",
        "systems": systems,
    }


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark BabyTheorist production filter generation throughput.",
    )
    p.add_argument("--benchmark-mode", default="multi-B")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fixed-depth", type=int, default=3,
                   help="Use <=0 to sample from the configured depth prior.")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--split", choices=("train", "eval"), default="train")
    p.add_argument("--n-shapes", type=int, default=0,
                   help="Use <=0 for the benchmark-mode default.")
    p.add_argument("--shape-min", type=int, default=6)
    p.add_argument("--shape-max", type=int, default=12)
    p.add_argument("--world-size", type=int, default=24)
    p.add_argument("--min-support", type=int, default=2)
    p.add_argument("--max-support", type=int, default=5)
    p.add_argument("--max-pair-retries", type=int, default=100)
    p.add_argument("--max-dataset-attempts-per-task", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmups", type=int, default=0)
    p.add_argument("--jsonl-out", default=None,
                   help="Optional path to write run and summary JSON lines.")
    p.add_argument("--no-dominant", action="store_true",
                   help="Disable dominant sampling to isolate sampler/scene cost.")
    p.add_argument("--dominant-pool-dir", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.repeats < 1:
        raise ValueError(f"--repeats must be >= 1, got {args.repeats}")
    if args.warmups < 0:
        raise ValueError(f"--warmups must be >= 0, got {args.warmups}")

    out_fh = None
    if args.jsonl_out is not None:
        out_path = Path(args.jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("w", encoding="utf-8")

    records: list[dict[str, object]] = []
    try:
        total = args.warmups + args.repeats
        for raw_repeat in range(total):
            warmup = raw_repeat < args.warmups
            repeat = raw_repeat if warmup else raw_repeat - args.warmups
            record = _run_once(args, repeat=repeat, warmup=warmup)
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            if out_fh is not None:
                out_fh.write(line + "\n")
                out_fh.flush()
            if not warmup:
                records.append(record)
        if records:
            summary = _summary(args, records)
            line = json.dumps(summary, sort_keys=True)
            print(line, flush=True)
            if out_fh is not None:
                out_fh.write(line + "\n")
                out_fh.flush()
    finally:
        if out_fh is not None:
            out_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
