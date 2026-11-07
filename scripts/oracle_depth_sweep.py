"""Validate production-filter samples with a scene-level DP shortcut oracle.

This is an analysis tool, not part of normal generation.  For each sampled
task pair, it searches all programs shorter than the sampled depth by BFS over
rendered engine states, merging duplicate states as dynamic programming.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from babytheorist.benchmark import (
    benchmark_mode_default_n_shapes,
    resolve_primitive_curriculum,
)
from babytheorist.generation.config import LevelConfig, TaskConfig, _FILTER_PAIR_EXECUTION_RETRIES
from babytheorist.generation.filter_dataset import _sample_filter_program_with_optional_dominant
from babytheorist.generation.filter_pairs import (
    _sample_filter_pair_with_reason, _with_raster_order_constraint,
)
from babytheorist.generation.rejection import (
    _enum_q_steps,
)
from babytheorist.core import Program, Shape, Step, TaskPair
from babytheorist.engine import _rebuild_grid, apply_step
from babytheorist.filter.constraint import InfeasibleConstraint
from babytheorist.program import SamplerDeadEnd
from babytheorist.shapes import ShapeFactory


@dataclass
class PairSample:
    program: Program
    pair: TaskPair
    input_shapes: list[Shape]
    generated_attempts: int
    pair_failures: Counter[str]


@dataclass
class OracleResult:
    shortcut: bool | None
    shortcut_depth: int | None
    shortcut_program: Program | None
    budget_exceeded: str | None
    seen_states: int
    expanded_states: int
    step_attempts: int
    elapsed_sec: float


_WORKER = SimpleNamespace(args=None, depth=None, step_options=None)


def _shape_key(shape: Shape) -> tuple[tuple[int, int], tuple[int, int], bytes]:
    return shape.position, tuple(shape.grid.shape), shape.grid.tobytes()


def _state_key(shapes: list[Shape]) -> tuple[tuple[tuple[int, int], tuple[int, int], bytes], ...]:
    return tuple(_shape_key(shape) for shape in shapes)


def _output_matches(
    shapes: list[Shape],
    output_grid: np.ndarray,
    world_size: int,
) -> bool:
    grid = _rebuild_grid(shapes, world_size)
    return grid is not None and np.array_equal(grid, output_grid)


def scene_has_shortcut_dp(
    *,
    output_grid: np.ndarray,
    input_shapes: list[Shape],
    world_size: int,
    depth: int,
    step_options: list[Step],
    max_states: int,
    max_oracle_sec: float,
    record_shortcut_program: bool = False,
) -> OracleResult:
    """Search for any shorter program using BFS over distinct engine states."""
    t0 = time.perf_counter()
    max_depth = max(0, depth - 1)
    initial = [shape.copy() for shape in input_shapes]

    if _output_matches(initial, output_grid, world_size):
        return OracleResult(
            shortcut=True,
            shortcut_depth=0,
            shortcut_program=[],
            budget_exceeded=None,
            seen_states=1,
            expanded_states=0,
            step_attempts=0,
            elapsed_sec=time.perf_counter() - t0,
        )

    visited = {_state_key(initial)}
    frontier = [initial]
    programs: dict[object, tuple[Step, ...]] = (
        {_state_key(initial): ()} if record_shortcut_program else {}
    )
    expanded_states = 0
    step_attempts = 0

    for current_depth in range(max_depth):
        next_frontier: list[list[Shape]] = []
        for state in frontier:
            if max_oracle_sec > 0 and time.perf_counter() - t0 > max_oracle_sec:
                return OracleResult(
                    shortcut=None,
                    shortcut_depth=None,
                    budget_exceeded="time",
                    seen_states=len(visited),
                    expanded_states=expanded_states,
                    step_attempts=step_attempts,
                    elapsed_sec=time.perf_counter() - t0,
                )
            expanded_states += 1
            state_key = _state_key(state)
            prefix = programs.get(state_key, ()) if record_shortcut_program else ()
            for step in step_options:
                step_attempts += 1
                next_state = apply_step(step, state, world_size)
                if next_state is None:
                    continue
                key = _state_key(next_state)
                if key in visited:
                    continue
                visited.add(key)
                if record_shortcut_program:
                    programs[key] = (*prefix, step)
                if len(visited) > max_states:
                    return OracleResult(
                        shortcut=None,
                        shortcut_depth=None,
                        shortcut_program=None,
                        budget_exceeded="states",
                        seen_states=len(visited),
                        expanded_states=expanded_states,
                        step_attempts=step_attempts,
                        elapsed_sec=time.perf_counter() - t0,
                    )
                if _output_matches(next_state, output_grid, world_size):
                    return OracleResult(
                        shortcut=True,
                        shortcut_depth=current_depth + 1,
                        shortcut_program=(
                            list(programs[key])
                            if record_shortcut_program else None
                        ),
                        budget_exceeded=None,
                        seen_states=len(visited),
                        expanded_states=expanded_states,
                        step_attempts=step_attempts,
                        elapsed_sec=time.perf_counter() - t0,
                    )
                next_frontier.append(next_state)
        frontier = next_frontier
        if not frontier:
            break

    return OracleResult(
        shortcut=False,
        shortcut_depth=None,
        shortcut_program=None,
        budget_exceeded=None,
        seen_states=len(visited),
        expanded_states=expanded_states,
        step_attempts=step_attempts,
        elapsed_sec=time.perf_counter() - t0,
    )


def _build_level_cfg(args, depth: int) -> LevelConfig:
    curriculum = resolve_primitive_curriculum(args.benchmark_mode, args.level)
    return LevelConfig(
        level=args.level,
        primitives=list(curriculum.effective_primitives),
        selectors=curriculum.selectors,
        fixed_depth=depth,
        palette=tuple(range(1, 10)),
        use_dominant=not args.no_dominant,
        curriculum=curriculum if not args.no_dominant else None,
    )


def _build_task_cfg(args, n_shapes: int) -> TaskConfig:
    return TaskConfig(
        mode="train",
        n_shapes_range=(n_shapes, n_shapes),
        world_size=args.world_size,
        shape_size_range=(args.shape_min, args.shape_max),
        max_program_retries=args.max_program_retries,
        max_pair_retries=args.max_pair_retries,
        max_dataset_attempts_per_task=args.max_dataset_attempts_per_task,
        gap=args.gap,
    )


def _sample_pair(
    args,
    *,
    depth: int,
    rng: np.random.Generator,
) -> PairSample | None:
    level_cfg = _build_level_cfg(args, depth)
    n_shapes = (
        args.n_shapes
        if args.n_shapes > 0
        else benchmark_mode_default_n_shapes(args.benchmark_mode)
    )
    task_cfg = _build_task_cfg(args, n_shapes)
    vocab = frozenset(level_cfg.primitives)
    selector_types = frozenset(level_cfg.selectors)
    uses_by_index = "by_index" in selector_types
    factory = ShapeFactory(size_range=task_cfg.shape_size_range)
    pair_failures: Counter[str] = Counter()

    for attempt in range(1, args.max_dataset_attempts_per_task + 1):
        initial_grid_shapes = [
            "hollow" if rng.random() < args.initial_hollow_prob else "solid"
            for _ in range(n_shapes)
        ]
        try:
            program, constraint = _sample_filter_program_with_optional_dominant(
                level_cfg,
                depth,
                n_shapes,
                vocab,
                selector_types,
                rng,
                initial_grid_shapes,
                pool_seed=args.pool_seed,
                dominant_pool_dir=args.dominant_pool_dir,
            )
            if uses_by_index:
                constraint = _with_raster_order_constraint(constraint)
        except (SamplerDeadEnd, InfeasibleConstraint):
            pair_failures["program_dead_end"] += 1
            continue

        pair, reason = _sample_filter_pair_with_reason(
            constraint,
            program,
            factory,
            level_cfg,
            task_cfg,
            rng,
            uses_by_index=uses_by_index,
            gap=task_cfg.gap,
            execution_retries=_FILTER_PAIR_EXECUTION_RETRIES,
        )
        if pair is None:
            pair_failures[str(reason)] += 1
            continue
        task_pair, input_shapes = pair
        return PairSample(
            program=program,
            pair=task_pair,
            input_shapes=input_shapes,
            generated_attempts=attempt,
            pair_failures=pair_failures,
        )
    return None


def _parse_depths(raw: str) -> list[int]:
    depths = [int(part) for part in raw.split(",") if part.strip()]
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError(f"--depths must contain positive ints, got {raw!r}")
    return depths


def _record_config(args) -> dict[str, object]:
    keys = (
        "benchmark_mode",
        "dominant_pool_dir",
        "gap",
        "initial_hollow_prob",
        "indices",
        "level",
        "max_dataset_attempts_per_task",
        "max_oracle_sec",
        "max_pair_retries",
        "max_program_retries",
        "max_states",
        "n_per_depth",
        "n_shapes",
        "no_dominant",
        "pool_seed",
        "record_shortcut_program",
        "seed",
        "shape_max",
        "shape_min",
        "world_size",
        "workers",
    )
    return {key: getattr(args, key) for key in keys}


def _seed_for(args, depth: int, index: int) -> int:
    ss = np.random.SeedSequence([int(args.seed), int(depth), int(index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])


def _parse_indices(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    indices = [int(part) for part in raw.split(",") if part.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError(f"--indices must contain non-negative ints, got {raw!r}")
    return indices


def _validate_one(args, depth: int, index: int, step_options: list[Step]) -> dict[str, object]:
    rng = np.random.default_rng(_seed_for(args, depth, index))
    t_generate = time.perf_counter()
    sample = _sample_pair(args, depth=depth, rng=rng)
    generate_sec = time.perf_counter() - t_generate
    if sample is None:
        return {
            "index": index,
            "status": "sample_none",
            "generate_sec": generate_sec,
            "oracle_sec": 0.0,
            "seen_states": 0,
            "expanded_states": 0,
            "step_attempts": 0,
            "sample_retries": {},
            "violation": None,
        }

    oracle = scene_has_shortcut_dp(
        output_grid=sample.pair.output_grid,
        input_shapes=sample.input_shapes,
        world_size=args.world_size,
        depth=depth,
        step_options=step_options,
        max_states=args.max_states,
        max_oracle_sec=args.max_oracle_sec,
        record_shortcut_program=args.record_shortcut_program,
    )
    if oracle.shortcut is True:
        status = "shortcut"
        violation = {
            "index": index,
            "program": [str(step) for step in sample.program],
            "shortcut_depth": oracle.shortcut_depth,
            "shortcut_program": (
                [str(step) for step in oracle.shortcut_program]
                if oracle.shortcut_program is not None else None
            ),
        }
    elif oracle.shortcut is False:
        status = "validated"
        violation = None
    else:
        status = f"budget_{oracle.budget_exceeded}"
        violation = None

    return {
        "index": index,
        "status": status,
        "generate_sec": generate_sec,
        "oracle_sec": oracle.elapsed_sec,
        "seen_states": oracle.seen_states,
        "expanded_states": oracle.expanded_states,
        "step_attempts": oracle.step_attempts,
        "sample_retries": dict(sample.pair_failures),
        "violation": violation,
    }


def _init_worker(args, depth: int, step_options: list[Step]) -> None:
    global _WORKER
    _WORKER = SimpleNamespace(
        args=args,
        depth=depth,
        step_options=step_options,
    )


def _validate_one_worker(index: int) -> dict[str, object]:
    return _validate_one(
        _WORKER.args,
        _WORKER.depth,
        index,
        _WORKER.step_options,
    )


def _oracle_mp_context():
    """Use fork COW on Linux, otherwise respect the platform default."""
    if sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def _iter_depth_results(
    args, depth: int, step_options: list[Step],
) -> list[dict[str, object]]:
    indices = _parse_indices(args.indices)
    if indices is None:
        indices = list(range(args.n_per_depth))
    if args.workers <= 1:
        return [
            _validate_one(args, depth, index, step_options)
            for index in indices
        ]

    ctx = _oracle_mp_context()
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args, depth, step_options),
    ) as pool:
        return list(pool.imap_unordered(_validate_one_worker, indices))


def _run_depth(args, depth: int, out_fh) -> dict[str, object]:
    level_cfg = _build_level_cfg(args, depth)
    n_shapes = (
        args.n_shapes
        if args.n_shapes > 0
        else benchmark_mode_default_n_shapes(args.benchmark_mode)
    )
    step_options = _enum_q_steps(
        tuple(level_cfg.primitives),
        tuple(level_cfg.selectors),
        n_shapes,
        tuple(level_cfg.palette),
    )
    summary: Counter[str] = Counter()
    oracle_sec = 0.0
    generate_sec = 0.0
    seen_states_max = 0
    expanded_states_max = 0
    step_attempts_max = 0
    first_violation: dict[str, object] | None = None

    wall_t0 = time.perf_counter()
    for result in _iter_depth_results(args, depth, step_options):
        summary[str(result["status"])] += 1
        generate_sec += float(result["generate_sec"])
        oracle_sec += float(result["oracle_sec"])
        seen_states_max = max(seen_states_max, int(result["seen_states"]))
        expanded_states_max = max(
            expanded_states_max, int(result["expanded_states"]),
        )
        step_attempts_max = max(step_attempts_max, int(result["step_attempts"]))

        violation = result.get("violation")
        if violation is not None and (
            first_violation is None
            or int(violation["index"]) < int(first_violation["index"])
        ):
            first_violation = violation

        sample_retries = result.get("sample_retries", {})
        for reason, count in sample_retries.items():
            summary[f"sample_retry_{reason}"] += count
    wall_sec = time.perf_counter() - wall_t0

    record = {
        "record_type": "depth_summary",
        "config": _record_config(args),
        "depth": depth,
        "n_shapes": n_shapes,
        "step_options": len(step_options),
        "counts": dict(sorted(summary.items())),
        "generate_sec": generate_sec,
        "oracle_sec": oracle_sec,
        "wall_sec": wall_sec,
        "seen_states_max": seen_states_max,
        "expanded_states_max": expanded_states_max,
        "step_attempts_max": step_attempts_max,
        "first_violation": first_violation,
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if out_fh is not None:
        out_fh.write(line + "\n")
        out_fh.flush()
    return record


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample fixed-depth production-filter tasks and DP-check shortcuts.",
    )
    p.add_argument("--benchmark-mode", default="multi-B")
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--depths", default="1,2,3,4,5")
    p.add_argument("--indices", default=None,
                   help="Comma-separated sample indices to validate instead of 0..n-1.")
    p.add_argument("--n-per-depth", type=int, default=20)
    p.add_argument("--n-shapes", type=int, default=0,
                   help="Use <=0 for the benchmark-mode default.")
    p.add_argument("--shape-min", type=int, default=6)
    p.add_argument("--shape-max", type=int, default=12)
    p.add_argument("--world-size", type=int, default=24)
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--max-program-retries", type=int, default=50)
    p.add_argument("--max-pair-retries", type=int, default=100)
    p.add_argument("--max-dataset-attempts-per-task", type=int, default=30)
    p.add_argument("--max-states", type=int, default=100_000)
    p.add_argument("--max-oracle-sec", type=float, default=30.0,
                   help="Per-task oracle budget; use <=0 for no time budget.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-seed", type=int, default=42)
    p.add_argument("--record-shortcut-program", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--initial-hollow-prob", type=float, default=0.0)
    p.add_argument("--dominant-pool-dir", default=None)
    p.add_argument("--no-dominant", action="store_true")
    p.add_argument("--jsonl-out", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.n_per_depth < 1:
        raise ValueError(f"--n-per-depth must be >= 1, got {args.n_per_depth}")
    if args.max_states < 1:
        raise ValueError(f"--max-states must be >= 1, got {args.max_states}")
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    out_fh = None
    if args.jsonl_out is not None:
        out_path = Path(args.jsonl_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_fh = out_path.open("w", encoding="utf-8")
    try:
        for depth in _parse_depths(args.depths):
            _run_depth(args, depth, out_fh)
    finally:
        if out_fh is not None:
            out_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
