"""Command-line interface for BabyTheorist dataset generation."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from babytheorist.primitives import PRIMITIVES
from babytheorist.dominant import (
    abstract_program_to_strings,
)
from babytheorist.program import (
    build_bigram, build_pool,
)
from babytheorist.review import save
from babytheorist.version import (
    BENCHMARK_SPEC_VERSION,
    __version__ as BABYTHEORIST_VERSION,
    default_dataset_release_version,
)
from babytheorist.benchmark import (
    _BENCHMARK_MODES,
    _DEFAULT_SELECTORS,
    _benchmark_mode_default_n_shapes,
    _normalize_benchmark_mode,
    _resolve_primitive_curriculum,
)
from shape_profiles import (
    SHAPE_PROFILE_STANDARD as _SHAPE_PROFILE_STANDARD,
    SHAPE_PROFILES as _SHAPE_PROFILES,
    shape_profile_defaults as _shape_profile_defaults,
)

from babytheorist.generation.config import (
    LevelConfig,
    TaskConfig,
    _ATTEMPT_SCHEDULERS,
    _DEPTH_PRIOR_PIECEWISE,
    _FILTER_SYSTEM,
    _MODES,
    _depth_center,
    _depth_distribution_params,
    _depth_support_max,
    _dominant_pool_max_depth,
)
from babytheorist.generation.dominant import (
    FilterGenerationStats,
    _curriculum_metadata,
    _dominant_viability_metadata,
    _ensure_abstract_pool,
    _ensure_dominant_library,
    _merge_pool_segments,
)
from babytheorist.generation.filter_dataset import generate_filter_dataset
from babytheorist.generation.filter_pairs import (
    _filter_grounding_version,
    _filter_rules_version,
)
from babytheorist.generation.rejection import generate_dataset, load_excluded_pairs

def _git_output(args: list[str]) -> str | None:
    """Return stripped git output, or None outside a git checkout."""
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip()


def _git_revision_info() -> dict[str, object]:
    """Best-effort git revision metadata for generated artifacts."""
    commit = _git_output(["rev-parse", "--short", "HEAD"])
    dirty = _git_output(["status", "--porcelain"])
    return {
        "git_commit": commit,
        "git_dirty": bool(dirty) if dirty is not None else None,
    }


def _build_cli():
    import argparse

    p = argparse.ArgumentParser(
        prog="generate.py",
        description="Generate a BabyTheorist benchmark dataset for one curriculum level.",
    )
    p.add_argument("--level", type=int, default=1,
                   help="Curriculum level id. Depth is sampled from the "
                        "official prior centered at level+1.")
    p.add_argument("--benchmark-mode", type=str, default=None,
                   help="Benchmark preset. Official suites are multi_dev, "
                        "single_a, multi_a, single_b, multi_b. Hyphenated "
                        "forms such as single-A are accepted, and legacy "
                        "single_easy/multi_easy/single_hard/multi_hard names "
                        "remain aliases. A suites use "
                        "base primitives only; multi_dev uses translate-only "
                        "primitives; B suites unlock hollow at "
                        "L3, fill_interior at L5, and add_border at L7. "
                        "Explicit --primitives/--selectors override the vocabulary.")
    p.add_argument("--primitives", type=str, default=None,
                   help="Comma-separated primitive names (default: all 14, "
                        "or the selected --benchmark-mode preset).")
    p.add_argument("--selectors", type=str, default=None,
                   help="Comma-separated selector types (default: ALL,by_index, "
                        "or the selected --benchmark-mode preset).")
    p.add_argument("--n", type=int, default=100, help="Number of tasks (>= 1).")
    p.add_argument("--mode", type=str, default="train", choices=list(_MODES),
                   help="'train' = 1 pair per task; 'eval' = demo + query + Y_q.")
    p.add_argument("--max-support", type=int, default=5,
                   help="Eval only: maximum adaptive support observations "
                        "before rejecting a task. Default 5.")
    p.add_argument("--min-support", type=int, default=2,
                   help="Eval only: minimum supports per task; the adaptive "
                        "loop starts at K=min_support instead of 1. Tasks "
                        "whose program is unique with fewer supports are "
                        "padded up to this floor. Default 2.")
    p.add_argument("--allow-ambiguous-eval", dest="eval_unique_support",
                   action="store_false", default=True,
                   help="Eval only: keep the legacy K=1 behavior where any "
                        "output in non-singleton Y_q is accepted.")
    p.add_argument("--n-shapes", type=int, default=None,
                   help="Force exactly N shapes per pair. If omitted with an "
                        "official benchmark mode, defaults to multi_dev=2, "
                        "single_a=1, single_b=1, multi_a=2, multi_b=3; "
                        "otherwise samples 2..4.")
    p.add_argument("--shape-profile", type=str, default=_SHAPE_PROFILE_STANDARD,
                   choices=_SHAPE_PROFILES,
                   help="Shape sampler profile. Defaults to 'standard': "
                        "size_range=(4,16), compactness=8, and bordered/hollow "
                        "minimum target size 9 with randomized bordered/hollow "
                        "inner walk sizing. Explicit shape flags override "
                        "individual profile defaults.")
    p.add_argument("--shape-min", type=int, default=None,
                   help="Minimum shape size. Defaults to the selected "
                        "--shape-profile.")
    p.add_argument("--shape-max", type=int, default=None,
                   help="Maximum shape size. Defaults to the selected "
                        "--shape-profile.")
    p.add_argument("--shape-compactness", type=float, default=None,
                   help="Random-walk gravity factor for shape silhouettes. "
                        "Defaults to the selected --shape-profile.")
    p.add_argument("--shape-border-min", type=int, default=None,
                   help="Minimum target size for bordered/hollow shapes. "
                        "Defaults to the selected --shape-profile; use 0 to "
                        "disable the profile minimum.")
    p.add_argument("--shape-max-bbox-span", type=int, default=None,
                   help="Maximum sampled object bounding-box height/width. "
                        "Defaults to the selected --shape-profile; use 0 to "
                        "disable the profile cap.")
    p.add_argument("--world-size", type=int, default=15)
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--weights", type=str, default=None, help="Inline JSON sampling weights.")
    p.add_argument("--no-dominant", "--no-frequent-subprograms",
                   dest="use_dominant", action="store_false", default=True,
                   help="Skip the frequent-subprogram pool; sample each primitive "
                        "directly from the bigram. --no-dominant is kept as "
                        "the legacy compatibility spelling.")
    p.add_argument("--backend", type=str, default="filter",
                   choices=("rejection", "filter"),
                   help="'filter' (default) = constraint-forward symbolic "
                        "rule stack (R1/R3/R5/R5b/R6/R7' + I1). "
                        "'rejection' = legacy rejection sampler with "
                        "scene shortcut checks.")
    p.add_argument("--initial-hollow-prob", type=float, default=0.0,
                   help="(filter backend only) per-shape probability of "
                        "starting hollow. Default 0.0 = every shape solid.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel worker processes (1 = sequential). "
                        "Output is reproducible for a fixed seed and n.")
    p.add_argument("--attempt-scheduler", type=str, default="ordered",
                   choices=_ATTEMPT_SCHEDULERS,
                   help="Filter-backend attempt result ordering used when "
                        "--workers > 1. 'ordered' preserves the legacy seed "
                        "order. 'unordered' consumes completed attempts first "
                        "to reduce long-tail head-of-line blocking and is an "
                        "experimental sampling-policy change.")
    p.add_argument("--attempt-timeout-sec", type=float, default=None,
                   help="Filter backend only: optional wall-clock timeout per "
                        "dataset attempt. Timed-out attempts are rejected and "
                        "recorded in metadata. Default unset preserves legacy "
                        "behavior.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pool-seed", type=int, default=None,
                   help="Separate seed for build_pool (default: same as --seed). "
                        "Use to keep the frequent-subprogram pool identical "
                        "across runs that differ only in task-sampling seed.")
    p.add_argument("--dominant-pool-dir", "--frequent-subprogram-pool-dir",
                   dest="dominant_pool_dir", type=str, default=None,
                   help="Directory for persistent abstract frequent-subprogram "
                        "JSON caches. The legacy --dominant-pool-dir spelling "
                        "and dominant_* metadata keys are kept for compatibility. "
                        "If a matching cache exists it is loaded; otherwise the "
                        "pool is built once and saved there.")
    p.add_argument("--dataset-version", type=str, default=None,
                   help="Dataset release identifier written to metadata. "
                        "Default: BabyTheorist-<benchmark-mode>-v<babytheorist_version>, "
                        "or BabyTheorist-custom-v<babytheorist_version> without a benchmark mode.")
    p.add_argument("--out", type=str, required=True,
                   help="Output path. `.h5`/`.hdf5` writes a task-row HDF5 export "
                        "((N, P, H, W) inputs/outputs); else JSON.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Existing dataset files or directories to dedup against. "
                        "Directories auto-expand to *.h5/*.hdf5/*.json. The "
                        "current run's --out is skipped automatically.")
    p.add_argument("--viz", type=str, default=None, help="Also render a preview PNG here.")
    p.add_argument("--viz-max", type=int, default=12)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    try:
        if args.benchmark_mode is not None:
            args.benchmark_mode = _normalize_benchmark_mode(args.benchmark_mode)
        curriculum = (
            _resolve_primitive_curriculum(args.benchmark_mode, args.level)
            if args.benchmark_mode is not None else None
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.benchmark_mode is None:
        preset_primitives = list(PRIMITIVES)
        preset_selectors = _DEFAULT_SELECTORS
    else:
        preset_primitives = list(curriculum.effective_primitives)
        preset_selectors = curriculum.selectors

    primitives = (
        [p.strip() for p in args.primitives.split(",") if p.strip()]
        if args.primitives is not None else preset_primitives
    )
    selectors = (
        tuple(s.strip() for s in args.selectors.split(",") if s.strip())
        if args.selectors is not None else preset_selectors
    )
    try:
        weights = json.loads(args.weights) if args.weights else {}
        primitive_weights = weights.get("primitive_weights") or None
        bigram_weights = (
            {(a, b): float(v) for a, b, v in weights.get("bigram_weights", [])} or None
        )
        pool_seed = args.pool_seed if args.pool_seed is not None else args.seed
        level_curriculum = (
            curriculum
            if args.primitives is None and args.selectors is None
            else None
        )
        level_cfg = LevelConfig(
            level=args.level, primitives=primitives, selectors=selectors,
            use_dominant=args.use_dominant,
            primitive_weights=primitive_weights,
            curriculum=level_curriculum,
        )
        depth_center = _depth_center(level_cfg.level)
        depth_support_max = _depth_support_max(level_cfg)
        dominant_pool_max_depth = _dominant_pool_max_depth(level_cfg)
        default_n_shapes = _benchmark_mode_default_n_shapes(args.benchmark_mode)
        n_shapes_range = (
            (args.n_shapes, args.n_shapes)
            if args.n_shapes is not None
            else (
                (default_n_shapes, default_n_shapes)
                if default_n_shapes is not None
                else (2, 4)
            )
        )
        shape_defaults = _shape_profile_defaults(args.shape_profile)
        default_shape_min, default_shape_max = shape_defaults["shape_size_range"]
        shape_min = (
            args.shape_min if args.shape_min is not None else default_shape_min
        )
        shape_max = (
            args.shape_max if args.shape_max is not None else default_shape_max
        )
        shape_compactness = (
            args.shape_compactness
            if args.shape_compactness is not None
            else float(shape_defaults["shape_compactness"])
        )
        if args.shape_border_min is None:
            shape_bordered_hollow_min_size = shape_defaults[
                "shape_bordered_hollow_min_size"
            ]
        elif args.shape_border_min == 0:
            shape_bordered_hollow_min_size = None
        else:
            shape_bordered_hollow_min_size = args.shape_border_min
        if args.shape_max_bbox_span is None:
            shape_max_bbox_span = shape_defaults["shape_max_bbox_span"]
        elif args.shape_max_bbox_span == 0:
            shape_max_bbox_span = None
        else:
            shape_max_bbox_span = args.shape_max_bbox_span
        task_cfg = TaskConfig(
            mode=args.mode, world_size=args.world_size, gap=args.gap,
            n_shapes_range=n_shapes_range,
            shape_size_range=(shape_min, shape_max),
            shape_compactness=shape_compactness,
            shape_bordered_hollow_min_size=shape_bordered_hollow_min_size,
            shape_bordered_hollow_random_inner_size=bool(
                shape_defaults["shape_bordered_hollow_random_inner_size"]
            ),
            shape_max_bbox_span=shape_max_bbox_span,
            eval_unique_support=args.eval_unique_support,
            max_support=args.max_support,
            min_support=args.min_support,
        )
        dataset_release_version = (
            args.dataset_version
            if args.dataset_version is not None
            else default_dataset_release_version(args.benchmark_mode)
        )
        extra_info: dict | None = {
            "babytheorist_version": BABYTHEORIST_VERSION,
            "benchmark_spec_version": BENCHMARK_SPEC_VERSION,
            "dataset_release_version": dataset_release_version,
            "generator_backend": args.backend,
            "filter_system": (
                _FILTER_SYSTEM if args.backend == "filter" else None
            ),
            "filter_rules_version": (
                _filter_rules_version()
                if args.backend == "filter" else None
            ),
            "filter_grounding_version": (
                _filter_grounding_version()
                if args.backend == "filter" else None
            ),
            **_git_revision_info(),
            "mode": args.benchmark_mode,
            "benchmark_suite": (
                curriculum.suite if curriculum is not None else None
            ),
            "primitive_vocabulary": list(level_cfg.primitives),
            "selector_vocabulary": list(level_cfg.selectors),
            "depth_distribution": _DEPTH_PRIOR_PIECEWISE,
            "depth_center": depth_center,
            "depth_support": [1, depth_support_max],
            "depth_distribution_params": _depth_distribution_params(level_cfg),
            "eval_policy": (
                "unique_support" if task_cfg.eval_unique_support else "ambiguous_yq"
            ),
            "max_support": task_cfg.max_support,
            "min_support": task_cfg.min_support,
            "shape_generation": {
                "profile": args.shape_profile,
                "shape_size_range": list(task_cfg.shape_size_range),
                "shape_compactness": task_cfg.shape_compactness,
                "shape_bordered_hollow_min_size": (
                    task_cfg.shape_bordered_hollow_min_size
                ),
                "shape_bordered_hollow_random_inner_size": (
                    task_cfg.shape_bordered_hollow_random_inner_size
                ),
                "shape_max_bbox_span": task_cfg.shape_max_bbox_span,
            },
        }
        if args.backend == "filter" and args.attempt_scheduler != "ordered":
            extra_info["attempt_scheduler"] = args.attempt_scheduler
        if args.backend == "filter" and args.attempt_timeout_sec is not None:
            extra_info["attempt_timeout_sec"] = args.attempt_timeout_sec
        if level_cfg.curriculum is not None:
            extra_info.update(_curriculum_metadata(level_cfg.curriculum))
        seen_pair_hashes = load_excluded_pairs(args.exclude, args.out)
        if seen_pair_hashes:
            print(f"excluding {len(seen_pair_hashes)} existing pairs")
        if args.backend == "filter":
            filter_run_stats = FilterGenerationStats()
            tasks = generate_filter_dataset(
                level_cfg, task_cfg, n_tasks=args.n, seed=args.seed,
                initial_hollow_prob=args.initial_hollow_prob,
                n_workers=args.workers,
                attempt_scheduler=args.attempt_scheduler,
                attempt_timeout_sec=args.attempt_timeout_sec,
                bigram_weights=bigram_weights,
                pool_seed=pool_seed,
                dominant_pool_dir=args.dominant_pool_dir,
                seen_pair_hashes=seen_pair_hashes,
                stats=filter_run_stats,
            )
            extra_info["filter_generation_stats"] = (
                filter_run_stats.generation_dict()
            )
            extra_info["filter_reject_stats"] = filter_run_stats.reject_dict()
            if args.use_dominant:
                n_shapes_meta = (
                    args.n_shapes if args.n_shapes else task_cfg.n_shapes_range[1]
                )
                if level_cfg.curriculum is not None:
                    meta_library = _ensure_dominant_library(
                        level_cfg,
                        n_shapes=n_shapes_meta,
                        selector_types=frozenset(selectors),
                        vocab=frozenset(primitives),
                        bigram_weights=bigram_weights,
                        pool_seed=pool_seed,
                        dominant_pool_dir=args.dominant_pool_dir,
                        task_cfg=task_cfg,
                    )
                    meta_pool = _merge_pool_segments(
                        meta_library,
                        level_cfg.curriculum.inherited_segments
                        + (
                            (level_cfg.curriculum.current_intro_segment,)
                            if level_cfg.curriculum.current_intro_segment else ()
                        ),
                    )
                    segment_sizes = {
                        name: {str(d): len(entries) for d, entries in pool.items()}
                        for name, pool in sorted(meta_library.segments.items())
                    }
                else:
                    meta_initial_grid_shapes = (
                        ["hollow"] * n_shapes_meta
                        if args.initial_hollow_prob == 1.0 else None
                    )
                    _meta_bigram, meta_pool = _ensure_abstract_pool(
                        level_cfg,
                        n_shapes=n_shapes_meta,
                        selector_types=frozenset(selectors),
                        vocab=frozenset(primitives),
                        bigram_weights=bigram_weights,
                        initial_grid_shapes=meta_initial_grid_shapes,
                        pool_seed=pool_seed,
                        dominant_pool_dir=args.dominant_pool_dir,
                    )
                    segment_sizes = None
                reps = [
                    abstract_program_to_strings(p)
                    for p in list(meta_pool.get(dominant_pool_max_depth, {}))[:10]
                ]
                slot_kind_counts = Counter(
                    ref.kind
                    for entries in meta_pool.values()
                    for prog in entries
                    for step in prog
                    for ref in (
                        (step.source,)
                        if step.target is None
                        else (step.source, step.target)
                    )
                )
                dominant_meta = {
                    "dominant_system": "abstract_slots_v1",
                    "dominant_grounding_system": (
                        _filter_grounding_version()
                    ),
                    "dominant_kind": "abstract_slots",
                    "dominant_pool_seed": pool_seed,
                    "dominant_pool_size": {
                        str(d): len(entries) for d, entries in meta_pool.items()
                    },
                    "dominant_pool_max_depth": dominant_pool_max_depth,
                    "dominant_pool_dir": args.dominant_pool_dir,
                    "dominant_viability": _dominant_viability_metadata(task_cfg),
                    "dominant_p_direct": 0.5,
                    "dominant_depths": list(range(1, dominant_pool_max_depth + 1)),
                    "dominant_programs": reps,
                    "dominant_slot_kinds": dict(slot_kind_counts),
                }
                if segment_sizes is not None:
                    dominant_meta["dominant_segment_sizes"] = segment_sizes
                extra_info.update(dominant_meta)
        else:
            # Build bigram/pool once so generation and the saved metadata
            # share the exact same pool object.
            bigram = build_bigram(primitives, bigram_weights=bigram_weights)
            pool = (
                build_pool(bigram, primitives, max_depth=depth_support_max,
                           primitive_weights=primitive_weights,
                           rng=np.random.default_rng(pool_seed))
                if args.use_dominant else None
            )
            tasks = generate_dataset(
                level_cfg, task_cfg, n_tasks=args.n, seed=args.seed,
                bigram=bigram, pool=pool, n_workers=args.workers,
                seen_pair_hashes=seen_pair_hashes,
            )
            if pool is not None:
                dominant_programs = [
                    list(p) for p in pool.get(depth_support_max, {})
                ]
                prim_in_pool = Counter(p for prog in dominant_programs for p in prog)
                extra_info.update({
                    "dominant_primitives": dict(prim_in_pool.most_common()),
                    "dominant_programs": dominant_programs,
                })
        program_hist = Counter(
            step.primitive for task in tasks for step in task.program
        )
        extra_info["program_primitive_histogram"] = dict(
            sorted(program_hist.items())
        )
        save(tasks, args.out, extra_info=extra_info)
    except (ValueError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"saved {len(tasks)} tasks to {args.out}")

    if args.viz is not None:
        from utils.viz import render_dataset
        render_dataset(tasks, path=args.viz, max_tasks=args.viz_max)
        print(f"saved preview of {min(len(tasks), args.viz_max)} tasks to {args.viz}")

    return 0


__all__ = [
    "_git_output",
    "_git_revision_info",
    "_build_cli",
    "main",
]
