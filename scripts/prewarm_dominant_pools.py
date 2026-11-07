#!/usr/bin/env python
"""Prebuild append-only filter dominant libraries for benchmark suites."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babytheorist.benchmark import (  # noqa: E402
    benchmark_mode_default_n_shapes,
    resolve_primitive_curriculum,
)
from babytheorist.generation.config import (  # noqa: E402
    LevelConfig, TaskConfig, _dominant_pool_max_depth,
)
from babytheorist.generation.dominant import (  # noqa: E402
    _ABSTRACT_POOL_CACHE, _ensure_dominant_library, _merge_pool_segments,
)
from shape_profiles import shape_profile_defaults  # noqa: E402
from babytheorist.version import BENCHMARK_SPEC_VERSION, __version__ as BABYTHEORIST_VERSION  # noqa: E402


SUITES = (
    "multi_dev", "single_a", "multi_a", "single_b", "multi_b",
)


def _resolve_cache_dir(raw: Path | None) -> Path:
    cache_dir = raw or Path("data") / f"dominant_pools-v{BABYTHEORIST_VERSION}"
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return cache_dir


def _pool_size_by_depth(pool, max_depth: int) -> dict[str, int]:
    return {
        str(depth): len(pool.get(depth, {}))
        for depth in range(1, max_depth + 1)
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _official_task_config(suite: str) -> TaskConfig:
    defaults = shape_profile_defaults("standard")
    n_shapes = benchmark_mode_default_n_shapes(suite)
    if n_shapes is None:
        n_shapes = 2
    return TaskConfig(
        mode="train",
        world_size=15,
        gap=2,
        n_shapes_range=(n_shapes, n_shapes),
        shape_size_range=tuple(defaults["shape_size_range"]),
        shape_compactness=float(defaults["shape_compactness"]),
        shape_bordered_hollow_min_size=defaults[
            "shape_bordered_hollow_min_size"
        ],
        shape_bordered_hollow_random_inner_size=bool(
            defaults["shape_bordered_hollow_random_inner_size"]
        ),
        shape_max_bbox_span=defaults["shape_max_bbox_span"],
    )


def _parse_suites(raw: str) -> list[str]:
    suites = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown suite(s): {', '.join(unknown)}"
        )
    return suites


def _prewarm_suite(
    suite: str,
    *,
    curriculum_level: int,
    max_depth: int,
    pool_seed: int,
    cache_dir: Path,
) -> dict:
    curriculum = resolve_primitive_curriculum(suite, curriculum_level)
    level_cfg = LevelConfig(
        level=curriculum_level,
        fixed_depth=max_depth,
        primitives=list(curriculum.effective_primitives),
        selectors=curriculum.selectors,
        use_dominant=True,
        curriculum=curriculum,
    )
    selector_types = frozenset(level_cfg.selectors)
    vocab = frozenset(level_cfg.primitives)
    before_files = set(cache_dir.glob("abstract_library_*.json"))
    started = time.perf_counter()

    library = _ensure_dominant_library(
        level_cfg,
        curriculum.n_shapes,
        selector_types,
        vocab,
        pool_seed=pool_seed,
        dominant_pool_dir=cache_dir,
        task_cfg=_official_task_config(suite),
    )
    active_segment_names = curriculum.inherited_segments + (
        (curriculum.current_intro_segment,)
        if curriculum.current_intro_segment else ()
    )
    active_pool = _merge_pool_segments(library, active_segment_names)
    after_files = set(cache_dir.glob("abstract_library_*.json"))
    written_files = sorted(_display_path(path) for path in after_files - before_files)

    return {
        "suite": suite,
        "curriculum_level": curriculum_level,
        "max_depth": _dominant_pool_max_depth(level_cfg),
        "pool_seed": pool_seed,
        "n_shapes": curriculum.n_shapes,
        "introduced_primitive": curriculum.introduced_primitive,
        "effective_primitives": list(curriculum.effective_primitives),
        "inherited_segments": list(curriculum.inherited_segments),
        "current_intro_segment": curriculum.current_intro_segment,
        "active_pool_size_by_depth": _pool_size_by_depth(
            active_pool, _dominant_pool_max_depth(level_cfg),
        ),
        "segment_pool_size_by_depth": {
            name: _pool_size_by_depth(pool, _dominant_pool_max_depth(level_cfg))
            for name, pool in sorted(library.segments.items())
        },
        "written_files": written_files,
        "cache_result": "write" if written_files else "hit",
        "wall_sec": round(time.perf_counter() - started, 3),
    }


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Prebuild append-only BabyTheorist dominant libraries.",
    )
    p.add_argument(
        "--suites",
        type=_parse_suites,
        default=list(SUITES),
        help="Comma-separated suites. Default: " + ",".join(SUITES),
    )
    p.add_argument(
        "--level",
        type=int,
        default=7,
        help="Curriculum level used to resolve primitive unlocks.",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Maximum direct dominant entry depth to prebuild.",
    )
    p.add_argument("--pool-seed", type=int, default=42)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Dominant library cache directory. Defaults to data/dominant_pools-v<version>.",
    )
    p.add_argument("--summary-out", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.level < 1:
        print("error: --level must be >= 1", file=sys.stderr)
        return 2
    if args.max_depth < 1:
        print("error: --max-depth must be >= 1", file=sys.stderr)
        return 2

    cache_dir = _resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _ABSTRACT_POOL_CACHE.clear()
    started = time.perf_counter()
    summary = {
        "babytheorist_version": BABYTHEORIST_VERSION,
        "benchmark_spec_version": BENCHMARK_SPEC_VERSION,
        "cache_dir": _display_path(cache_dir),
        "suites": [],
    }

    for suite in args.suites:
        suite_summary = _prewarm_suite(
            suite,
            curriculum_level=args.level,
            max_depth=args.max_depth,
            pool_seed=args.pool_seed,
            cache_dir=cache_dir,
        )
        summary["suites"].append(suite_summary)
        print(
            f"{suite}: depth<= {suite_summary['max_depth']} "
            f"{suite_summary['cache_result']} wall={suite_summary['wall_sec']}s",
            flush=True,
        )

    summary["total_wall_sec"] = round(time.perf_counter() - started, 3)
    if args.summary_out is not None:
        summary_out = args.summary_out
        if not summary_out.is_absolute():
            summary_out = ROOT / summary_out
        summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
