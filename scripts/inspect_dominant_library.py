#!/usr/bin/env python3
"""Print human-readable BabyTheorist abstract dominant libraries."""

from __future__ import annotations

import argparse
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
from babytheorist.dominant import AbstractPool, abstract_program_to_strings  # noqa: E402
from babytheorist.dominant import _compose_programs  # noqa: E402
from babytheorist.version import __version__ as BABYTHEORIST_VERSION  # noqa: E402


SUITES = (
    "multi_dev", "single_a", "multi_a", "single_b", "multi_b",
)
DEFAULT_LEVELS = (1, 2, 4, 6)
DEFAULT_DEPTHS = (1, 2, 3, 5, 10)
COMPOSITION_MODES = ("reuse", "fresh")


def _resolve_cache_dir(raw: Path | None) -> Path:
    cache_dir = raw or Path("data") / f"dominant_pools-v{BABYTHEORIST_VERSION}"
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    return cache_dir


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_suites(raw: str) -> list[str]:
    suites = _parse_csv(raw)
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown suite(s): {', '.join(unknown)}"
        )
    return suites


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


def _program_one_line(program) -> str:
    return " -> ".join(abstract_program_to_strings(program))


def _format_pool_entries(
    pool: AbstractPool,
    *,
    provenance: dict[int, dict] | None,
    depths: tuple[int, ...],
    max_per_depth: int,
    max_parents: int,
    indent: str,
    show_parents: bool,
) -> list[str]:
    lines: list[str] = []
    for depth in depths:
        entries = pool.get(depth, {})
        if not entries:
            lines.append(f"{indent}depth {depth}: <empty>")
            continue
        lines.append(f"{indent}depth {depth}: {len(entries)} entries")
        ranked = sorted(entries.items(), key=lambda kv: -kv[1])[:max_per_depth]
        for idx, (program, score) in enumerate(ranked, start=1):
            record = (
                provenance.get(depth, {}).get(program)
                if provenance is not None else None
            )
            entry_id = record.get("id") if record else f"rank:{idx:02d}"
            lines.append(
                f"{indent}  {idx:02d}. id={entry_id} score={score:.4f} "
                f"{_program_one_line(program)}"
            )
            if show_parents and depth > 1:
                parents = list(record.get("parents", [])) if record else []
                if parents:
                    count = record.get("count")
                    valid_samples = record.get("valid_samples")
                    if count is not None and valid_samples is not None:
                        lines.append(
                            f"{indent}      sample_count: "
                            f"{count}/{valid_samples}"
                        )
                    for parent in parents[:max_parents]:
                        split = int(parent["split"])
                        lines.append(
                            f"{indent}      parent: count={parent['count']} "
                            f"source={parent['source']} depth {split} + "
                            f"{depth - split}, mode={parent['mode']}"
                        )
                        lines.append(
                            f"{indent}        left_id:  {parent['left_id']}"
                        )
                        lines.append(
                            f"{indent}        right_id: {parent['right_id']}"
                        )
                    if len(parents) > max_parents:
                        lines.append(
                            f"{indent}      ... {len(parents) - max_parents} "
                            "more parent paths"
                        )
                else:
                    parent = _find_parent(pool, program, depth)
                    if parent is None:
                        lines.append(f"{indent}      parent: <not tracked>")
                    else:
                        split, mode, left, right = parent
                        lines.append(
                            f"{indent}      parent: <recovered> depth {split} + "
                            f"{depth - split}, mode={mode}"
                        )
                        lines.append(f"{indent}        left:  {_program_one_line(left)}")
                        lines.append(f"{indent}        right: {_program_one_line(right)}")
    return lines


def _find_parent(
    pool: AbstractPool,
    program,
    depth: int,
):
    for split in range(1, depth):
        for left in pool.get(split, {}):
            for right in pool.get(depth - split, {}):
                for mode in COMPOSITION_MODES:
                    if _compose_programs(left, right, mode) == program:
                        return split, mode, left, right
    return None


def _active_segment_names(curriculum) -> tuple[str, ...]:
    return curriculum.inherited_segments + (
        (curriculum.current_intro_segment,)
        if curriculum.current_intro_segment else ()
    )


def _merge_segment_provenance(library, names: tuple[str, ...]):
    merged: dict[int, dict] = {}
    scores: dict[int, dict] = {}
    for name in names:
        segment_provenance = library.provenance.get(name, {})
        for depth, entries in library.segments.get(name, {}).items():
            score_bucket = scores.setdefault(depth, {})
            prov_bucket = merged.setdefault(depth, {})
            for program, score in entries.items():
                previous = score_bucket.get(program)
                if previous is not None and score <= previous:
                    continue
                record = segment_provenance.get(depth, {}).get(program)
                if record is None:
                    continue
                score_bucket[program] = score
                prov_bucket[program] = record
    return merged


def _inspect_one(
    suite: str,
    level: int,
    *,
    pool_seed: int,
    cache_dir: Path | None,
    depths: tuple[int, ...],
    max_per_depth: int,
    max_parents: int,
    direct_max_depth: int | None,
    show_segments: bool,
    show_parents: bool,
) -> str:
    curriculum = resolve_primitive_curriculum(suite, level)
    level_cfg = LevelConfig(
        level=level,
        fixed_depth=direct_max_depth,
        primitives=list(curriculum.effective_primitives),
        selectors=curriculum.selectors,
        use_dominant=True,
        curriculum=curriculum,
    )
    started = time.perf_counter()
    library = _ensure_dominant_library(
        level_cfg,
        curriculum.n_shapes,
        frozenset(curriculum.selectors),
        frozenset(curriculum.effective_primitives),
        pool_seed=pool_seed,
        dominant_pool_dir=cache_dir,
        task_cfg=_official_task_config(suite),
    )
    wall = time.perf_counter() - started
    max_depth = _dominant_pool_max_depth(level_cfg)
    visible_depths = tuple(d for d in depths if d <= max_depth)
    active_names = _active_segment_names(curriculum)
    active_pool = _merge_pool_segments(library, active_names)
    active_provenance = _merge_segment_provenance(library, active_names)

    lines = [
        f"## {suite} level {level}",
        (
            f"n_shapes={curriculum.n_shapes} selectors={list(curriculum.selectors)} "
            f"introduced={curriculum.introduced_primitive} "
            f"max_depth={max_depth} load_or_build_sec={wall:.3f}"
        ),
        f"effective_primitives={list(curriculum.effective_primitives)}",
        f"active_segments={list(active_names)}",
        "",
        "active pool",
    ]
    lines.extend(
        _format_pool_entries(
            active_pool,
            provenance=active_provenance,
            depths=visible_depths,
            max_per_depth=max_per_depth,
            max_parents=max_parents,
            indent="  ",
            show_parents=show_parents,
        )
    )

    if show_segments:
        for name, pool in sorted(library.segments.items()):
            lines.extend(["", f"segment: {name}"])
            lines.extend(
                _format_pool_entries(
                    pool,
                    provenance=library.provenance.get(name, {}),
                    depths=visible_depths,
                    max_per_depth=max_per_depth,
                    max_parents=max_parents,
                    indent="  ",
                    show_parents=show_parents,
                )
            )

    return "\n".join(lines)


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect BabyTheorist abstract dominant programs by suite/level.",
    )
    parser.add_argument(
        "--suites",
        type=_parse_suites,
        default=list(SUITES),
        help="Comma-separated suites. Default: all official suites.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="Levels to inspect. Default: 1 3 5 7.",
    )
    parser.add_argument(
        "--suite-levels",
        type=_parse_csv,
        default=None,
        help=(
            "Comma-separated suite:level pairs. Overrides --suites/--levels, "
            "for example single_a:1,multi_a:1,single_b:7,multi_b:7."
        ),
    )
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEPTHS),
        help="Dominant depths to print. Default: 1 2 3 5 10.",
    )
    parser.add_argument("--max-per-depth", type=int, default=3)
    parser.add_argument("--max-parents", type=int, default=3)
    parser.add_argument("--pool-seed", type=int, default=42)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Dominant cache dir. Default: data/dominant_pools-v<version>.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Build in memory without reading/writing the dominant cache dir.",
    )
    parser.add_argument(
        "--direct-max-depth",
        type=int,
        default=None,
        help="Debug/quick inspection cap. Omit for official direct max depth.",
    )
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="Also print each raw library segment in addition to active pool.",
    )
    parser.add_argument(
        "--show-parents",
        action="store_true",
        help="For printed entries, show one lower-depth parent decomposition.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)
    if args.suite_levels is not None:
        suite_level_pairs: list[tuple[str, int]] = []
        for item in args.suite_levels:
            try:
                suite, level_s = item.split(":", 1)
                level = int(level_s)
            except ValueError as e:
                raise ValueError(
                    "--suite-levels entries must look like suite:level"
                ) from e
            if suite not in SUITES:
                raise ValueError(f"unknown suite in --suite-levels: {suite!r}")
            if level < 1:
                raise ValueError("--suite-levels levels must be >= 1")
            suite_level_pairs.append((suite, level))
    elif not args.levels:
        raise ValueError("--levels must not be empty")
    elif any(level < 1 for level in args.levels):
        raise ValueError("--levels must be >= 1")
    if not args.depths:
        raise ValueError("--depths must not be empty")
    if any(depth < 1 for depth in args.depths):
        raise ValueError("--depths must be >= 1")
    if args.max_per_depth < 1:
        raise ValueError("--max-per-depth must be >= 1")
    if args.max_parents < 1:
        raise ValueError("--max-parents must be >= 1")
    if args.direct_max_depth is not None and args.direct_max_depth < 1:
        raise ValueError("--direct-max-depth must be >= 1")

    cache_dir = None if args.no_cache else _resolve_cache_dir(args.cache_dir)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    _ABSTRACT_POOL_CACHE.clear()

    sections: list[str] = []
    if args.suite_levels is None:
        suite_level_pairs = [
            (suite, level)
            for suite in args.suites
            for level in args.levels
        ]
    for suite, level in suite_level_pairs:
        sections.append(
            _inspect_one(
                suite,
                level,
                pool_seed=args.pool_seed,
                cache_dir=cache_dir,
                depths=tuple(sorted(set(args.depths))),
                max_per_depth=args.max_per_depth,
                max_parents=args.max_parents,
                direct_max_depth=args.direct_max_depth,
                show_segments=args.show_segments,
                show_parents=args.show_parents,
            )
        )
    output = _document_header(args) + "\n\n" + "\n\n".join(sections) + "\n"
    if args.out is not None:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


def _document_header(args: argparse.Namespace) -> str:
    if args.no_cache:
        cache_note = "cache disabled for this inspection run"
    else:
        cache_note = str(_resolve_cache_dir(args.cache_dir))
    lines = [
        "# BabyTheorist Dominant Program Inspection",
        "",
        f"- pool_seed: {args.pool_seed}",
        f"- cache: {cache_note}",
        f"- direct_max_depth: {args.direct_max_depth or 'official'}",
        f"- max_per_depth: {args.max_per_depth}",
        f"- max_parents: {args.max_parents}",
    ]
    if args.suite_levels is not None:
        lines.append(f"- suite-level snapshot: {', '.join(args.suite_levels)}")
        hard_final = {"single_b:6", "multi_b:6"}
        if hard_final & set(args.suite_levels):
            lines.append(
                "- hard-mode note: add_border first unlocks at level 6; "
                "BabyTheoristGym levels 1, 2-3, and 4-5 use smaller hard libraries."
            )
    else:
        lines.append(f"- suites: {', '.join(args.suites)}")
        lines.append(
            "- levels: " + ", ".join(str(level) for level in args.levels)
        )
    lines.append(
        "- parent lines show tracked accepted-sample parent id/count "
        "histograms when provenance is present."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
