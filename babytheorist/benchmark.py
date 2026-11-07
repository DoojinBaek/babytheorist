"""Benchmark-suite presets and primitive curricula for BabyTheorist generation."""

from __future__ import annotations

from dataclasses import dataclass


BENCHMARK_MODES: tuple[str, ...] = (
    "multi_dev",
    "single_a",
    "multi_a",
    "single_b",
    "multi_b",
)

BENCHMARK_MODE_ALIASES: dict[str, str] = {
    "singlea": "single_a",
    "single_easy": "single_a",
    "multia": "multi_a",
    "multi_easy": "multi_a",
    "singleb": "single_b",
    "single_hard": "single_b",
    "multib": "multi_b",
    "multi_hard": "multi_b",
}

DEFAULT_SELECTORS: tuple[str, ...] = ("ALL", "by_index")
WITHHELD_PRIMITIVES: tuple[str, ...] = (
    "hollow",
    "fill_interior",
    "add_border",
)
SINGLE_BASE_PRIMITIVES: tuple[str, ...] = (
    "rot90",
    "flip_h",
    "flip_v",
    "next_color",
    "translate_up",
    "translate_down",
    "translate_left",
    "translate_right",
)
TRANSLATE_PRIMITIVES: tuple[str, ...] = (
    "translate_up",
    "translate_down",
    "translate_left",
    "translate_right",
)
MULTI_BASE_PRIMITIVES: tuple[str, ...] = (
    *SINGLE_BASE_PRIMITIVES,
    "swap_colors",
    "align_row",
    "align_col",
)
EASY_SELECTORS: tuple[str, ...] = ("by_index",)
HARD_SELECTORS: tuple[str, ...] = (
    "ALL",
    "by_index",
    "by_color",
    "is_hollow",
    "topmost",
    "bottommost",
    "leftmost",
    "rightmost",
)
BENCHMARK_DEFAULT_N_SHAPES: dict[str, int] = {
    "multi_dev": 2,
    "single_a": 1,
    "single_b": 1,
    "multi_a": 2,
    "multi_b": 3,
}

BASE_POOL_SIZE_PER_DEPTH = 20
APPEND_POOL_SIZE_PER_DEPTH = 10
INTRO_SAMPLING_RATE = 0.5


@dataclass(frozen=True)
class PrimitiveCurriculum:
    """Resolved primitive curriculum for an official benchmark suite."""

    suite: str
    primitive_curriculum: str
    base_primitives: tuple[str, ...]
    withheld_primitives: tuple[str, ...]
    introduced_primitive: str | None
    effective_primitives: tuple[str, ...]
    selectors: tuple[str, ...]
    n_shapes: int
    inherited_segments: tuple[str, ...]
    current_intro_segment: str | None
    dominant_library_policy: str
    base_pool_size_per_depth: int
    append_pool_size_per_depth: int
    intro_sampling_rate: float


def introduced_primitive_for_level(level: int) -> str | None:
    if level in (2, 3):
        return "hollow"
    if level in (4, 5):
        return "fill_interior"
    if level in (6, 7):
        return "add_border"
    return None


def normalize_benchmark_mode(mode: str) -> str:
    """Return the public canonical benchmark mode for aliases.

    Public benchmark names are the A/B labels. The previous descriptive names
    are accepted as legacy aliases so existing scripts and metadata can still
    be interpreted.
    """
    key = mode.strip().replace("-", "_").lower()
    if key in BENCHMARK_MODES:
        return key
    try:
        return BENCHMARK_MODE_ALIASES[key]
    except KeyError as e:
        aliases = ", ".join(sorted(BENCHMARK_MODE_ALIASES))
        raise ValueError(
            f"benchmark mode must be one of {BENCHMARK_MODES} "
            f"(aliases: {aliases}), got {mode!r}"
        ) from e


def unlocked_primitives_through_level(level: int) -> tuple[str, ...]:
    unlocked: list[str] = []
    if level >= 2:
        unlocked.append("hollow")
    if level >= 4:
        unlocked.append("fill_interior")
    if level >= 6:
        unlocked.append("add_border")
    return tuple(unlocked)


def resolve_primitive_curriculum(mode: str, level: int) -> PrimitiveCurriculum:
    """Resolve official suite + level into primitive curriculum settings."""
    suite = normalize_benchmark_mode(mode)
    is_dev = suite == "multi_dev"
    is_multi = suite in ("multi_dev", "multi_a", "multi_b")
    is_hard = suite in ("single_b", "multi_b")
    if is_dev:
        base = TRANSLATE_PRIMITIVES
    else:
        base = MULTI_BASE_PRIMITIVES if is_multi else SINGLE_BASE_PRIMITIVES
    selectors = HARD_SELECTORS if is_hard and is_multi else EASY_SELECTORS
    introduced = introduced_primitive_for_level(level) if is_hard else None
    unlocked = unlocked_primitives_through_level(level) if is_hard else ()
    effective = base + unlocked
    inherited_unlocks = tuple(p for p in unlocked if p != introduced)
    inherited = ("base",) + tuple(f"append:{p}" for p in inherited_unlocks)
    current_intro = f"append:{introduced}" if introduced is not None else None
    return PrimitiveCurriculum(
        suite=suite,
        primitive_curriculum=(
            "translate_only_dev_v1" if is_dev else "primitive_curriculum_v1"
        ),
        base_primitives=base,
        withheld_primitives=WITHHELD_PRIMITIVES,
        introduced_primitive=introduced,
        effective_primitives=effective,
        selectors=selectors,
        n_shapes=BENCHMARK_DEFAULT_N_SHAPES[suite],
        inherited_segments=inherited,
        current_intro_segment=current_intro,
        dominant_library_policy="append_only",
        base_pool_size_per_depth=BASE_POOL_SIZE_PER_DEPTH,
        append_pool_size_per_depth=APPEND_POOL_SIZE_PER_DEPTH,
        intro_sampling_rate=INTRO_SAMPLING_RATE if current_intro else 0.0,
    )


def benchmark_mode_preset(
    mode: str,
    level: int = 1,
) -> tuple[list[str], tuple[str, ...]]:
    """Return the official benchmark vocabulary for a benchmark mode."""
    curriculum = resolve_primitive_curriculum(mode, level)
    return list(curriculum.effective_primitives), curriculum.selectors


def benchmark_mode_default_n_shapes(mode: str | None) -> int | None:
    """Default fixed object count for official benchmark modes."""
    if mode is None:
        return None
    mode = normalize_benchmark_mode(mode)
    try:
        return BENCHMARK_DEFAULT_N_SHAPES[mode]
    except KeyError as e:
        raise ValueError(
            f"benchmark mode must be one of {BENCHMARK_MODES}, got {mode!r}"
        ) from e


# Private-name aliases kept for generate.py and existing tests.
_BENCHMARK_MODES = BENCHMARK_MODES
_BENCHMARK_MODE_ALIASES = BENCHMARK_MODE_ALIASES
_DEFAULT_SELECTORS = DEFAULT_SELECTORS
_WITHHELD_PRIMITIVES = WITHHELD_PRIMITIVES
_SINGLE_BASE_PRIMITIVES = SINGLE_BASE_PRIMITIVES
_TRANSLATE_PRIMITIVES = TRANSLATE_PRIMITIVES
_MULTI_BASE_PRIMITIVES = MULTI_BASE_PRIMITIVES
_EASY_SELECTORS = EASY_SELECTORS
_HARD_SELECTORS = HARD_SELECTORS
_BENCHMARK_DEFAULT_N_SHAPES = BENCHMARK_DEFAULT_N_SHAPES
_BASE_POOL_SIZE_PER_DEPTH = BASE_POOL_SIZE_PER_DEPTH
_APPEND_POOL_SIZE_PER_DEPTH = APPEND_POOL_SIZE_PER_DEPTH
_INTRO_SAMPLING_RATE = INTRO_SAMPLING_RATE
_introduced_primitive_for_level = introduced_primitive_for_level
_unlocked_primitives_through_level = unlocked_primitives_through_level
_resolve_primitive_curriculum = resolve_primitive_curriculum
_benchmark_mode_preset = benchmark_mode_preset
_benchmark_mode_default_n_shapes = benchmark_mode_default_n_shapes
_normalize_benchmark_mode = normalize_benchmark_mode
