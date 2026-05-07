"""Generation configuration and curriculum depth policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babytheorist.core import (
    SELECTOR_TYPES,
)
from babytheorist.primitives import PRIMITIVES
from babytheorist.benchmark import (
    PrimitiveCurriculum,
    _DEFAULT_SELECTORS,
)

_MODES = ("train", "eval")
_ATTEMPT_SCHEDULERS = ("ordered", "unordered")
_DEPTH_PRIOR_PIECEWISE = "piecewise_geometric"
_FILTER_SYSTEM = "filter"
_PAIR_FAILURE_SCENE_NONE = "scene_none"
_PAIR_FAILURE_RASTER_ORDER = "raster_order"
_PAIR_FAILURE_ENGINE_NONE = "engine_none"
_PAIR_FAILURE_IDENTITY = "identity_pair"
_FILTER_PAIR_EXECUTION_RETRIES = 16

_DEPTH_CENTER_SHIFT = 1
_DEPTH_ALPHA = 0.60
_DEPTH_BETA = 0.60
_DEPTH_RHO = 0.35
_DEPTH_MASS_LEFT = 0.80
_DEPTH_MASS_SHOULDER = 0.15
_DEPTH_MASS_TAIL = 0.05
_DEPTH_TAIL_TRUNCATION = 6

_DOMINANT_POOL_MAX_DEPTH = 10
_DOMINANT_DEPTH_ONE_POLICY = "atomic_valid_kind_neutral_slots_v4"
_DOMINANT_VIABILITY_POLICY = "grounding_pair_bbox_local_success_fixed_size_v7"
_DOMINANT_VIABILITY_CANDIDATE_FACTOR = 50
_DOMINANT_VIABILITY_TRIAL_SCHEDULE = ((10, 20),)
_DOMINANT_VIABILITY_MAX_PAIR_RETRIES = 8
_DOMINANT_VIABILITY_EXECUTION_RETRIES = 2


# === Configs ===============================================================


@dataclass
class LevelConfig:
    """One curriculum level.

    Attributes:
        level: Curriculum level id.
        primitives: Allowed primitive names.
        selectors: Allowed selector `type` strings (see SELECTOR_TYPES).
                   Default `("ALL", "by_index")` — global broadcast plus
                   raster-order per-shape targeting.
        fixed_depth: Optional Python-only escape hatch for tests/debugging.
                     When None, depth is sampled from the official
                     piecewise-geometric prior.
        palette: Color integers (1-9) allowed in generated shapes.
        use_dominant: If True, sample from the compositional dominant-program
                      pool; if False, sample each primitive from the raw bigram.
        primitive_weights: Per-primitive sampling weights (default 1.0).
                           Used by `build_pool` when `use_dominant=True` and
                           by `sample_bigram` when `use_dominant=False`.
    """

    level: int
    primitives: list[str]
    selectors: tuple[str, ...] = _DEFAULT_SELECTORS
    fixed_depth: int | None = None
    palette: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
    use_dominant: bool = True
    primitive_weights: dict[str, float] | None = None
    curriculum: PrimitiveCurriculum | None = None


@dataclass
class TaskConfig:
    """Settings shared across levels.

    `mode` drives pair count and whether `valid_query_outputs` is computed:
        "train" → 1 pair (just (x, y)), no Y_q.
        "eval"  → support pair(s) plus one query pair + Y_q.
    """

    # task / pair shape
    mode: str = "train"
    n_shapes_range: tuple[int, int] = (2, 4)
    world_size: int = 20
    gap: int = 2
    shape_size_range: tuple[int, int] = (4, 10)
    shape_compactness: float = 8.0
    shape_bordered_hollow_min_size: int | None = None
    shape_bordered_hollow_random_inner_size: bool = False
    shape_max_bbox_span: int | None = None

    # retry / validation
    validate_programs: bool = True
    max_program_retries: int = 50
    max_pair_retries: int = 100
    # Eval mode defaults to adaptive support: add support observations until
    # the query answer set is unique, or reject the task at max_support.
    eval_unique_support: bool = True
    max_support: int = 5
    # Force at least this many supports per eval task before unique-Y_q
    # acceptance. Default 2 gives every eval row at least two demonstrations
    # while preserving adaptive growth up to max_support.
    min_support: int = 2
    # Strict Y_q (demo and every alternative must work on query) drops
    # ~3x more tasks than the permissive rule at T3, so 10 is tight at
    # the eval end. 30 gives comfortable headroom without runaway loops.
    max_dataset_attempts_per_task: int = 30

    # Phase 2 brute-force scene shortcut check
    # (``_scene_has_shorter_shortcut``). Enumerates all shorter programs Q
    # and rejects a scene where some Q reproduces P's output grid.
    # Default **off** because this is an *analysis* tool (used to validate
    # that C(P) is sound on samples); data generation uses the symbolic
    # ``symbolic_constraints`` predicate in ``babytheorist.predicates`` instead.
    scene_shortcut_check: bool = False
    # Upper bound on enumerated Q count; above this, the brute-force scene
    # check is skipped. 50K comfortably covers ALL-selector vocabs at
    # depths 1-6; by_index at depth 5 can exceed it.
    scene_shortcut_q_cap: int = 50_000

    # Phase 2 symbolic scene predicate (C(P)). Reject a scene when some
    # shorter Q is symbolically guaranteed to produce the same output
    # grid. Rule-based (no Q enumeration) — fast.
    symbolic_constraints: bool = True

    # When ``symbolic_constraints=True`` AND C(P) *defers* (no branch
    # can analyze — e.g. multi-color shape + mixed vocab), optionally
    # fall back to brute-force Q enumeration on that pair. Intended for
    # verification and analysis only — production should keep this
    # False and instead extend C(P) coverage so no defer is needed.
    symbolic_constraints_fallback: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"TaskConfig.mode must be one of {_MODES}, got {self.mode!r}")
        lo_n, hi_n = self.n_shapes_range
        if lo_n < 1 or hi_n < lo_n:
            raise ValueError(
                "n_shapes_range must satisfy 1 <= min <= max, "
                f"got {self.n_shapes_range}"
            )
        if self.max_support < 1:
            raise ValueError(f"max_support must be >= 1, got {self.max_support}")
        if self.min_support < 1:
            raise ValueError(f"min_support must be >= 1, got {self.min_support}")
        if self.min_support > self.max_support:
            raise ValueError(
                f"min_support={self.min_support} > max_support={self.max_support}"
            )
        lo_s, hi_s = self.shape_size_range
        if lo_s < 1 or hi_s < lo_s:
            raise ValueError(
                "shape_size_range must satisfy 1 <= min <= max, "
                f"got {self.shape_size_range}"
            )
        if self.shape_compactness < 0:
            raise ValueError(
                "shape_compactness must be >= 0, "
                f"got {self.shape_compactness}"
            )
        if (
            self.shape_bordered_hollow_min_size is not None
            and self.shape_bordered_hollow_min_size < 1
        ):
            raise ValueError(
                "shape_bordered_hollow_min_size must be >= 1 or None, "
                f"got {self.shape_bordered_hollow_min_size}"
            )
        if self.shape_max_bbox_span is not None and self.shape_max_bbox_span < 1:
            raise ValueError(
                "shape_max_bbox_span must be >= 1 or None, "
                f"got {self.shape_max_bbox_span}"
            )

    @property
    def n_pairs(self) -> int:
        if self.mode != "eval":
            return 1
        return self.max_support + 1 if self.eval_unique_support else 2


# === Validation ===========================================================


def _validate_level_cfg(cfg: LevelConfig) -> None:
    if cfg.level < 1:
        raise ValueError(f"LevelConfig.level must be >= 1, got {cfg.level}")
    if not cfg.primitives:
        raise ValueError("LevelConfig.primitives must be non-empty")
    unknown = [p for p in cfg.primitives if p not in PRIMITIVES]
    if unknown:
        raise ValueError(f"unknown primitive names: {unknown}")
    if not cfg.selectors:
        raise ValueError("LevelConfig.selectors must be non-empty")
    bad_sel = [s for s in cfg.selectors if s not in SELECTOR_TYPES]
    if bad_sel:
        raise ValueError(f"unknown selector types: {bad_sel}")
    if cfg.fixed_depth is not None and cfg.fixed_depth < 1:
        raise ValueError(f"fixed_depth must be >= 1, got {cfg.fixed_depth}")
    if not cfg.palette or any(c < 1 or c > 9 for c in cfg.palette):
        raise ValueError(f"palette must be non-empty subset of 1..9, got {cfg.palette}")


def _depth_center(level: int) -> int:
    """Official depth-prior center for curriculum level `level`."""
    return level + _DEPTH_CENTER_SHIFT


def _depth_support_max(level_cfg: LevelConfig) -> int:
    """Maximum generated depth needed for sampling and dominant pools."""
    if level_cfg.fixed_depth is not None:
        return level_cfg.fixed_depth
    return _depth_center(level_cfg.level) + 2 + _DEPTH_TAIL_TRUNCATION


def _dominant_pool_max_depth(level_cfg: LevelConfig) -> int:
    """Maximum depth built directly in the abstract dominant pool.

    Task depth sampling remains governed by ``_depth_support_max``.  When a
    sampled task is deeper than this cap, ``sample_abstract_program`` composes
    smaller abstract chunks via its walking path instead of requiring a direct
    pool entry at that depth.
    """
    return min(_depth_support_max(level_cfg), _DOMINANT_POOL_MAX_DEPTH)


def _depth_distribution_params(level_cfg: LevelConfig) -> dict[str, object]:
    """Metadata for the official depth prior."""
    return {
        "center_shift": _DEPTH_CENTER_SHIFT,
        "alpha": _DEPTH_ALPHA,
        "beta": _DEPTH_BETA,
        "rho": _DEPTH_RHO,
        "mass_left": _DEPTH_MASS_LEFT,
        "mass_shoulder": _DEPTH_MASS_SHOULDER,
        "mass_tail": _DEPTH_MASS_TAIL,
        "tail_truncation": _DEPTH_TAIL_TRUNCATION,
    }


def _sample_piecewise_geometric_depth(
    level: int, rng: np.random.Generator,
) -> int:
    """Sample from the official piecewise-geometric depth prior."""
    center = _depth_center(level)
    max_depth = center + 2 + _DEPTH_TAIL_TRUNCATION
    depths = np.arange(1, max_depth + 1)
    probs = np.zeros_like(depths, dtype=float)

    left = depths <= center
    left_weights = _DEPTH_ALPHA ** (center - depths[left])
    probs[left] = _DEPTH_MASS_LEFT * left_weights / left_weights.sum()

    probs[depths == center + 1] = _DEPTH_MASS_SHOULDER / (1.0 + _DEPTH_BETA)
    probs[depths == center + 2] = (
        _DEPTH_MASS_SHOULDER * _DEPTH_BETA / (1.0 + _DEPTH_BETA)
    )

    tail = depths > center + 2
    tail_weights = _DEPTH_RHO ** (depths[tail] - (center + 3))
    probs[tail] = _DEPTH_MASS_TAIL * tail_weights / tail_weights.sum()

    return int(rng.choice(depths, p=probs))


def _sample_depth(level_cfg: LevelConfig, rng: np.random.Generator) -> int:
    """Sample one task depth from the configured level prior."""
    if level_cfg.fixed_depth is not None:
        return level_cfg.fixed_depth
    return _sample_piecewise_geometric_depth(level_cfg.level, rng)


__all__ = [
    "_MODES",
    "_ATTEMPT_SCHEDULERS",
    "_DEPTH_PRIOR_PIECEWISE",
    "_FILTER_SYSTEM",
    "_PAIR_FAILURE_SCENE_NONE",
    "_PAIR_FAILURE_RASTER_ORDER",
    "_PAIR_FAILURE_ENGINE_NONE",
    "_PAIR_FAILURE_IDENTITY",
    "_FILTER_PAIR_EXECUTION_RETRIES",
    "_DEPTH_CENTER_SHIFT",
    "_DEPTH_ALPHA",
    "_DEPTH_BETA",
    "_DEPTH_RHO",
    "_DEPTH_MASS_LEFT",
    "_DEPTH_MASS_SHOULDER",
    "_DEPTH_MASS_TAIL",
    "_DEPTH_TAIL_TRUNCATION",
    "_DOMINANT_POOL_MAX_DEPTH",
    "_DOMINANT_DEPTH_ONE_POLICY",
    "_DOMINANT_VIABILITY_POLICY",
    "_DOMINANT_VIABILITY_CANDIDATE_FACTOR",
    "_DOMINANT_VIABILITY_TRIAL_SCHEDULE",
    "_DOMINANT_VIABILITY_MAX_PAIR_RETRIES",
    "_DOMINANT_VIABILITY_EXECUTION_RETRIES",
    "LevelConfig",
    "TaskConfig",
    "_validate_level_cfg",
    "_depth_center",
    "_depth_support_max",
    "_dominant_pool_max_depth",
    "_depth_distribution_params",
    "_sample_piecewise_geometric_depth",
    "_sample_depth",
]
