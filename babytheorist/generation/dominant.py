"""Dominant abstract-program pools for filter generation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from babytheorist.filter.constraint import InfeasibleConstraint
from babytheorist.dominant import (
    abstract_program_to_strings,
    abstract_pool_from_jsonable,
    abstract_pool_to_jsonable,
    abstract_provenance_entry_ids,
    abstract_provenance_from_jsonable,
    abstract_provenance_to_jsonable,
    build_abstract_pool,
    build_append_abstract_pool,
    DOMINANT_PROVENANCE_POLICY,
    RECURSIVE_COMPOSITION_POLICY,
    RECURSIVE_COMPOSITION_SAMPLE_COUNT,
    sample_abstract_program,
    AbstractPool,
    AbstractProgram,
    AbstractProvenance,
    AbstractRef,
)
from babytheorist.program import (
    SamplerDeadEnd, build_bigram,
)
from babytheorist.shapes import ShapeFactory
from babytheorist.version import (
    BENCHMARK_SPEC_VERSION,
    __version__ as BABYTHEORIST_VERSION,
)
from babytheorist.benchmark import (
    PrimitiveCurriculum,
    _WITHHELD_PRIMITIVES,
)

from babytheorist.generation.config import (
    LevelConfig,
    TaskConfig,
    _DOMINANT_DEPTH_ONE_POLICY,
    _DOMINANT_VIABILITY_CANDIDATE_FACTOR,
    _DOMINANT_VIABILITY_EXECUTION_RETRIES,
    _DOMINANT_VIABILITY_MAX_PAIR_RETRIES,
    _DOMINANT_VIABILITY_POLICY,
    _DOMINANT_VIABILITY_TRIAL_SCHEDULE,
    _FILTER_PAIR_EXECUTION_RETRIES,
    _FILTER_SYSTEM,
    _dominant_pool_max_depth,
)
from babytheorist.generation.filter_pairs import (
    _decode_grounding_signature,
    _filter_grounding_version,
    _filter_rules_version,
    _filter_sampler_module,
    _sample_filter_pair_with_reason,
    _with_raster_order_constraint,
)
from babytheorist.generation.rejection import _shape_factory_from_task_config

_ABSTRACT_POOL_CACHE: dict[tuple, object] = {}


@dataclass
class DominantLibrary:
    """Segmented dominant library for append-only benchmark curricula."""

    bigram: np.ndarray
    segments: dict[str, AbstractPool]
    metadata: dict
    provenance: dict[str, AbstractProvenance] = field(default_factory=dict)


@dataclass
class FilterGenerationStats:
    """Aggregated diagnostics for one filter-backend generation run."""

    filter_system: str = _FILTER_SYSTEM
    requested_tasks: int = 0
    produced_tasks: int = 0
    workers: int = 1
    dataset_attempts: int = 0
    none_attempts: int = 0
    attempt_timeouts: int = 0
    duplicate_rejects: int = 0
    program_attempts: int = 0
    program_dead_ends: int = 0
    sampler_calls: int = 0
    sampler_dead_ends: int = 0
    sampler_steps: int = 0
    sampler_candidate_count_total: int = 0
    sampler_valid_candidate_count_total: int = 0
    sampler_reject_counts: Counter[str] = field(default_factory=Counter)
    sampler_elapsed_sec: float = 0.0
    dominant_grounding_attempts: int = 0
    dominant_grounding_dead_ends: int = 0
    pair_attempts: int = 0
    pair_failures: int = 0
    pair_failure_reasons: Counter[str] = field(default_factory=Counter)
    pair_elapsed_sec: float = 0.0
    grounding_signature_attempts: Counter[str] = field(default_factory=Counter)
    grounding_signature_successes: Counter[str] = field(default_factory=Counter)
    grounding_signature_pair_failures: dict[str, Counter[str]] = field(
        default_factory=dict
    )
    elapsed_sec: float = 0.0

    def record_grounding_signature(self, signature: str | None) -> None:
        if signature is not None:
            self.grounding_signature_attempts[signature] += 1

    def record_grounding_success(self, signature: str | None) -> None:
        if signature is not None:
            self.grounding_signature_successes[signature] += 1

    def record_pair_failure(
        self,
        reason: str | None,
        signature: str | None = None,
    ) -> None:
        reason = reason or "unknown"
        self.pair_failures += 1
        self.pair_failure_reasons[str(reason)] += 1
        if signature is not None:
            bucket = self.grounding_signature_pair_failures.setdefault(
                signature, Counter()
            )
            bucket[str(reason)] += 1

    def merge_sampler(self, sampler_stats) -> None:
        data = sampler_stats.as_dict()
        candidate_counts = data.get("candidate_counts", [])
        valid_counts = data.get("valid_candidate_counts", [])
        self.sampler_steps += len(candidate_counts)
        self.sampler_candidate_count_total += sum(int(v) for v in candidate_counts)
        self.sampler_valid_candidate_count_total += sum(
            int(v) for v in valid_counts
        )
        self.sampler_dead_ends += int(data.get("dead_end_count", 0))
        self.sampler_reject_counts.update(data.get("reject_counts", {}))

    def generation_dict(self) -> dict[str, object]:
        task_rate = (
            self.produced_tasks / self.elapsed_sec
            if self.elapsed_sec > 0.0 else None
        )
        avg_candidates = (
            self.sampler_candidate_count_total / self.sampler_steps
            if self.sampler_steps else None
        )
        avg_valid_candidates = (
            self.sampler_valid_candidate_count_total / self.sampler_steps
            if self.sampler_steps else None
        )
        return {
            "filter_system": self.filter_system,
            "requested_tasks": self.requested_tasks,
            "produced_tasks": self.produced_tasks,
            "workers": self.workers,
            "dataset_attempts": self.dataset_attempts,
            "none_attempts": self.none_attempts,
            "attempt_timeouts": self.attempt_timeouts,
            "duplicate_rejects": self.duplicate_rejects,
            "program_attempts": self.program_attempts,
            "program_dead_ends": self.program_dead_ends,
            "sampler_calls": self.sampler_calls,
            "sampler_dead_ends": self.sampler_dead_ends,
            "sampler_steps": self.sampler_steps,
            "avg_candidates_per_step": avg_candidates,
            "avg_valid_candidates_per_step": avg_valid_candidates,
            "sampler_elapsed_sec": self.sampler_elapsed_sec,
            "avg_sampler_sec_per_call": (
                self.sampler_elapsed_sec / self.sampler_calls
                if self.sampler_calls else None
            ),
            "dominant_grounding_attempts": self.dominant_grounding_attempts,
            "dominant_grounding_dead_ends": self.dominant_grounding_dead_ends,
            "pair_attempts": self.pair_attempts,
            "pair_execution_retries": _FILTER_PAIR_EXECUTION_RETRIES,
            "pair_failures": self.pair_failures,
            "pair_failure_reasons": dict(sorted(self.pair_failure_reasons.items())),
            "pair_elapsed_sec": self.pair_elapsed_sec,
            "avg_pair_sec_per_attempt": (
                self.pair_elapsed_sec / self.pair_attempts
                if self.pair_attempts else None
            ),
            "grounding_signature_count": len(self.grounding_signature_attempts),
            "grounding_signature_stats": self.grounding_signature_dict(),
            "elapsed_sec": self.elapsed_sec,
            "tasks_per_sec": task_rate,
        }

    def reject_dict(self) -> dict[str, int]:
        return dict(sorted(self.sampler_reject_counts.items()))

    def grounding_signature_dict(self, limit: int = 20) -> list[dict[str, object]]:
        rows = []
        signatures = set(self.grounding_signature_attempts)
        signatures.update(self.grounding_signature_pair_failures)
        signatures.update(self.grounding_signature_successes)
        for signature in signatures:
            failures = self.grounding_signature_pair_failures.get(
                signature, Counter()
            )
            rows.append({
                "signature": _decode_grounding_signature(signature),
                "program_attempts": self.grounding_signature_attempts[signature],
                "task_successes": self.grounding_signature_successes[signature],
                "pair_failures": int(sum(failures.values())),
                "pair_failure_reasons": dict(sorted(failures.items())),
            })
        rows.sort(
            key=lambda row: (
                -int(row["pair_failures"]),
                -int(row["program_attempts"]),
                str(row["signature"]),
            )
        )
        return rows[:limit]


def _pool_initial_grid_shapes(
    initial_grid_shapes: list[str] | None,
) -> list[str] | None:
    """Coarsen initial-shape state for abstract-pool caching.

    The concrete sampler still validates every grounded program against
    the exact per-task ``initial_grid_shapes``. Pool construction only
    needs a separate all-hollow pool to admit first-step
    ``fill_interior`` candidates. Mixed hollow/solid patterns would
    otherwise create many expensive cache entries during long runs.
    """
    if initial_grid_shapes and all(gs == "hollow" for gs in initial_grid_shapes):
        return list(initial_grid_shapes)
    return None


def _abstract_pool_key(
    level_cfg,
    n_shapes: int,
    bigram_weights,
    initial_grid_shapes,
    pool_seed,
) -> tuple:
    primitive_weights = tuple(sorted((level_cfg.primitive_weights or {}).items()))
    bigram_items = tuple(sorted((bigram_weights or {}).items()))
    return (
        tuple(level_cfg.primitives),
        tuple(level_cfg.selectors),
        tuple(level_cfg.palette),
        primitive_weights,
        bigram_items,
        n_shapes,
        tuple(initial_grid_shapes) if initial_grid_shapes is not None else None,
        pool_seed,
        _dominant_pool_max_depth(level_cfg),
        RECURSIVE_COMPOSITION_POLICY,
        RECURSIVE_COMPOSITION_SAMPLE_COUNT,
        DOMINANT_PROVENANCE_POLICY,
    )


def _dominant_pool_cache_metadata(
    level_cfg,
    n_shapes: int,
    bigram_weights,
    initial_grid_shapes,
    pool_seed,
) -> dict:
    return {
        "format": "babytheorist.abstract_dominant_pool.v1",
        "babytheorist_version": BABYTHEORIST_VERSION,
        "benchmark_spec_version": BENCHMARK_SPEC_VERSION,
        "primitives": list(level_cfg.primitives),
        "selectors": list(level_cfg.selectors),
        "palette": list(level_cfg.palette),
        "primitive_weights": [
            [name, weight]
            for name, weight in sorted((level_cfg.primitive_weights or {}).items())
        ],
        "bigram_weights": [
            [a, b, v] for (a, b), v in sorted((bigram_weights or {}).items())
        ],
        "n_shapes": int(n_shapes),
        "initial_grid_shapes": (
            list(initial_grid_shapes)
            if initial_grid_shapes is not None else None
        ),
        "pool_seed": pool_seed,
        "max_depth": _dominant_pool_max_depth(level_cfg),
        "dominant_depth_one_policy": _DOMINANT_DEPTH_ONE_POLICY,
        "dominant_composition_policy": RECURSIVE_COMPOSITION_POLICY,
        "dominant_composition_sample_count": RECURSIVE_COMPOSITION_SAMPLE_COUNT,
        "dominant_provenance_policy": DOMINANT_PROVENANCE_POLICY,
    }


def _dominant_pool_cache_path(cache_dir: str | Path, metadata: dict) -> Path:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"abstract_pool_{digest}.json"


def _load_cached_abstract_pool(
    cache_dir: str | Path | None, metadata: dict,
):
    if cache_dir is None:
        return None
    path = _dominant_pool_cache_path(cache_dir, metadata)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if payload.get("metadata") != metadata:
        return None
    return abstract_pool_from_jsonable(payload["pool"])


def _save_cached_abstract_pool(
    cache_dir: str | Path | None, metadata: dict, pool,
) -> None:
    if cache_dir is None:
        return
    path = _dominant_pool_cache_path(cache_dir, metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "pool": abstract_pool_to_jsonable(pool),
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _dominant_library_key(
    level_cfg: LevelConfig,
    n_shapes: int,
    bigram_weights,
    pool_seed,
    task_cfg: TaskConfig | None = None,
) -> tuple:
    curriculum = level_cfg.curriculum
    if curriculum is None:
        raise ValueError("segmented dominant library requires a curriculum")
    primitive_weights = tuple(sorted((level_cfg.primitive_weights or {}).items()))
    bigram_items = tuple(sorted((bigram_weights or {}).items()))
    return (
        curriculum.suite,
        level_cfg.level,
        tuple(curriculum.selectors),
        tuple(level_cfg.palette),
        primitive_weights,
        bigram_items,
        n_shapes,
        tuple(curriculum.base_primitives),
        curriculum.introduced_primitive,
        tuple(curriculum.effective_primitives),
        pool_seed,
        _dominant_pool_max_depth(level_cfg),
        _filter_rules_version(),
        _filter_grounding_version(),
        RECURSIVE_COMPOSITION_POLICY,
        RECURSIVE_COMPOSITION_SAMPLE_COUNT,
        _dominant_viability_signature(task_cfg),
    )


def _curriculum_metadata(curriculum: PrimitiveCurriculum) -> dict:
    return {
        "primitive_curriculum": curriculum.primitive_curriculum,
        "base_primitives": list(curriculum.base_primitives),
        "withheld_primitives": list(curriculum.withheld_primitives),
        "introduced_primitive": curriculum.introduced_primitive,
        "effective_primitives": list(curriculum.effective_primitives),
        "dominant_library_policy": curriculum.dominant_library_policy,
        "inherited_segments": list(curriculum.inherited_segments),
        "current_intro_segment": curriculum.current_intro_segment,
        "base_pool_size_per_depth": curriculum.base_pool_size_per_depth,
        "append_pool_size_per_depth": curriculum.append_pool_size_per_depth,
        "intro_sampling_rate": curriculum.intro_sampling_rate,
        "dominant_depth_one_policy": _DOMINANT_DEPTH_ONE_POLICY,
        "dominant_composition_policy": RECURSIVE_COMPOSITION_POLICY,
        "dominant_composition_sample_count": RECURSIVE_COMPOSITION_SAMPLE_COUNT,
    }


def _dominant_library_cache_metadata(
    level_cfg: LevelConfig,
    n_shapes: int,
    bigram_weights,
    pool_seed,
    task_cfg: TaskConfig | None = None,
) -> dict:
    curriculum = level_cfg.curriculum
    if curriculum is None:
        raise ValueError("segmented dominant library requires a curriculum")
    return {
        "format": "babytheorist.abstract_dominant_library.v6",
        "babytheorist_version": BABYTHEORIST_VERSION,
        "benchmark_spec_version": BENCHMARK_SPEC_VERSION,
        "filter_rules_version": _filter_rules_version(),
        "filter_grounding_version": _filter_grounding_version(),
        "suite": curriculum.suite,
        "level": int(level_cfg.level),
        "selectors": list(curriculum.selectors),
        "palette": list(level_cfg.palette),
        "primitive_weights": [
            [name, weight]
            for name, weight in sorted((level_cfg.primitive_weights or {}).items())
        ],
        "bigram_weights": [
            [a, b, v] for (a, b), v in sorted((bigram_weights or {}).items())
        ],
        "n_shapes": int(n_shapes),
        "pool_seed": pool_seed,
        "max_depth": _dominant_pool_max_depth(level_cfg),
        "dominant_viability": _dominant_viability_metadata(task_cfg),
        **_curriculum_metadata(curriculum),
    }


def _dominant_library_cache_path(cache_dir: str | Path, metadata: dict) -> Path:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"abstract_library_{digest}.json"


def _load_cached_dominant_library(
    cache_dir: str | Path | None, metadata: dict, bigram: np.ndarray,
) -> DominantLibrary | None:
    if cache_dir is None:
        return None
    path = _dominant_library_cache_path(cache_dir, metadata)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if payload.get("metadata") != metadata:
        return None
    segments = {
        name: abstract_pool_from_jsonable(raw)
        for name, raw in payload["segments"].items()
    }
    provenance = {
        name: abstract_provenance_from_jsonable(raw)
        for name, raw in payload.get("provenance", {}).items()
    }
    return DominantLibrary(
        bigram=bigram,
        segments=segments,
        metadata=metadata,
        provenance=provenance,
    )


def _save_cached_dominant_library(
    cache_dir: str | Path | None, library: DominantLibrary,
) -> None:
    if cache_dir is None:
        return
    path = _dominant_library_cache_path(cache_dir, library.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": library.metadata,
        "segments": {
            name: abstract_pool_to_jsonable(pool)
            for name, pool in sorted(library.segments.items())
        },
        "provenance": {
            name: abstract_provenance_to_jsonable(provenance)
            for name, provenance in sorted(library.provenance.items())
        },
    }
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _append_initial_grid_shapes(
    introduced: str, n_shapes: int,
) -> list[str] | None:
    if introduced == "fill_interior":
        return ["hollow"] * n_shapes
    return None


def _merge_segment_dict(
    segments: dict[str, AbstractPool], names: tuple[str, ...],
) -> AbstractPool:
    merged: AbstractPool = {}
    for name in names:
        for depth, entries in segments.get(name, {}).items():
            bucket = merged.setdefault(depth, {})
            for program, score in entries.items():
                bucket[program] = max(bucket.get(program, 0.0), score)
    return merged


def _merge_segment_entry_ids(
    segments: dict[str, AbstractPool],
    provenances: dict[str, AbstractProvenance],
    names: tuple[str, ...],
) -> dict[int, dict[AbstractProgram, str]]:
    merged: dict[int, dict[AbstractProgram, str]] = {}
    scores: dict[int, dict[AbstractProgram, float]] = {}
    segment_ids = {
        name: abstract_provenance_entry_ids(provenances.get(name, {}))
        for name in names
    }
    for name in names:
        ids = segment_ids.get(name, {})
        for depth, entries in segments.get(name, {}).items():
            id_bucket = merged.setdefault(depth, {})
            score_bucket = scores.setdefault(depth, {})
            for program, score in entries.items():
                previous = score_bucket.get(program)
                if previous is not None and score <= previous:
                    continue
                program_id = ids.get(depth, {}).get(program)
                if program_id is None:
                    continue
                score_bucket[program] = score
                id_bucket[program] = program_id
    return merged


def _synthetic_pool_provenance(
    pool: AbstractPool, id_prefix: str,
) -> AbstractProvenance:
    provenance: AbstractProvenance = {}
    for depth, entries in pool.items():
        bucket: dict[AbstractProgram, dict] = {}
        for index, (program, score) in enumerate(entries.items(), start=1):
            bucket[program] = {
                "id": f"{id_prefix}d{depth}.{index:02d}",
                "depth": depth,
                "index": index,
                "score": float(score),
                "parents": [],
            }
        provenance[depth] = bucket
    return provenance


def _pool_build_parts(result, id_prefix: str) -> tuple[AbstractPool, AbstractProvenance]:
    if hasattr(result, "pool") and hasattr(result, "provenance"):
        return result.pool, result.provenance
    pool = result
    return pool, _synthetic_pool_provenance(pool, id_prefix)


def _dominant_viability_task_signature(task_cfg: TaskConfig) -> tuple:
    return (
        int(task_cfg.world_size),
        int(task_cfg.gap),
        tuple(int(v) for v in task_cfg.shape_size_range),
        float(task_cfg.shape_compactness),
        task_cfg.shape_bordered_hollow_min_size,
        bool(task_cfg.shape_bordered_hollow_random_inner_size),
        task_cfg.shape_max_bbox_span,
    )


def _dominant_viability_signature(task_cfg: TaskConfig | None):
    signature = (
        _DOMINANT_VIABILITY_POLICY,
        "pair" if task_cfg is not None else "grounding",
        int(_DOMINANT_VIABILITY_CANDIDATE_FACTOR),
        tuple(
            (int(max_depth), int(trials))
            for max_depth, trials in _DOMINANT_VIABILITY_TRIAL_SCHEDULE
        ),
        int(_DOMINANT_VIABILITY_MAX_PAIR_RETRIES),
        int(_DOMINANT_VIABILITY_EXECUTION_RETRIES),
    )
    if task_cfg is None:
        return signature
    return signature + (_dominant_viability_task_signature(task_cfg),)


def _dominant_viability_metadata(task_cfg: TaskConfig | None) -> dict:
    metadata = {
        "policy": _DOMINANT_VIABILITY_POLICY,
        "preflight": (
            "concrete_grounding_and_pair"
            if task_cfg is not None else "concrete_grounding_only"
        ),
        "candidate_factor": _DOMINANT_VIABILITY_CANDIDATE_FACTOR,
        "trial_schedule": [
            {"max_depth": max_depth, "trials": trials}
            for max_depth, trials in _DOMINANT_VIABILITY_TRIAL_SCHEDULE
        ],
        "max_pair_retries": _DOMINANT_VIABILITY_MAX_PAIR_RETRIES,
        "execution_retries": _DOMINANT_VIABILITY_EXECUTION_RETRIES,
    }
    if task_cfg is not None:
        metadata["task_signature"] = {
            "world_size": int(task_cfg.world_size),
            "gap": int(task_cfg.gap),
            "shape_size_range": list(task_cfg.shape_size_range),
            "shape_compactness": float(task_cfg.shape_compactness),
            "shape_bordered_hollow_min_size": (
                task_cfg.shape_bordered_hollow_min_size
            ),
            "shape_bordered_hollow_random_inner_size": bool(
                task_cfg.shape_bordered_hollow_random_inner_size
            ),
            "shape_max_bbox_span": task_cfg.shape_max_bbox_span,
        }
    return metadata


def _dominant_viability_trials(depth: int) -> int:
    for max_depth, trials in _DOMINANT_VIABILITY_TRIAL_SCHEDULE:
        if depth <= max_depth:
            return trials
    return _DOMINANT_VIABILITY_TRIAL_SCHEDULE[-1][1]


def _dominant_preflight_seed(
    pool_seed: int | None,
    segment_name: str,
    depth: int,
    abstract: AbstractProgram,
) -> int:
    payload = json.dumps(
        {
            "pool_seed": pool_seed,
            "segment": segment_name,
            "depth": depth,
            "program": abstract_program_to_strings(abstract),
            "policy": _DOMINANT_VIABILITY_POLICY,
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def _abstract_has_generation_preflight_success(
    abstract: AbstractProgram,
    *,
    depth: int,
    level_cfg: LevelConfig,
    task_cfg: TaskConfig | None,
    factory: ShapeFactory | None,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    initial_grid_shapes: list[str] | None,
    pool_seed: int | None,
    segment_name: str,
) -> bool:
    sample_program_with_constraint = (
        _filter_sampler_module().sample_program_with_constraint
    )
    try:
        intro_grid_shapes, ref_grid_shapes = _initial_grid_shapes_for_abstract(
            abstract, n_shapes, initial_grid_shapes,
        )
    except SamplerDeadEnd:
        return False
    seed = _dominant_preflight_seed(pool_seed, segment_name, depth, abstract)
    for offset in range(_dominant_viability_trials(depth)):
        rng = np.random.default_rng(seed + offset)
        try:
            _program, _constraint = sample_program_with_constraint(
                depth=depth,
                n_shapes=n_shapes,
                vocab=vocab,
                selector_types=selector_types,
                rng=rng,
                initial_grid_shapes=intro_grid_shapes,
                palette=level_cfg.palette,
                abstract_program=abstract,
                ref_initial_grid_shapes=ref_grid_shapes,
            )
        except (SamplerDeadEnd, InfeasibleConstraint):
            continue
        if task_cfg is None or factory is None:
            return True
        uses_by_index = "by_index" in selector_types
        if uses_by_index:
            _constraint = _with_raster_order_constraint(_constraint)
        pair, _failure_reason = _sample_filter_pair_with_reason(
            _constraint,
            _program,
            factory,
            level_cfg,
            task_cfg,
            rng,
            uses_by_index=uses_by_index,
            gap=task_cfg.gap,
            execution_retries=_DOMINANT_VIABILITY_EXECUTION_RETRIES,
        )
        if pair is None:
            continue
        return True
    return False


def _dominant_entry_filter_factory(
    *,
    level_cfg: LevelConfig,
    task_cfg: TaskConfig | None,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    initial_grid_shapes: list[str] | None,
    pool_seed: int | None,
    target_size: int,
    segment_name: str,
):
    preflight_task_cfg = (
        replace(task_cfg, max_pair_retries=_DOMINANT_VIABILITY_MAX_PAIR_RETRIES)
        if task_cfg is not None else None
    )
    factory = (
        _shape_factory_from_task_config(preflight_task_cfg)
        if preflight_task_cfg is not None else None
    )

    def entry_filter(
        depth: int, entries: dict[AbstractProgram, float],
    ) -> dict[AbstractProgram, float]:
        if depth == 1:
            return {abstract: 1.0 for abstract in entries}
        kept: dict[AbstractProgram, float] = {}
        for abstract, score in entries.items():
            if _abstract_has_generation_preflight_success(
                abstract,
                depth=depth,
                level_cfg=level_cfg,
                task_cfg=preflight_task_cfg,
                factory=factory,
                n_shapes=n_shapes,
                selector_types=selector_types,
                vocab=vocab,
                initial_grid_shapes=initial_grid_shapes,
                pool_seed=pool_seed,
                segment_name=segment_name,
            ):
                kept[abstract] = score
                if len(kept) >= target_size:
                    return kept
        if len(kept) < target_size:
            raise SamplerDeadEnd(
                f"dominant {segment_name} depth {depth} produced only "
                f"{len(kept)}/{target_size} preflight-success candidates"
            )
        return kept

    return entry_filter


def _ensure_dominant_library(
    level_cfg: LevelConfig,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    *,
    bigram_weights=None,
    pool_seed=None,
    dominant_pool_dir=None,
    task_cfg: TaskConfig | None = None,
) -> DominantLibrary:
    curriculum = level_cfg.curriculum
    if curriculum is None:
        raise ValueError("segmented dominant library requires a curriculum")
    expected_vocab = frozenset(level_cfg.primitives)
    if vocab != expected_vocab:
        raise ValueError(
            "segmented dominant library vocab mismatch: "
            f"got {sorted(vocab)}, expected {sorted(expected_vocab)}"
        )
    curriculum_vocab = frozenset(curriculum.effective_primitives)
    if expected_vocab != curriculum_vocab:
        raise ValueError(
            "level primitives do not match primitive curriculum: "
            f"got {sorted(expected_vocab)}, expected {sorted(curriculum_vocab)}"
        )
    key = _dominant_library_key(
        level_cfg, n_shapes, bigram_weights, pool_seed, task_cfg,
    )
    if key in _ABSTRACT_POOL_CACHE:
        return _ABSTRACT_POOL_CACHE[key]

    bigram = build_bigram(level_cfg.primitives, bigram_weights=bigram_weights)
    metadata = _dominant_library_cache_metadata(
        level_cfg, n_shapes, bigram_weights, pool_seed, task_cfg,
    )
    cached = _load_cached_dominant_library(dominant_pool_dir, metadata, bigram)
    if cached is not None:
        _ABSTRACT_POOL_CACHE[key] = cached
        return cached

    pool_rng = (
        np.random.default_rng(pool_seed)
        if pool_seed is not None else None
    )
    max_depth = _dominant_pool_max_depth(level_cfg)
    base_primitives = list(curriculum.base_primitives)
    base_bigram = build_bigram(base_primitives, bigram_weights=bigram_weights)
    base_target_size = curriculum.base_pool_size_per_depth
    base_pool_size = base_target_size * _DOMINANT_VIABILITY_CANDIDATE_FACTOR
    base_filter = _dominant_entry_filter_factory(
        level_cfg=level_cfg,
        task_cfg=task_cfg,
        n_shapes=n_shapes,
        selector_types=selector_types,
        vocab=frozenset(base_primitives),
        initial_grid_shapes=None,
        pool_seed=pool_seed,
        target_size=base_target_size,
        segment_name="base",
    )
    base_build = build_abstract_pool(
        base_bigram,
        base_primitives,
        max_depth=max_depth,
        n_shapes=n_shapes,
        selector_types=selector_types,
        vocab=frozenset(base_primitives),
        pool_size=base_pool_size,
        primitive_weights=level_cfg.primitive_weights,
        rng=pool_rng,
        initial_grid_shapes=None,
        palette=level_cfg.palette,
        track_provenance=True,
        id_prefix="base:",
        entry_filter=base_filter,
    )
    base_pool, base_provenance = _pool_build_parts(base_build, "base:")
    segments: dict[str, AbstractPool] = {"base": base_pool}
    provenance: dict[str, AbstractProvenance] = {"base": base_provenance}
    unlocked_so_far: list[str] = []
    for introduced in _WITHHELD_PRIMITIVES:
        if introduced not in curriculum.effective_primitives:
            continue
        inherited_names = ("base",) + tuple(
            f"append:{primitive}" for primitive in unlocked_so_far
        )
        inherited_pool = _merge_segment_dict(segments, inherited_names)
        inherited_entry_ids = _merge_segment_entry_ids(
            segments, provenance, inherited_names,
        )
        append_vocab = frozenset(
            curriculum.base_primitives + tuple(unlocked_so_far) + (introduced,)
        )
        segment_name = f"append:{introduced}"
        append_initial_grid_shapes = _append_initial_grid_shapes(
            introduced, n_shapes,
        )
        append_target_size = curriculum.append_pool_size_per_depth
        append_pool_size = append_target_size * _DOMINANT_VIABILITY_CANDIDATE_FACTOR
        append_filter = _dominant_entry_filter_factory(
            level_cfg=level_cfg,
            task_cfg=task_cfg,
            n_shapes=n_shapes,
            selector_types=selector_types,
            vocab=append_vocab,
            initial_grid_shapes=append_initial_grid_shapes,
            pool_seed=pool_seed,
            target_size=append_target_size,
            segment_name=segment_name,
        )
        append_build = build_append_abstract_pool(
            inherited_pool,
            introduced,
            max_depth=max_depth,
            n_shapes=n_shapes,
            selector_types=selector_types,
            vocab=append_vocab,
            pool_size=append_pool_size,
            rng=pool_rng,
            initial_grid_shapes=append_initial_grid_shapes,
            palette=level_cfg.palette,
            bigram=bigram,
            primitives=level_cfg.primitives,
            inherited_entry_ids=inherited_entry_ids,
            track_provenance=True,
            id_prefix=f"{segment_name}:",
            entry_filter=append_filter,
        )
        append_pool, append_provenance = _pool_build_parts(
            append_build, f"{segment_name}:",
        )
        segments[segment_name] = append_pool
        provenance[segment_name] = append_provenance
        unlocked_so_far.append(introduced)
    library = DominantLibrary(
        bigram=bigram,
        segments=segments,
        metadata=metadata,
        provenance=provenance,
    )
    _save_cached_dominant_library(dominant_pool_dir, library)
    _ABSTRACT_POOL_CACHE[key] = library
    return library


def _merge_pool_segments(
    library: DominantLibrary, names: tuple[str, ...],
) -> AbstractPool:
    return _merge_segment_dict(library.segments, names)


def _pool_nonempty_depths(pool: AbstractPool) -> list[int]:
    return [d for d, entries in pool.items() if entries]


def _sample_intro_abstract_program(
    library: DominantLibrary,
    inherited_pool: AbstractPool,
    intro_pool: AbstractPool,
    primitives: list[str],
    depth: int,
    rng: np.random.Generator,
) -> AbstractProgram:
    """Sample one current-intro append entry and fill the rest inherited."""
    intro_depths = [d for d in _pool_nonempty_depths(intro_pool) if d <= depth]
    if not intro_depths:
        raise SamplerDeadEnd(f"intro pool has no entries for depth <= {depth}")
    d_intro = int(rng.choice(intro_depths))
    intro_items = list(intro_pool[d_intro].items())
    intro_weights = np.array([score for _, score in intro_items], dtype=np.float64)
    intro = intro_items[int(rng.choice(len(intro_items), p=intro_weights / intro_weights.sum()))][0]
    remaining = depth - d_intro
    if remaining == 0:
        return intro
    if not _pool_nonempty_depths(inherited_pool):
        raise SamplerDeadEnd("inherited pool is empty for intro composition")
    left_depth = int(rng.integers(0, remaining + 1))
    right_depth = remaining - left_depth
    out: AbstractProgram = ()
    if left_depth:
        out = sample_abstract_program(
            inherited_pool, library.bigram, primitives, left_depth, rng,
        )
    from babytheorist.dominant import _with_fresh_scopes  # internal composition helper
    out = _with_fresh_scopes(out, intro)
    if right_depth:
        suffix = sample_abstract_program(
            inherited_pool, library.bigram, primitives, right_depth, rng,
        )
        out = _with_fresh_scopes(out, suffix)
    return out


def _fill_refs_require_hollow(program: AbstractProgram) -> dict[AbstractRef, str]:
    return {
        step.source: "hollow"
        for step in program
        if step.primitive == "fill_interior"
    }


def _initial_grid_shapes_for_abstract(
    abstract: AbstractProgram,
    n_shapes: int,
    fallback: list[str] | None,
) -> tuple[list[str] | None, dict[AbstractRef, str] | None]:
    fill_refs = _fill_refs_require_hollow(abstract)
    if not fill_refs:
        return fallback, None
    if len(fill_refs) > n_shapes:
        raise SamplerDeadEnd(
            f"fill_interior intro needs {len(fill_refs)} hollow slots, got {n_shapes}"
        )
    initial = ["solid"] * n_shapes
    for i, _ref in enumerate(sorted(fill_refs)):
        initial[i] = "hollow"
    return initial, fill_refs


def _sample_from_dominant_library(
    library: DominantLibrary,
    level_cfg: LevelConfig,
    depth: int,
    rng: np.random.Generator,
) -> tuple[AbstractProgram, str]:
    curriculum = level_cfg.curriculum
    if curriculum is None:
        raise ValueError("segmented dominant sampling requires a curriculum")
    inherited_pool = _merge_pool_segments(library, curriculum.inherited_segments)
    intro_pool = (
        library.segments.get(curriculum.current_intro_segment or "", {})
        if curriculum.current_intro_segment else {}
    )
    if depth == 1:
        active_segments = curriculum.inherited_segments + (
            (curriculum.current_intro_segment,)
            if curriculum.current_intro_segment else ()
        )
        return (
            sample_abstract_program(
                _merge_pool_segments(library, active_segments),
                library.bigram,
                level_cfg.primitives,
                depth,
                rng,
            ),
            "depth1_uniform",
        )
    if (
        curriculum.current_intro_segment
        and intro_pool
        and rng.random() < curriculum.intro_sampling_rate
    ):
        return (
            _sample_intro_abstract_program(
                library, inherited_pool, intro_pool,
                level_cfg.primitives, depth, rng,
            ),
            "intro",
        )
    return (
        sample_abstract_program(
            inherited_pool, library.bigram, level_cfg.primitives, depth, rng,
        ),
        "inherited",
    )


def _ensure_abstract_pool(
    level_cfg,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    *,
    bigram_weights=None,
    initial_grid_shapes=None,
    pool_seed=None,
    dominant_pool_dir=None,
):
    """Build, load, or fetch the process-local abstract dominant pool."""
    pool_initial_grid_shapes = _pool_initial_grid_shapes(initial_grid_shapes)
    key = _abstract_pool_key(
        level_cfg, n_shapes, bigram_weights, pool_initial_grid_shapes, pool_seed,
    )
    if key not in _ABSTRACT_POOL_CACHE:
        bigram = build_bigram(level_cfg.primitives, bigram_weights=bigram_weights)
        metadata = _dominant_pool_cache_metadata(
            level_cfg,
            n_shapes,
            bigram_weights,
            pool_initial_grid_shapes,
            pool_seed,
        )
        pool = _load_cached_abstract_pool(dominant_pool_dir, metadata)
        if pool is None:
            pool_rng = (
                np.random.default_rng(pool_seed)
                if pool_seed is not None else None
            )
            pool = build_abstract_pool(
                bigram,
                level_cfg.primitives,
                max_depth=_dominant_pool_max_depth(level_cfg),
                n_shapes=n_shapes,
                selector_types=selector_types,
                vocab=vocab,
                primitive_weights=level_cfg.primitive_weights,
                rng=pool_rng,
                initial_grid_shapes=pool_initial_grid_shapes,
                palette=level_cfg.palette,
            )
            _save_cached_abstract_pool(dominant_pool_dir, metadata, pool)
        _ABSTRACT_POOL_CACHE[key] = (bigram, pool)
    return _ABSTRACT_POOL_CACHE[key]


def _dominant_pool_shape_counts(task_cfg: TaskConfig) -> tuple[int, ...]:
    lo_n, hi_n = task_cfg.n_shapes_range
    return tuple(range(lo_n, hi_n + 1))


def _dominant_pool_initial_grid_shapes(
    n_shapes: int, initial_hollow_prob: float,
) -> tuple[list[str] | None, ...]:
    if initial_hollow_prob <= 0.0:
        return (None,)
    all_hollow = ["hollow"] * n_shapes
    if initial_hollow_prob >= 1.0:
        return (all_hollow,)
    return (None, all_hollow)


def _prewarm_filter_dominant_pools(
    level_cfg: LevelConfig,
    task_cfg: TaskConfig,
    initial_hollow_prob: float,
    *,
    bigram_weights=None,
    pool_seed=None,
    dominant_pool_dir=None,
) -> None:
    """Populate parent-process dominant pools before worker fork."""
    if not level_cfg.use_dominant:
        return
    selector_types = frozenset(level_cfg.selectors)
    vocab = frozenset(level_cfg.primitives)
    for n_shapes in _dominant_pool_shape_counts(task_cfg):
        if level_cfg.curriculum is not None:
            _ensure_dominant_library(
                level_cfg,
                n_shapes,
                selector_types,
                vocab,
                bigram_weights=bigram_weights,
                pool_seed=pool_seed,
                dominant_pool_dir=dominant_pool_dir,
                task_cfg=task_cfg,
            )
            continue
        for initial_grid_shapes in _dominant_pool_initial_grid_shapes(
            n_shapes, initial_hollow_prob,
        ):
            _ensure_abstract_pool(
                level_cfg,
                n_shapes,
                selector_types,
                vocab,
                bigram_weights=bigram_weights,
                initial_grid_shapes=initial_grid_shapes,
                pool_seed=pool_seed,
                dominant_pool_dir=dominant_pool_dir,
            )


__all__ = [
    "_ABSTRACT_POOL_CACHE",
    "DominantLibrary",
    "FilterGenerationStats",
    "_pool_initial_grid_shapes",
    "_abstract_pool_key",
    "_dominant_pool_cache_metadata",
    "_dominant_pool_cache_path",
    "_load_cached_abstract_pool",
    "_save_cached_abstract_pool",
    "_dominant_library_key",
    "_curriculum_metadata",
    "_dominant_library_cache_metadata",
    "_dominant_library_cache_path",
    "_load_cached_dominant_library",
    "_save_cached_dominant_library",
    "_append_initial_grid_shapes",
    "_merge_segment_dict",
    "_merge_segment_entry_ids",
    "_synthetic_pool_provenance",
    "_pool_build_parts",
    "_dominant_viability_task_signature",
    "_dominant_viability_signature",
    "_dominant_viability_metadata",
    "_dominant_viability_trials",
    "_dominant_preflight_seed",
    "_abstract_has_generation_preflight_success",
    "_dominant_entry_filter_factory",
    "_ensure_dominant_library",
    "_merge_pool_segments",
    "_pool_nonempty_depths",
    "_sample_intro_abstract_program",
    "_fill_refs_require_hollow",
    "_initial_grid_shapes_for_abstract",
    "_sample_from_dominant_library",
    "_ensure_abstract_pool",
    "_dominant_pool_shape_counts",
    "_dominant_pool_initial_grid_shapes",
    "_prewarm_filter_dominant_pools",
]
