"""Filter-backend task and dataset orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from babytheorist.core import (
    Shape, Task, TaskPair,
)
from babytheorist.filter.constraint import InfeasibleConstraint
from babytheorist.dominant import (
    sample_abstract_program,
)
from babytheorist.program import (
    SamplerDeadEnd,
)

from babytheorist.generation.config import (
    LevelConfig,
    TaskConfig,
    _ATTEMPT_SCHEDULERS,
    _FILTER_PAIR_EXECUTION_RETRIES,
    _FILTER_SYSTEM,
    _sample_depth,
    _validate_level_cfg,
)
from babytheorist.generation.dominant import (
    FilterGenerationStats,
    _ensure_abstract_pool,
    _ensure_dominant_library,
    _initial_grid_shapes_for_abstract,
    _prewarm_filter_dominant_pools,
    _sample_from_dominant_library,
)
from babytheorist.generation.filter_pairs import (
    _filter_generate_module,
    _filter_sampler_module,
    _filter_task_from_mpt,
    _program_grounding_signature,
    _resolve_int_or_range,
    _sample_filter_pair_with_reason,
    _with_raster_order_constraint,
)
from babytheorist.generation.rejection import (
    _FILTER_PARALLEL_BATCH_PER_WORKER,
    _accept_unique,
    _seed_batches,
    _shape_factory_from_task_config,
    _task_from_pairs_with_shapes,
)

_fw = SimpleNamespace()  # populated in each filter worker process


class _FilterAttemptTimeoutError(BaseException):
    pass


@dataclass(frozen=True)
class _FilterAttemptTimedOut:
    timeout_sec: float


def _sample_filter_program_with_optional_dominant(
    level_cfg,
    task_cfg,
    depth,
    n_shapes,
    vocab,
    selector_types,
    rng,
    initial_grid_shapes,
    bigram_weights=None,
    pool_seed=None,
    dominant_pool_dir=None,
    stats: FilterGenerationStats | None = None,
):
    sample_program_with_constraint = (
        _filter_sampler_module().sample_program_with_constraint
    )

    def sample_with_stats(**kwargs):
        sampler_stats = None
        if stats is not None:
            from babytheorist.filter.types import SamplerStats
            sampler_stats = SamplerStats()
            kwargs["stats"] = sampler_stats
        if stats is not None:
            stats.sampler_calls += 1
        t0 = time.perf_counter()
        try:
            return sample_program_with_constraint(**kwargs)
        finally:
            if stats is not None:
                stats.sampler_elapsed_sec += time.perf_counter() - t0
            if stats is not None and sampler_stats is not None:
                stats.merge_sampler(sampler_stats)

    if not level_cfg.use_dominant:
        return sample_with_stats(
            depth=depth,
            n_shapes=n_shapes,
            vocab=vocab,
            selector_types=selector_types,
            rng=rng,
            initial_grid_shapes=initial_grid_shapes,
            palette=level_cfg.palette,
        )

    if level_cfg.curriculum is not None:
        library = _ensure_dominant_library(
            level_cfg,
            n_shapes,
            selector_types,
            vocab,
            bigram_weights=bigram_weights,
            pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
            task_cfg=task_cfg,
        )
        for _ in range(20):
            if stats is not None:
                stats.dominant_grounding_attempts += 1
            abstract, _branch = _sample_from_dominant_library(
                library, level_cfg, depth, rng,
            )
            try:
                intro_grid_shapes, ref_grid_shapes = _initial_grid_shapes_for_abstract(
                    abstract, n_shapes, initial_grid_shapes,
                )
                return sample_with_stats(
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
            except SamplerDeadEnd:
                if stats is not None:
                    stats.dominant_grounding_dead_ends += 1
                continue
        raise SamplerDeadEnd(
            f"segmented dominant grounding exhausted for depth={depth}, "
            f"n_shapes={n_shapes}"
        )

    bigram, pool = _ensure_abstract_pool(
        level_cfg,
        n_shapes,
        selector_types,
        vocab,
        bigram_weights=bigram_weights,
        initial_grid_shapes=initial_grid_shapes,
        pool_seed=pool_seed,
        dominant_pool_dir=dominant_pool_dir,
    )
    for _ in range(20):
        if stats is not None:
            stats.dominant_grounding_attempts += 1
        abstract = sample_abstract_program(
            pool, bigram, level_cfg.primitives, depth, rng,
        )
        try:
            return sample_with_stats(
                depth=depth,
                n_shapes=n_shapes,
                vocab=vocab,
                selector_types=selector_types,
                rng=rng,
                initial_grid_shapes=initial_grid_shapes,
                palette=level_cfg.palette,
                abstract_program=abstract,
            )
        except SamplerDeadEnd:
            if stats is not None:
                stats.dominant_grounding_dead_ends += 1
            continue
    raise SamplerDeadEnd(
        f"abstract dominant grounding exhausted for depth={depth}, n_shapes={n_shapes}"
    )


def _generate_unique_filter_task(
    level_cfg, task_cfg, rng, depth, n_shapes, factory, vocab,
    selector_types, initial_hollow_prob, bigram_weights=None, pool_seed=None,
    dominant_pool_dir=None,
    stats: FilterGenerationStats | None = None,
):
    uses_by_index = "by_index" in selector_types
    for _ in range(task_cfg.max_program_retries):
        if stats is not None:
            stats.program_attempts += 1
        n_shapes_k = _resolve_int_or_range(n_shapes, rng)
        initial_grid_shapes = [
            "hollow" if rng.random() < initial_hollow_prob else "solid"
            for _ in range(n_shapes_k)
        ]
        try:
            program, constraint = _sample_filter_program_with_optional_dominant(
                level_cfg, task_cfg, depth, n_shapes_k, vocab, selector_types, rng,
                initial_grid_shapes, bigram_weights=bigram_weights,
                pool_seed=pool_seed,
                dominant_pool_dir=dominant_pool_dir,
                stats=stats,
            )
            if uses_by_index:
                constraint = _with_raster_order_constraint(constraint)
        except SamplerDeadEnd:
            if stats is not None:
                stats.program_dead_ends += 1
            continue

        grounding_signature = (
            _program_grounding_signature(program) if stats is not None else None
        )
        if stats is not None:
            stats.record_grounding_signature(grounding_signature)

        if stats is not None:
            stats.pair_attempts += 1
        t_pair = time.perf_counter()
        query, failure_reason = _sample_filter_pair_with_reason(
            constraint, program, factory, level_cfg, task_cfg, rng,
            uses_by_index=uses_by_index,
            gap=task_cfg.gap,
            execution_retries=_FILTER_PAIR_EXECUTION_RETRIES,
        )
        if stats is not None:
            stats.pair_elapsed_sec += time.perf_counter() - t_pair
        if query is None:
            if stats is not None:
                stats.record_pair_failure(failure_reason, grounding_signature)
            continue

        supports: list[tuple[TaskPair, list[Shape]]] = []
        for _support_idx in range(task_cfg.max_support):
            if stats is not None:
                stats.pair_attempts += 1
            t_pair = time.perf_counter()
            support, failure_reason = _sample_filter_pair_with_reason(
                constraint, program, factory, level_cfg, task_cfg, rng,
                uses_by_index=uses_by_index,
                gap=task_cfg.gap,
                execution_retries=_FILTER_PAIR_EXECUTION_RETRIES,
            )
            if stats is not None:
                stats.pair_elapsed_sec += time.perf_counter() - t_pair
            if support is None:
                if stats is not None:
                    stats.record_pair_failure(
                        failure_reason, grounding_signature,
                    )
                break
            supports.append(support)
            task = _task_from_pairs_with_shapes(
                program, supports + [query], level_cfg, task_cfg, uses_by_index,
            )
            if task is not None:
                if stats is not None:
                    stats.record_grounding_success(grounding_signature)
                return task
    return None


def _generate_filter_multipair_task(
    level_cfg, task_cfg, rng, depth, n_shapes, factory, vocab, selector_types,
    initial_hollow_prob, bigram_weights=None, pool_seed=None,
    dominant_pool_dir=None,
    stats: FilterGenerationStats | None = None,
):
    uses_by_index = "by_index" in selector_types
    for _ in range(task_cfg.max_program_retries):
        if stats is not None:
            stats.program_attempts += 1
        n_shapes_k = _resolve_int_or_range(n_shapes, rng)
        initial_grid_shapes = [
            "hollow" if rng.random() < initial_hollow_prob else "solid"
            for _ in range(n_shapes_k)
        ]
        try:
            program, constraint = _sample_filter_program_with_optional_dominant(
                level_cfg, task_cfg, depth, n_shapes_k, vocab, selector_types, rng,
                initial_grid_shapes, bigram_weights=bigram_weights,
                pool_seed=pool_seed,
                dominant_pool_dir=dominant_pool_dir,
                stats=stats,
            )
            if uses_by_index:
                constraint = _with_raster_order_constraint(constraint)
        except (SamplerDeadEnd, InfeasibleConstraint):
            if stats is not None:
                stats.program_dead_ends += 1
            continue

        grounding_signature = (
            _program_grounding_signature(program) if stats is not None else None
        )
        if stats is not None:
            stats.record_grounding_signature(grounding_signature)

        pairs: list[tuple[np.ndarray, list[Shape], np.ndarray]] = []
        for _pair_idx in range(task_cfg.n_pairs):
            if stats is not None:
                stats.pair_attempts += 1
            t_pair = time.perf_counter()
            pair, failure_reason = _sample_filter_pair_with_reason(
                constraint, program, factory, level_cfg, task_cfg, rng,
                uses_by_index=uses_by_index,
                gap=task_cfg.gap,
                execution_retries=_FILTER_PAIR_EXECUTION_RETRIES,
            )
            if stats is not None:
                stats.pair_elapsed_sec += time.perf_counter() - t_pair
            if pair is None:
                if stats is not None:
                    stats.record_pair_failure(
                        failure_reason, grounding_signature,
                    )
                break
            task_pair, input_shapes = pair
            pairs.append((task_pair.input_grid, input_shapes, task_pair.output_grid))

        if len(pairs) == task_cfg.n_pairs:
            MultiPairTask = _filter_generate_module().MultiPairTask
            mpt = MultiPairTask(program=program, constraint=constraint, pairs=pairs)
            task = _filter_task_from_mpt(mpt, level_cfg, task_cfg, uses_by_index)
            if task is None:
                continue
            if stats is not None:
                stats.record_grounding_success(grounding_signature)
            return task
    return None


def _generate_filter_attempt(
    level_cfg, task_cfg, seed, initial_hollow_prob, bigram_weights=None,
    pool_seed=None, dominant_pool_dir=None,
    stats: FilterGenerationStats | None = None,
):
    vocab = frozenset(level_cfg.primitives)
    selector_types = frozenset(level_cfg.selectors)
    lo_n, hi_n = task_cfg.n_shapes_range
    n_shapes = lo_n if lo_n == hi_n else (lo_n, hi_n)
    rng = np.random.default_rng(seed)
    depth = _sample_depth(level_cfg, rng)
    factory = _shape_factory_from_task_config(task_cfg)
    if task_cfg.mode == "eval" and task_cfg.eval_unique_support:
        return _generate_unique_filter_task(
            level_cfg, task_cfg, rng, depth, n_shapes, factory, vocab,
            selector_types, initial_hollow_prob, bigram_weights=bigram_weights,
            pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
            stats=stats,
        )
    return _generate_filter_multipair_task(
        level_cfg, task_cfg, rng, depth, n_shapes, factory, vocab,
        selector_types, initial_hollow_prob, bigram_weights=bigram_weights,
        pool_seed=pool_seed,
        dominant_pool_dir=dominant_pool_dir,
        stats=stats,
    )


def _generate_filter_attempt_with_timeout(
    level_cfg, task_cfg, seed, initial_hollow_prob, bigram_weights=None,
    pool_seed=None, dominant_pool_dir=None,
    stats: FilterGenerationStats | None = None,
    attempt_timeout_sec: float | None = None,
):
    if attempt_timeout_sec is None:
        return _generate_filter_attempt(
            level_cfg, task_cfg, seed, initial_hollow_prob,
            bigram_weights=bigram_weights,
            pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
            stats=stats,
        )
    if attempt_timeout_sec <= 0.0:
        raise ValueError(
            f"attempt_timeout_sec must be > 0 when set, got {attempt_timeout_sec}"
        )
    if not hasattr(signal, "SIGALRM"):
        raise ValueError("attempt_timeout_sec requires SIGALRM support")

    def raise_timeout(_signum, _frame):
        raise _FilterAttemptTimeoutError

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(attempt_timeout_sec))
    try:
        return _generate_filter_attempt(
            level_cfg, task_cfg, seed, initial_hollow_prob,
            bigram_weights=bigram_weights,
            pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
            stats=stats,
        )
    except _FilterAttemptTimeoutError:
        return _FilterAttemptTimedOut(float(attempt_timeout_sec))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _init_filter_worker(
    level_cfg, task_cfg, initial_hollow_prob, bigram_weights, pool_seed,
    dominant_pool_dir, attempt_timeout_sec,
):
    global _fw
    selector_types = frozenset(level_cfg.selectors)
    lo_n, hi_n = task_cfg.n_shapes_range
    _fw = SimpleNamespace(
        level_cfg=level_cfg,
        task_cfg=task_cfg,
        factory=_shape_factory_from_task_config(task_cfg),
        vocab=frozenset(level_cfg.primitives),
        selector_types=selector_types,
        n_shapes=lo_n if lo_n == hi_n else (lo_n, hi_n),
        n_pairs=task_cfg.n_pairs,
        uses_by_index="by_index" in selector_types,
        initial_hollow_prob=initial_hollow_prob,
        bigram_weights=bigram_weights,
        pool_seed=pool_seed,
        dominant_pool_dir=dominant_pool_dir,
        attempt_timeout_sec=attempt_timeout_sec,
    )


def _filter_worker_task(seed):
    return _generate_filter_attempt_with_timeout(
        _fw.level_cfg, _fw.task_cfg, seed, _fw.initial_hollow_prob,
        bigram_weights=_fw.bigram_weights,
        pool_seed=_fw.pool_seed,
        dominant_pool_dir=_fw.dominant_pool_dir,
        attempt_timeout_sec=_fw.attempt_timeout_sec,
    )


def _sequential_filter_attempts(
    level_cfg, task_cfg, seed, max_attempts, initial_hollow_prob,
    bigram_weights=None, pool_seed=None, dominant_pool_dir=None,
    attempt_timeout_sec: float | None = None,
    stats: FilterGenerationStats | None = None,
):
    for batch in _seed_batches(seed, max_attempts):
        for attempt_seed in batch:
            yield _generate_filter_attempt_with_timeout(
                level_cfg, task_cfg, attempt_seed, initial_hollow_prob,
                bigram_weights=bigram_weights, pool_seed=pool_seed,
                dominant_pool_dir=dominant_pool_dir,
                attempt_timeout_sec=attempt_timeout_sec,
                stats=stats,
            )


def _parallel_filter_attempts(
    level_cfg, task_cfg, seed, max_attempts, n_workers, initial_hollow_prob,
    bigram_weights=None, pool_seed=None, dominant_pool_dir=None,
    attempt_scheduler="ordered",
    attempt_timeout_sec: float | None = None,
):
    from multiprocessing import Pool

    if attempt_scheduler not in _ATTEMPT_SCHEDULERS:
        raise ValueError(
            f"attempt_scheduler must be one of {_ATTEMPT_SCHEDULERS}, "
            f"got {attempt_scheduler!r}"
        )
    batch_size = max(n_workers, n_workers * _FILTER_PARALLEL_BATCH_PER_WORKER)
    p = Pool(
        n_workers, initializer=_init_filter_worker,
        initargs=(
            level_cfg, task_cfg, initial_hollow_prob, bigram_weights, pool_seed,
            dominant_pool_dir, attempt_timeout_sec,
        ),
    )
    try:
        for batch in _seed_batches(seed, max_attempts, batch_size=batch_size):
            if attempt_scheduler == "unordered":
                yield from p.imap_unordered(_filter_worker_task, batch, chunksize=1)
            else:
                yield from p.imap(_filter_worker_task, batch, chunksize=1)
    except GeneratorExit:
        p.terminate()
        p.join()
        raise
    except BaseException:
        p.terminate()
        p.join()
        raise
    else:
        p.close()
        p.join()


def generate_filter_dataset(
    level_cfg: LevelConfig, task_cfg: TaskConfig | None = None,
    n_tasks: int = 100, seed: int | None = None,
    initial_hollow_prob: float = 0.0,
    n_workers: int = 1,
    attempt_scheduler: str = "ordered",
    attempt_timeout_sec: float | None = None,
    bigram_weights: dict[tuple[str, str], float] | None = None,
    pool_seed: int | None = None,
    dominant_pool_dir: str | Path | None = None,
    seen_pair_hashes: set[bytes] | None = None,
    stats: FilterGenerationStats | None = None,
) -> list[Task]:
    """Filter-backend dataset generation.

    Uses the constraint-forward rule stack (R1/R3/R5/R5b/R6/R7' + I1)
    instead of the rejection-based ``_generate_pair`` path. When
    ``level_cfg.use_dominant`` is true, primitive draws come from the
    abstract dominant pool and concrete grounding is still rechecked by
    the same filter stack.

    Preserves the same ``Task`` contract (including ``valid_query_outputs``
    for eval mode).
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if task_cfg is None:
        task_cfg = TaskConfig()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    if attempt_scheduler not in _ATTEMPT_SCHEDULERS:
        raise ValueError(
            f"attempt_scheduler must be one of {_ATTEMPT_SCHEDULERS}, "
            f"got {attempt_scheduler!r}"
        )
    if attempt_timeout_sec is not None and attempt_timeout_sec <= 0.0:
        raise ValueError(
            f"attempt_timeout_sec must be > 0 when set, got {attempt_timeout_sec}"
        )
    if not 0.0 <= initial_hollow_prob <= 1.0:
        raise ValueError(
            f"initial_hollow_prob must be in [0, 1], got {initial_hollow_prob}"
        )
    _validate_level_cfg(level_cfg)
    if stats is not None:
        stats.filter_system = _FILTER_SYSTEM
        stats.requested_tasks = n_tasks
        stats.workers = n_workers

    if seen_pair_hashes is None:
        seen_pair_hashes = set()
    tasks: list[Task] = []
    max_attempts = n_tasks * task_cfg.max_dataset_attempts_per_task
    t0 = time.perf_counter()
    if n_workers > 1:
        _prewarm_filter_dominant_pools(
            level_cfg,
            task_cfg,
            initial_hollow_prob,
            bigram_weights=bigram_weights,
            pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
        )
    if n_workers <= 1:
        attempts = _sequential_filter_attempts(
            level_cfg, task_cfg, seed, max_attempts, initial_hollow_prob,
            bigram_weights=bigram_weights, pool_seed=pool_seed,
            dominant_pool_dir=dominant_pool_dir,
            attempt_timeout_sec=attempt_timeout_sec,
            stats=stats,
        )
    else:
        attempts = _parallel_filter_attempts(
            level_cfg, task_cfg, seed, max_attempts, n_workers,
            initial_hollow_prob, bigram_weights=bigram_weights,
            pool_seed=pool_seed, dominant_pool_dir=dominant_pool_dir,
            attempt_scheduler=attempt_scheduler,
            attempt_timeout_sec=attempt_timeout_sec,
        )
    with tqdm(total=n_tasks, desc=f"L{level_cfg.level}", unit="task") as bar:
        for task in attempts:
            if stats is not None:
                stats.dataset_attempts += 1
            if isinstance(task, _FilterAttemptTimedOut):
                if stats is not None:
                    stats.attempt_timeouts += 1
                    stats.none_attempts += 1
                    bar.set_postfix(
                        attempts=stats.dataset_attempts,
                        rejected=stats.none_attempts + stats.duplicate_rejects,
                        timeouts=stats.attempt_timeouts,
                        refresh=False,
                    )
                continue
            if task is None:
                if stats is not None:
                    stats.none_attempts += 1
                    bar.set_postfix(
                        attempts=stats.dataset_attempts,
                        rejected=stats.none_attempts + stats.duplicate_rejects,
                        timeouts=stats.attempt_timeouts,
                        refresh=False,
                    )
                continue
            if not _accept_unique(task, seen_pair_hashes):
                if stats is not None:
                    stats.duplicate_rejects += 1
                    bar.set_postfix(
                        attempts=stats.dataset_attempts,
                        rejected=stats.none_attempts + stats.duplicate_rejects,
                        timeouts=stats.attempt_timeouts,
                        refresh=False,
                    )
                continue
            tasks.append(task)
            if stats is not None:
                stats.produced_tasks = len(tasks)
                bar.set_postfix(
                    attempts=stats.dataset_attempts,
                    rejected=stats.none_attempts + stats.duplicate_rejects,
                    timeouts=stats.attempt_timeouts,
                    refresh=False,
                )
            bar.update(1)
            if len(tasks) >= n_tasks:
                break
    close_attempts = getattr(attempts, "close", None)
    if close_attempts is not None:
        close_attempts()
    if stats is not None:
        stats.elapsed_sec = time.perf_counter() - t0

    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"generate_filter_dataset: only produced {len(tasks)}/{n_tasks} "
            f"tasks after {max_attempts} attempts (level={level_cfg.level})"
        )
    return tasks


__all__ = [
    "_fw",
    "_sample_filter_program_with_optional_dominant",
    "_generate_unique_filter_task",
    "_generate_filter_multipair_task",
    "_generate_filter_attempt",
    "_generate_filter_attempt_with_timeout",
    "_FilterAttemptTimeoutError",
    "_FilterAttemptTimedOut",
    "_init_filter_worker",
    "_filter_worker_task",
    "_sequential_filter_attempts",
    "_parallel_filter_attempts",
    "generate_filter_dataset",
]
