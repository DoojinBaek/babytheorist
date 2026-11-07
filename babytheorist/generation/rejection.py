"""Legacy rejection-based BabyTheorist task generation."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tqdm import tqdm

from babytheorist.canonical import (
    is_minimal_tier1, is_minimal_tier2, is_minimal_tier3,
    is_minimal_multi_shape_by_index, is_minimal_tier4_swap,
    is_minimal_tier5_align,
)
from babytheorist.core import (
    Program, Selector, Shape, Step, Task, TaskPair,
    MULTI_SELECTORS, SINGLE_SELECTORS,
)
from babytheorist.engine import apply_program
from babytheorist.evaluation import (
    compute_valid_query_outputs,
    compute_valid_query_outputs_from_supports,
    enumerate_selector_pool,
)
from babytheorist.predicates import scene_verdict
from babytheorist.primitives import category_of
from babytheorist.program import (
    SamplerDeadEnd, build_bigram, no_dead_writes, trajectory_distinct,
)
from babytheorist.shapes import ShapeFactory, place_shapes

from babytheorist.generation.config import (
    LevelConfig,
    TaskConfig,
    _sample_depth,
    _validate_level_cfg,
)

def _sample_selector(
    allowed: tuple[str, ...], palette: tuple[int, ...], rng: np.random.Generator,
    n_shapes: int,
) -> Selector:
    """Pick one `Selector` uniformly from the allowed types.

    `n_shapes` bounds the 1-based `by_index` param so every sampled selector
    hits a real object at runtime.
    """
    t = str(rng.choice(list(allowed)))
    if t == "by_color":
        return Selector("by_color", param=int(rng.choice(palette)))
    if t == "by_index":
        return Selector("by_index", param=int(rng.integers(1, n_shapes + 1)))
    return Selector(t)


_SYMBOLIC_SELECTOR_TYPES = frozenset({"ALL", "by_index", "is_hollow"})


def _has_dynamic_selector(program: Program) -> bool:
    """True iff any step uses a selector the symbolic layer can't
    resolve — currently ``topmost``/``bottommost``/``leftmost``/
    ``rightmost``/``by_color``. ``is_hollow`` is supported via
    symbolic ``grid_shape``. Phase 1 filtering bypasses programs with
    unsupported selectors and defers to Phase 2 scene-level checks."""
    for step in program:
        if step.selector.type not in _SYMBOLIC_SELECTOR_TYPES:
            return True
        if step.target is not None and step.target.type not in _SYMBOLIC_SELECTOR_TYPES:
            return True
    return False


def _passes_phase1_filter(
    program: Program,
    n_shapes: int,
    vocab: frozenset[str],
    *,
    grid_shapes: list[str] | None = None,
) -> bool:
    """Run all Phase 1 filters: trajectory distinct + no dead writes +
    Tier 1 canonical minimality. Returns True iff the program passes
    every applicable rule.

    Programs with dynamic selectors skip filtering (V1 symbolic layer
    only handles ``ALL`` and ``by_index``); those are caught later by
    Phase 2 checks in ``_generate_pair``.
    """
    if _has_dynamic_selector(program):
        return True
    if not trajectory_distinct(program, n_shapes, grid_shapes=grid_shapes):
        return False
    if not no_dead_writes(program, n_shapes, grid_shapes=grid_shapes):
        return False
    if not is_minimal_tier1(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    if not is_minimal_tier2(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    if not is_minimal_tier3(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    if not is_minimal_multi_shape_by_index(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    if not is_minimal_tier4_swap(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    if not is_minimal_tier5_align(
        program, n_shapes, vocab=vocab, grid_shapes=grid_shapes
    ):
        return False
    return True


def random_program_filtered(
    depth: int,
    primitives: list[str],
    selectors: tuple[str, ...],
    palette: tuple[int, ...],
    bigram: np.ndarray,
    rng: np.random.Generator,
    n_shapes: int,
    *,
    primitive_weights: dict[str, float] | None = None,
    grid_shapes: list[str] | None = None,
    max_attempts_per_step: int = 50,
) -> Program:
    """Forward-incremental sampler with Phase 1 rule filtering.

    At each step, draws a primitive from the bigram (conditioned on the
    previous primitive), draws selectors, and tries to append a
    ``Step``. If the extended program fails any of
    ``trajectory_distinct`` / ``no_dead_writes`` / ``is_minimal_tier1``,
    resamples just that step up to ``max_attempts_per_step`` times. If
    every retry fails, raises ``SamplerDeadEnd`` — the outer loop in
    ``generate_task`` then retries the whole program with a fresh RNG.

    Stronger than the old full-sequence sampler: every prefix must pass
    the complete Phase 1 filter before the next step is sampled.

    Programs with dynamic selectors (``topmost``/``by_color``/etc.)
    bypass filtering — V1 symbolic state only handles ``ALL`` and
    ``by_index``. Scene-level checks in ``_generate_pair`` pick up the
    slack for those.
    """
    if len(set(primitives)) != len(primitives):
        raise ValueError(f"duplicate primitives: {primitives!r}")

    idx = {p: i for i, p in enumerate(primitives)}
    vocab = frozenset(primitives)
    weights_map = primitive_weights or {}
    weight_vec = np.array(
        [weights_map.get(p, 1.0) for p in primitives], dtype=np.float64,
    )
    single_only = tuple(s for s in selectors if s in SINGLE_SELECTORS)

    program: Program = []
    for step_idx in range(depth):
        prev = program[-1].primitive if program else None
        probs = (bigram[idx[prev]] if prev is not None else np.ones(len(primitives))) * weight_vec
        total = probs.sum()
        if total <= 0:
            raise SamplerDeadEnd(
                f"step {step_idx}: zero-weight primitive distribution (prev={prev!r})"
            )
        probs = probs / total

        accepted: Step | None = None
        for _ in range(max_attempts_per_step):
            name = primitives[int(rng.choice(len(primitives), p=probs))]
            try:
                if category_of(name) == "relational":
                    if not single_only:
                        continue
                    src = _sample_selector(single_only, palette, rng, n_shapes)
                    tgt = _sample_selector(single_only, palette, rng, n_shapes)
                    if src == tgt:
                        continue
                    step = Step(src, name, target=tgt)
                else:
                    sel = _sample_selector(selectors, palette, rng, n_shapes)
                    step = Step(sel, name)
            except ValueError:
                continue

            candidate = program + [step]
            if not _passes_phase1_filter(
                candidate, n_shapes, vocab, grid_shapes=grid_shapes
            ):
                continue
            accepted = step
            break

        if accepted is None:
            raise SamplerDeadEnd(
                f"step {step_idx}: no valid continuation after "
                f"{max_attempts_per_step} attempts"
            )
        program.append(accepted)
    return program


# === Phase 2 scene-level shortcut enumeration =============================


def _enum_q_steps(
    primitives: tuple[str, ...], selectors: tuple[str, ...],
    n_shapes: int, palette: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> list[Step]:
    """All valid ``Step`` instances for the given vocab + selectors.

    Selector expansion: ``ALL``, ``is_hollow``, ``topmost``, ``bottommost``,
    ``leftmost``, ``rightmost`` each contribute one instance. ``by_index``
    expands to ``by_index(1..n_shapes)``. ``by_color`` expands over
    ``palette``.

    Relational primitives pair only single-selectors (relational
    ``Step.__post_init__`` rejects multi-selectors) and skip self-pairs
    (precondition-fail by construction).
    """
    multi_types = [s for s in selectors if s in MULTI_SELECTORS]
    single_types = [s for s in selectors if s in SINGLE_SELECTORS]

    def _sel_instances(types):
        out: list[Selector] = []
        for t in types:
            if t == "by_index":
                for k in range(1, n_shapes + 1):
                    out.append(Selector("by_index", param=k))
            elif t == "by_color":
                for c in palette:
                    out.append(Selector("by_color", param=int(c)))
            else:
                # ALL, is_hollow, topmost, bottommost, leftmost, rightmost.
                out.append(Selector(t))
        return out

    multi_sels = _sel_instances(multi_types)
    single_sels = _sel_instances(single_types)

    steps: list[Step] = []
    for p in primitives:
        if category_of(p) == "relational":
            for src in single_sels:
                for tgt in single_sels:
                    if src == tgt:
                        continue
                    steps.append(Step(src, p, target=tgt))
        else:
            for sel in multi_sels + single_sels:
                steps.append(Step(sel, p))
    return steps


@lru_cache(maxsize=None)
def _cached_shorter_programs(
    primitives: tuple[str, ...], selectors: tuple[str, ...],
    n_shapes: int, depth: int, cap: int,
    palette: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9),
) -> tuple[tuple[Step, ...], ...]:
    """Enumerate every length-1..depth-1 program from the vocab.

    Returns a tuple of tuples (``Step`` is hashable). Cached by
    (primitives, selectors, n_shapes, depth, cap, palette) — the
    enumeration is deterministic and expensive enough at depth ≥ 4 that
    we want to amortise it across pairs/scenes with the same vocab.

    If the enumerated program count would exceed ``cap``, returns an
    empty tuple. Callers treat that as "scene check disabled".
    """
    if depth <= 1:
        return ()
    steps = _enum_q_steps(primitives, selectors, n_shapes, palette)
    # Quick size check — ``sum(len(steps) ** k for k in 1..depth-1)`` —
    # before materialising the program list.
    total = 0
    power = 1
    for _ in range(1, depth):
        power *= len(steps)
        total += power
        if total > cap:
            return ()
    programs: list[tuple[Step, ...]] = []
    for k in range(1, depth):
        for combo in product(steps, repeat=k):
            programs.append(combo)
    return tuple(programs)


def _q_unsat_selector(
    q: tuple[Step, ...],
    scene_colors: frozenset[int], scene_has_hollow: bool,
) -> bool:
    """True iff any step in ``q`` has a selector that can't match
    anything on this scene — specifically ``by_color(c)`` with ``c`` not
    present, or ``is_hollow`` when no hollow shapes are present. Such Q
    programs would fail at selector resolution; skipping them avoids a
    wasted ``apply_program`` call.

    Only the ``selector`` slot is checked; relational ``target`` always
    holds a single-selector (topmost/bottommost/leftmost/rightmost/
    by_index by ``Step.__post_init__``), so ``by_color``/``is_hollow``
    cannot appear there.
    """
    for step in q:
        sel = step.selector
        if sel.type == "by_color" and sel.param not in scene_colors:
            return True
        if sel.type == "is_hollow" and not scene_has_hollow:
            return True
    return False


def _scene_primary_colors(positioned: list[Shape]) -> frozenset[int]:
    """Primary colors of every shape in the scene. ``by_color`` selector
    resolves against ``shape.primary_color``, so this is the set of
    ``c`` values for which ``by_color(c)`` can match at least one shape.
    """
    return frozenset(s.primary_color for s in positioned)


def _scene_has_hollow(positioned: list[Shape]) -> bool:
    """True iff at least one shape in the scene is hollow. Mirrors the
    ``is_hollow`` selector's matching logic (``Shape.is_hollow``)."""
    return any(s.is_hollow for s in positioned)


def _scene_has_shorter_shortcut(
    shorter_qs: tuple[tuple[Step, ...], ...],
    positioned: list[Shape], world_size: int, output_P: np.ndarray,
) -> bool:
    """Return True iff some Q in ``shorter_qs`` executes successfully on
    ``positioned`` and produces a grid equal to ``output_P``.

    This is the Phase 2 complement to the Phase 1 symbolic filter:
    symbolically-distinct P and Q can still render to the same output
    grid on a particular scene (e.g. ``rot90`` on a rotationally-
    symmetric shape, or ``[rot90] ≠ [flip_h]`` collapsing on a
    symmetric shape). Caller skips the scene if this returns True.

    Optimisation: skip Q's whose selectors are guaranteed to return no
    shapes on this scene — ``by_color(c)`` with ``c`` not present, or
    ``is_hollow`` when no hollow shapes are in the scene. These Q's
    would fail at ``select`` anyway; the pre-check is O(|q|) and avoids
    the much heavier ``apply_program`` setup.
    """
    scene_colors = _scene_primary_colors(positioned)
    has_hollow = _scene_has_hollow(positioned)
    for q in shorter_qs:
        if _q_unsat_selector(q, scene_colors, has_hollow):
            continue
        output_Q, _ = apply_program(list(q), positioned, world_size)
        if output_Q is None:
            continue
        if np.array_equal(output_Q, output_P):
            return True
    return False


def _resolve_shorter_qs_for_pair(
    level_cfg: LevelConfig, task_cfg: TaskConfig,
    program_depth: int, n_shapes: int,
) -> tuple[tuple[Step, ...], ...]:
    """Build/fetch the cached shorter-program set for this (vocab, depth,
    n_shapes) combo. Returns ``()`` when depth ≤ 1, the Q enumeration
    exceeds cap, or neither scene-check flag requests it.

    The Q set is required by both:
      - ``scene_shortcut_check``: unconditional brute-force filter.
      - ``symbolic_constraints_fallback``: brute-force used only when
        C(P) defers (coverage gap).
    """
    if not (task_cfg.scene_shortcut_check
            or task_cfg.symbolic_constraints_fallback):
        return ()
    if program_depth <= 1:
        return ()
    return _cached_shorter_programs(
        tuple(level_cfg.primitives),
        tuple(level_cfg.selectors),
        n_shapes,
        program_depth,
        task_cfg.scene_shortcut_q_cap,
        tuple(level_cfg.palette),
    )


# === Pair / task / dataset generation =====================================


def _grid_sequence_all_distinct(
    input_grid: np.ndarray, grid_sequence: list[np.ndarray | None],
) -> bool:
    """True iff ``input_grid`` + every grid in ``grid_sequence`` forms a
    strictly distinct adjacent chain (each consecutive pair differs).

    **Phase 2 Rule** — per-step scene-level effectiveness. Engine's
    ``apply_step`` already rejects per-object no-ops (a transform that
    leaves a shape's grid identical, a movement that collides, etc.),
    so normally every adjacent pair differs by construction. This check
    is a defence-in-depth backstop for scene arrangements where the
    engine's object-level change still yields a visually-identical grid
    (rare, but possible under some multi-shape interactions). Combined
    with the ``input_grid != output_grid`` backstop it rejects programs
    that cycle on this scene even when per-step object changes are
    present.
    """
    prev = input_grid
    for g in grid_sequence:
        if g is None or np.array_equal(prev, g):
            return False
        prev = g
    return True


def _generate_pair(
    program: Program, cfg: TaskConfig, factory: ShapeFactory,
    palette: tuple[int, ...], rng: np.random.Generator, n_shapes: int,
    raster_sort: bool = False,
    shorter_qs: tuple[tuple[Step, ...], ...] = (),
    vocab: frozenset[str] | None = None,
    level_selectors: frozenset[str] | None = None,
) -> tuple[TaskPair, list[Shape]] | None:
    """Try up to `max_pair_retries` to build one pair. `BABYTHEORIST_DEBUG=1` re-raises.

    Returns `(pair, positioned_shapes)` — the positioned shape list is
    surfaced alongside the pair so eval-mode callers can feed it directly
    to `compute_valid_query_outputs` without reconstructing it from the
    rasterised grid.

    Set `raster_sort=True` only when `by_index` is in the selector set.
    List order is part of the engine's sequential-application contract
    (multi-selectors apply in list order, collisions depend on it), so
    sorting always would silently change the generation distribution.

    Scene-level safety checks on the produced pair:
      - Engine per-step change (inside `apply_program`).
      - Phase 2 per-step effectiveness: every adjacent pair of grids
        (input + each step's output) differs. See
        ``_grid_sequence_all_distinct``.
      - Global identity backstop: ``input_grid != output_grid``.
      - Phase 2 symbolic shortcut predicate C(P):
        ``babytheorist.predicates.is_scene_admissible``. Rejects the scene when
        a shorter Q is *symbolically* guaranteed to render identically.
        Requires ``vocab`` and ``level_selectors``; defers (accepts) if
        either is missing.
      - Optional brute-force shortcut check (``shorter_qs``). Off by
        default — used as an analysis/verification tool.
    """
    for _ in range(cfg.max_pair_retries):
        try:
            shapes = [factory.sample(rng, palette) for _ in range(n_shapes)]
            placed = place_shapes(shapes, world_size=cfg.world_size, gap=cfg.gap, rng=rng)
            if placed is None:
                continue
            input_grid, positioned = placed
            if raster_sort:
                positioned.sort(key=lambda s: s.position)
            output_grid, grid_sequence = apply_program(
                program, positioned, cfg.world_size,
            )
            if output_grid is None:
                continue
            # Phase 2: each step's grid must differ from the prior one.
            if not _grid_sequence_all_distinct(input_grid, grid_sequence):
                continue
            # Global backstop: cycles that restore the initial grid are
            # not caught by adjacent-distinct (e.g. input → A → input).
            if np.array_equal(input_grid, output_grid):
                continue
            # Symbolic scene-shortcut predicate (C(P)).
            if (
                cfg.symbolic_constraints
                and vocab is not None
                and level_selectors is not None
            ):
                verdict, analyzed = scene_verdict(
                    program, positioned, vocab, level_selectors,
                )
                if analyzed:
                    if not verdict:
                        continue  # C(P) verifiably rejects
                else:
                    # C(P) deferred — coverage gap. Fall back to
                    # brute-force if configured (and Q list available).
                    if cfg.symbolic_constraints_fallback and shorter_qs:
                        if _scene_has_shorter_shortcut(
                            shorter_qs, positioned,
                            cfg.world_size, output_grid,
                        ):
                            continue
            # Unconditional brute-force (analysis mode, default off).
            if (
                cfg.scene_shortcut_check
                and shorter_qs
                and _scene_has_shorter_shortcut(
                    shorter_qs, positioned, cfg.world_size, output_grid,
                )
            ):
                continue
            return TaskPair(
                input_grid=input_grid,
                output_grid=output_grid,
                n_shapes=len(positioned),
            ), positioned
        except Exception:
            if os.environ.get("BABYTHEORIST_DEBUG"):
                raise
            continue
    return None


def _task_from_pairs_with_shapes(
    program: Program,
    pairs_with_shapes: list[tuple[TaskPair, list[Shape]]],
    level_cfg: LevelConfig,
    task_cfg: TaskConfig,
    uses_by_index: bool,
) -> Task | None:
    """Attach eval answers and adaptively choose support K when requested."""
    if task_cfg.mode != "eval":
        return Task(
            pairs=[tp for tp, _ in pairs_with_shapes],
            program=program,
            level=level_cfg.level,
        )

    if len(pairs_with_shapes) < 2:
        return None

    pool_selectors = enumerate_selector_pool(
        level_cfg.selectors, level_cfg.palette,
        n_shapes=len(pairs_with_shapes[0][1]) if uses_by_index else None,
    )

    if not task_cfg.eval_unique_support:
        demo_pair, demo_shapes = pairs_with_shapes[0]
        query_pair, query_shapes = pairs_with_shapes[1]
        Y_q = compute_valid_query_outputs(
            program, demo_shapes, query_shapes,
            task_cfg.world_size, pool_selectors,
        )
        if Y_q is None:
            return None
        return Task(
            pairs=[demo_pair, query_pair],
            program=program,
            level=level_cfg.level,
            valid_query_outputs=Y_q,
        )

    query_pair, query_shapes = pairs_with_shapes[-1]
    support_candidates = pairs_with_shapes[:-1]
    max_k = min(task_cfg.max_support, len(support_candidates))
    k_lo = max(1, int(task_cfg.min_support))
    for k in range(k_lo, max_k + 1):
        support_shapes = [shapes for _, shapes in support_candidates[:k]]
        Y_q = compute_valid_query_outputs_from_supports(
            program, support_shapes, query_shapes,
            task_cfg.world_size, pool_selectors,
            # The second distinct output is enough to reject this K and
            # try a larger support set.
            max_outputs=2,
        )
        if Y_q is not None and len(Y_q) == 1:
            return Task(
                pairs=[tp for tp, _ in support_candidates[:k]] + [query_pair],
                program=program,
                level=level_cfg.level,
                valid_query_outputs=Y_q,
                n_support=k,
            )
    return None


def generate_task(
    level_cfg: LevelConfig, task_cfg: TaskConfig,
    factory: ShapeFactory,
    pool: dict[int, dict[tuple[str, ...], float]] | None, bigram: np.ndarray,
    rng: np.random.Generator,
) -> Task | None:
    """One program, `n_pairs` (input, output) pairs. None on retry exhaustion.

    In `eval` mode the task also carries `valid_query_outputs`. By default,
    support scenes are added until selector ambiguity yields a singleton
    query answer; legacy ambiguous K=1 eval is available via TaskConfig.
    """
    lo_n, hi_n = task_cfg.n_shapes_range
    # When `by_index` is in play, demo and query must share the same object
    # count — otherwise `by_index(k)` sampled against demo could point past
    # the query's shape list. Fix n_shapes once per task in that case;
    # otherwise keep the legacy per-pair sampling below.
    # The `else lo_n` branch is a no-op placeholder: `_sample_selector` only
    # reads `n_shapes` from the by_index path, and the pair loop re-samples
    # `n` per pair, so the value is unused in non-by_index runs.
    uses_by_index = "by_index" in level_cfg.selectors
    task_n_shapes = int(rng.integers(lo_n, hi_n + 1)) if uses_by_index else lo_n  # else lo_n is placeholder

    program: Program | None = None
    program_depth = _sample_depth(level_cfg, rng)
    for _ in range(task_cfg.max_program_retries):
        try:
            program = random_program_filtered(
                program_depth, level_cfg.primitives, level_cfg.selectors, level_cfg.palette,
                bigram, rng, n_shapes=task_n_shapes,
                primitive_weights=level_cfg.primitive_weights,
            )
            break
        except SamplerDeadEnd:
            # Step-level dead-end. Retry the whole program with fresh RNG.
            continue
    if program is None:
        return None

    pairs_with_shapes: list[tuple[TaskPair, list[Shape]]] = []
    # Level primitives / selectors define the universe from which
    # shorter Q's could be drawn — that's what C(P) soundness must
    # cover. The program's own used subset is a red herring.
    level_vocab = frozenset(level_cfg.primitives)
    level_selectors_set = frozenset(level_cfg.selectors)
    for pair_idx in range(task_cfg.n_pairs):
        n = task_n_shapes if uses_by_index else int(rng.integers(lo_n, hi_n + 1))
        shorter_qs = _resolve_shorter_qs_for_pair(
            level_cfg, task_cfg, program_depth, n,
        )
        result = _generate_pair(program, task_cfg, factory, level_cfg.palette,
                                rng, n, raster_sort=uses_by_index,
                                shorter_qs=shorter_qs,
                                vocab=level_vocab,
                                level_selectors=level_selectors_set)
        if result is None:
            return None
        pairs_with_shapes.append(result)

    return _task_from_pairs_with_shapes(
        program, pairs_with_shapes, level_cfg, task_cfg, uses_by_index,
    )


# === Parallel-generation worker hooks =====================================
# `multiprocessing.Pool` requires module-level callables. `_w` is set once
# per subprocess by `_init_worker` so that per-task IPC only has to carry
# a single seed integer instead of the entire (config, pool, bigram) blob.

_w = SimpleNamespace()  # populated in each worker process by _init_worker


def _shape_factory_from_task_config(task_cfg: TaskConfig) -> ShapeFactory:
    return ShapeFactory(
        size_range=task_cfg.shape_size_range,
        compactness=task_cfg.shape_compactness,
        bordered_hollow_min_size=task_cfg.shape_bordered_hollow_min_size,
        bordered_hollow_random_inner_size=(
            task_cfg.shape_bordered_hollow_random_inner_size
        ),
        max_bbox_span=task_cfg.shape_max_bbox_span,
    )


def _init_worker(level_cfg, task_cfg, pool, bigram):
    global _w
    _w = SimpleNamespace(
        level_cfg=level_cfg,
        task_cfg=task_cfg,
        factory=_shape_factory_from_task_config(task_cfg),
        pool=pool,
        bigram=bigram,
    )


def _worker_task(seed):
    return generate_task(
        _w.level_cfg, _w.task_cfg, _w.factory, _w.pool, _w.bigram,
        np.random.default_rng(seed),
    )


def _sequential_attempts(level_cfg, task_cfg, pool, bigram, seed, max_attempts):
    """Yield `max_attempts` `Task | None` in the calling process."""
    rng = np.random.default_rng(seed)
    factory = _shape_factory_from_task_config(task_cfg)
    for _ in range(max_attempts):
        yield generate_task(level_cfg, task_cfg, factory, pool, bigram, rng)


_PARALLEL_BATCH = 100_000  # seeds materialised per `imap` call
_FILTER_PARALLEL_BATCH_PER_WORKER = 32


def _seed_batches(seed, max_attempts, batch_size: int = _PARALLEL_BATCH):
    """Yield deterministic bounded batches of per-attempt seeds."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    seed_rng = np.random.default_rng(seed)
    remaining = max_attempts
    while remaining > 0:
        n = min(batch_size, remaining)
        yield [int(s) for s in seed_rng.integers(0, 2**63, size=n)]
        remaining -= n


# === Pair-level uniqueness ================================================
# blake2b/16 of (shape, bytes) for both grids — 128 bits, no shape aliasing.

_DATASET_EXTS = ("*.h5", "*.hdf5", "*.json")


def _pair_hash(input_grid: np.ndarray, output_grid: np.ndarray) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for g in (input_grid, output_grid):
        a = np.ascontiguousarray(g, dtype=np.int8)
        h.update(np.array(a.shape, dtype=np.int32).tobytes())
        h.update(a.tobytes())
    return h.digest()


def _accept_unique(task: Task, seen: set[bytes]) -> bool:
    """Reject the task on intra-task or `seen` collisions; mutate `seen` on accept."""
    hashes = {_pair_hash(p.input_grid, p.output_grid) for p in task.pairs}
    if len(hashes) < len(task.pairs) or hashes & seen:
        return False
    seen |= hashes
    return True


def _hashes_of_file(path: Path) -> set[bytes]:
    """All (input, output) pair hashes from a BabyTheorist dataset file."""
    if path.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            inputs, outputs = f["inputs"][:], f["outputs"][:]
        # (N, P, H, W) → (N*P, H, W); padding pairs are zero-grids and
        # cannot collide with valid (input != output) generated pairs.
        inputs = inputs.reshape(-1, *inputs.shape[2:])
        outputs = outputs.reshape(-1, *outputs.shape[2:])
        return {_pair_hash(i, o) for i, o in zip(inputs, outputs)}
    from babytheorist.review import load
    return {_pair_hash(p.input_grid, p.output_grid)
            for t in load(path) for p in t.pairs}


def load_excluded_pairs(paths, out_path=None) -> set[bytes]:
    """Hash every (input,output) pair from `paths` for dedup. A path may be a
    file or a directory — directories expand to *.h5/*.hdf5/*.json children.
    `out_path` (the run's own output) is skipped if matched, so re-running
    `--exclude DIR --out DIR/x.h5` does not deadlock against its prior result."""
    out_resolved = Path(out_path).resolve() if out_path else None
    seen: set[bytes] = set()
    for raw in paths:
        p = Path(raw)
        files = (
            sorted(f for ext in _DATASET_EXTS for f in p.glob(ext)
                   if not f.name.endswith(".info.json"))
            if p.is_dir() else [p]
        )
        for f in files:
            if out_resolved is not None and f.resolve() == out_resolved:
                continue
            seen |= _hashes_of_file(f)
    return seen


# === Filter-backend worker hooks ==========================================

_fw = SimpleNamespace()  # populated in each filter worker process


def _parallel_attempts(level_cfg, task_cfg, pool, bigram, seed, max_attempts, n_workers):
    """Same contract as `_sequential_attempts` but spread across `n_workers`
    `multiprocessing.Pool` processes. Reproducible from `seed`: per-task
    seeds are drawn deterministically from the parent rng and `imap`
    preserves submission order across workers.

    Seeds are materialised in bounded batches of `_PARALLEL_BATCH` rather
    than as a single lazy generator. Empirically this avoids a `Pool`
    teardown deadlock we hit with a lazy generator + early break (workers
    becoming zombies while `Pool.__exit__` blocks on result-handler join),
    and the per-batch list keeps peak RSS flat regardless of `max_attempts`.
    """
    from multiprocessing import Pool
    with Pool(n_workers, initializer=_init_worker,
              initargs=(level_cfg, task_cfg, pool, bigram)) as p:
        for batch in _seed_batches(seed, max_attempts):
            yield from p.imap(_worker_task, batch, chunksize=64)


def generate_dataset(
    level_cfg: LevelConfig, task_cfg: TaskConfig | None = None,
    n_tasks: int = 100, seed: int | None = None,
    bigram_weights: dict[tuple[str, str], float] | None = None,
    n_workers: int = 1,
    bigram: np.ndarray | None = None,
    pool: dict[int, dict[tuple[str, ...], float]] | None = None,
    seen_pair_hashes: set[bytes] | None = None,
) -> list[Task]:
    """Generate `n_tasks` tasks. `n_workers > 1` parallelises across processes.

    `pool` is retained for the legacy public signature. The current
    rejection sampler constructs valid prefixes directly from `bigram`.
    """
    if n_tasks < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if task_cfg is None:
        task_cfg = TaskConfig()
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")
    _validate_level_cfg(level_cfg)

    if bigram is None:
        bigram = build_bigram(level_cfg.primitives, bigram_weights=bigram_weights)

    if seen_pair_hashes is None:
        seen_pair_hashes = set()
    tasks: list[Task] = []
    max_attempts = n_tasks * task_cfg.max_dataset_attempts_per_task
    if n_workers <= 1:
        attempts = _sequential_attempts(level_cfg, task_cfg, pool, bigram,
                                        seed, max_attempts)
    else:
        attempts = _parallel_attempts(level_cfg, task_cfg, pool, bigram,
                                      seed, max_attempts, n_workers)
    with tqdm(total=n_tasks, desc=f"L{level_cfg.level}", unit="task") as bar:
        for t in attempts:
            if t is None:
                continue
            if not _accept_unique(t, seen_pair_hashes):
                continue
            tasks.append(t)
            bar.update(1)
            if len(tasks) >= n_tasks:
                break
    close_attempts = getattr(attempts, "close", None)
    if close_attempts is not None:
        close_attempts()

    if len(tasks) < n_tasks:
        raise RuntimeError(
            f"generate_dataset: only produced {len(tasks)}/{n_tasks} tasks "
            f"after {max_attempts} attempts (level={level_cfg.level})"
        )
    return tasks


__all__ = [
    "_sample_selector",
    "_SYMBOLIC_SELECTOR_TYPES",
    "_has_dynamic_selector",
    "_passes_phase1_filter",
    "random_program_filtered",
    "_enum_q_steps",
    "_cached_shorter_programs",
    "_q_unsat_selector",
    "_scene_primary_colors",
    "_scene_has_hollow",
    "_scene_has_shorter_shortcut",
    "_resolve_shorter_qs_for_pair",
    "_grid_sequence_all_distinct",
    "_generate_pair",
    "_task_from_pairs_with_shapes",
    "generate_task",
    "_w",
    "_shape_factory_from_task_config",
    "_init_worker",
    "_worker_task",
    "_sequential_attempts",
    "_PARALLEL_BATCH",
    "_FILTER_PARALLEL_BATCH_PER_WORKER",
    "_seed_batches",
    "_DATASET_EXTS",
    "_pair_hash",
    "_accept_unique",
    "_hashes_of_file",
    "load_excluded_pairs",
    "_fw",
    "_parallel_attempts",
    "generate_dataset",
]
