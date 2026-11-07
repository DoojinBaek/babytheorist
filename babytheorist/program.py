"""Program generation and validation.

Two concerns live here:

    1. Algebraic simplification (`simplify`, `effective_depth`, `is_valid`).
    2. Dominant-program sampling (`build_bigram`, `build_pool`,
       `sample_primitives`) — a primitive-level bigram feeds a compositional
       top-k pool, and programs are drawn either directly from `pool[depth]`
       or by concatenating chunks from `pool[1..K-1]`.

Rewrite rules (applied to *adjacent* steps sharing selector and target):

    1. Canceling pair: `(a, b)` where `a.cancels_with = b` → remove both
    2. Involution:     `(a, a)` where `a` is involution    → remove both
    3. Idempotent:     `(a, a)` where `a` is idempotent    → keep one
    4. Cyclic run:     `a` repeated `n` times → `a ** (n % order)`

A program is *valid* if its length equals its effective depth.
"""

from __future__ import annotations

import numpy as np

from .core import Program, Step
from .primitives import (
    canceling_pairs,
    cyclic_orders,
    idempotents,
    involutions,
)
from .symbolic import (
    PreconditionFail,
    apply_symbolic_program,
    step_reads,
    step_writes,
    write_targets,
)


__all__ = [
    "simplify", "effective_depth", "is_valid",
    "trajectory_distinct", "no_dead_writes",
    "build_bigram", "build_pool", "sample_primitives", "sample_bigram",
    "SamplerDeadEnd",
]


class SamplerDeadEnd(Exception):
    """Retryable dead-end in `sample_primitives` (distinct from real bugs)."""


def _require_unique(primitives: list[str]) -> None:
    """Reject duplicates — they desync `idx` and the bigram matrix."""
    if len(set(primitives)) != len(primitives):
        raise ValueError(f"duplicate primitives in vocabulary: {primitives!r}")


def _same_context(a: Step, b: Step) -> bool:
    """Two steps are in the same context if they target the same objects."""
    return a.selector == b.selector and a.target == b.target


def _reduce_cyclic_runs(program: Program) -> Program:
    """Collapse consecutive runs of a cyclic primitive modulo its order.

    `[rot90] * 4` (same context) → `[]`; `[rot90] * 5` → `[rot90]`.
    """
    if not program:
        return program

    orders = cyclic_orders()
    result: Program = []
    i = 0

    while i < len(program):
        step = program[i]
        if step.primitive in orders:
            order = orders[step.primitive]
            run_length = 1
            while (
                i + run_length < len(program)
                and program[i + run_length].primitive == step.primitive
                and _same_context(program[i + run_length], step)
            ):
                run_length += 1
            result.extend([step] * (run_length % order))
            i += run_length
        else:
            result.append(step)
            i += 1

    return result


def simplify(program: Program) -> Program:
    """Apply algebraic rewrites until the program stops shrinking."""
    cancel_set = set(canceling_pairs())
    involution_set = set(involutions())
    idempotent_set = set(idempotents())

    result = _reduce_cyclic_runs(list(program))

    changed = True
    while changed:
        changed = False
        new_result: Program = []
        i = 0

        while i < len(result):
            if i + 1 < len(result):
                a, b = result[i], result[i + 1]

                if (a.primitive, b.primitive) in cancel_set and _same_context(a, b):
                    i += 2
                    changed = True
                    continue

                if a.primitive == b.primitive and _same_context(a, b):
                    if a.primitive in involution_set:
                        i += 2
                        changed = True
                        continue
                    if a.primitive in idempotent_set:
                        new_result.append(a)
                        i += 2
                        changed = True
                        continue

            new_result.append(result[i])
            i += 1

        result = new_result

        if changed:
            prev_len = len(result)
            result = _reduce_cyclic_runs(result)
            if len(result) != prev_len:
                changed = True

    return result


def effective_depth(program: Program) -> int:
    """Length of the program after `simplify` runs to fixpoint."""
    return len(simplify(program))


def is_valid(program: Program) -> bool:
    """True iff no algebraic simplification applies."""
    return len(program) == effective_depth(program)


# === Phase 1 Rule 1 — Symbolic trajectory distinct =========================


def trajectory_distinct(
    program: Program,
    n_shapes: int,
    *,
    grid_shapes: list[str] | None = None,
) -> bool:
    """True iff no shape's symbolic state ever returns to a previously
    reached state under ``program``.

    Stronger than ``is_valid``. Catches group-theoretic identities that
    ``simplify`` misses:

        - ``rot90 x 4`` (cyclic) — cycled orientation back to identity
        - ``flip_h x 2`` (involution)
        - D4 composites like ``rot; flip; rot; flip`` (= e)
        - 3-cycle swaps ``swap(A,B); swap(B,C); swap(A,C)`` (A cycles back)
        - ``hollow; fill_interior`` round-trip
        - Cross-shape commutation with hidden self-loops
          (e.g. ``rot(A); trans(B); rot(A); rot(A); rot(A)`` — A's own
          trajectory cycles back, even though ``simplify`` can't cancel
          non-adjacent rot's interrupted by a step on a different shape)

    Does NOT catch overwrite dead-code (``align(A,B); align(A,C)`` —
    trajectory is syntactically distinct but first align is dead); that's
    Phase 1 Rule 2.

    Per-shape check protocol: a step that does not write to a given
    shape leaves that shape's state unchanged; including such
    "untouched" steps would spuriously flag programs like
    ``rot(A); trans(B)`` (A unchanged on step 2) as redundant. So for
    each shape we record its state only at steps that **write** to it
    (plus the initial state), then require pairwise distinctness on
    that filtered trajectory. A duplicate means the shape was modified
    and either returned to a prior configuration or was dead-written
    (e.g. ``hollow; hollow`` — second write leaves state unchanged,
    which the filter surfaces).

    Returns False if the program hits a symbolic precondition failure
    (e.g. ``fill_interior`` on a solid shape, ``align_row`` between
    already-aligned shapes) — these programs are symbolically infeasible
    and therefore invalid.
    """
    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False

    for i in range(n_shapes):
        filtered = [traj[0][i]]
        for k, step in enumerate(program):
            if i in write_targets(step, traj[k]):
                filtered.append(traj[k + 1][i])
        if len(set(filtered)) != len(filtered):
            return False
    return True


# === Phase 1 Rule 2 — Dataflow dead-code ===================================


def no_dead_writes(
    program: Program,
    n_shapes: int,
    *,
    grid_shapes: list[str] | None = None,
) -> bool:
    """True iff no step in ``program`` has all of its writes dead.

    **Definition**: a step-``k`` write to ``(shape_i, property P)`` is
    *dead* if looking at later steps, either

    - the property is **overwritten** (written again) before being
      read, OR
    - nothing reads it *and* nothing overwrites it — wait, in that case
      the value persists to the final state and is observable, so
      actually NOT dead.

    So precisely: a write is dead iff a later step writes the same
    (shape, property) *with no intervening computational read*.

    A **step** is dead iff all of its writes are dead. Removing a dead
    step yields a shorter program with the same final state → redundant.

    Catches (examples):
      - ``align_row(A, B); align_row(A, C)`` — first align's row write
        is overwritten by second with no intervening read.
      - ``translate_up(A); align_row(A, X)`` — translate's row write is
        discarded by align.
      - ``align_row(A, B); translate_up(A); align_row(A, C)`` — translate
        dead (first align is live via translate's computational row
        read).
      - ``hollow(A); hollow(A)`` — first hollow overwritten.
      - ``hollow(A); fill_interior(A); hollow(A)`` — fill's write is
        discarded by second hollow.

    Does NOT catch patterns where the write is consumed by a read on
    the path (e.g. ``rot; rot; rot; rot`` — each rot reads and writes
    orientation, all live). Those are ``trajectory_distinct`` /
    ``is_minimal_level1`` territory.

    **Important**: read/write sets use *computational* reads only —
    precondition checks like ``align_row``'s ``src.row != tgt.row``
    guard are excluded, because they don't affect output. See
    ``babytheorist.symbolic.PRIMITIVE_IO`` for the table.

    Returns False on ``PreconditionFail``: symbolically infeasible
    programs are also invalid.
    """
    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False

    reads_per = [step_reads(step, traj[k]) for k, step in enumerate(program)]
    writes_per = [step_writes(step, traj[k]) for k, step in enumerate(program)]

    for k in range(len(program)):
        writes_k = writes_per[k]
        if not writes_k:
            continue
        all_dead = True
        for target in writes_k:
            is_dead = False
            for m in range(k + 1, len(program)):
                if target in reads_per[m]:
                    break  # live — consumed by a later step
                if target in writes_per[m]:
                    is_dead = True
                    break  # dead — overwritten without intervening read
            if not is_dead:
                all_dead = False
                break
        if all_dead:
            return False
    return True


# === Dominant-program sampling ============================================


def build_bigram(
    primitives: list[str],
    bigram_weights: dict[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Row-stochastic `T[i, j] = P(next=j | i)`: uniform, zero canceling
    pairs and involution self-loops, optionally apply multiplicative spikes
    from `bigram_weights`, row-normalize. Spikes can't revive zeroed cells
    (mask runs first). Raises `ValueError` on duplicates or when masking
    kills every transition (e.g. `['flip_h']`).
    """
    _require_unique(primitives)
    idx = {p: i for i, p in enumerate(primitives)}
    mat = np.ones((len(primitives), len(primitives)), dtype=np.float64)

    for a, b in canceling_pairs(primitives):
        mat[idx[a], idx[b]] = 0.0
    for p in involutions():
        if p in idx:
            mat[idx[p], idx[p]] = 0.0

    if bigram_weights:
        for (a, b), w in bigram_weights.items():
            if a in idx and b in idx:
                mat[idx[a], idx[b]] *= w

    if not np.any(mat > 0):
        raise ValueError(f"build_bigram: no positive transitions for {primitives!r}")

    row_sums = mat.sum(axis=1, keepdims=True)
    return mat / np.where(row_sums > 0, row_sums, 1.0)


def build_pool(
    bigram: np.ndarray,
    primitives: list[str],
    max_depth: int,
    pool_size: int = 20,
    primitive_weights: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> dict[int, dict[tuple[str, ...], float]]:
    """Compositional top-k pool. `pool[1] = {(p,): w(p)}` with `w(p)` from
    `primitive_weights` (default 1.0), and `pool[d]` is the top `pool_size`
    of `{left∘right : left∈pool[s], right∈pool[d-s], 1≤s<d}` scored by
    `score(left) · T[last(left), first(right)] · score(right)` — which
    telescopes to `∏ w(p_i) · ∏ T[p_i, p_{i+1}]` along the sequence, so
    `primitive_weights` propagates to every level via the DP.

    If `rng` is given, the top-k cut becomes a **score-weighted sample
    without replacement** instead of strict argmax: higher-score candidates
    are still more likely to survive, but lower-score ones are not always
    excluded — which keeps the pool diverse when many candidates tie at the
    cutoff. Without `rng`, the original deterministic argmax top-k is used.
    """
    _require_unique(primitives)
    idx = {p: i for i, p in enumerate(primitives)}
    weights = primitive_weights or {}
    pool = {1: {(p,): weights.get(p, 1.0) for p in primitives}}
    for d in range(2, max_depth + 1):
        cands: dict[tuple[str, ...], float] = {}
        for s in range(1, d):
            for left, ls in pool[s].items():
                for right, rs in pool[d - s].items():
                    prog = left + right
                    score = ls * bigram[idx[left[-1]], idx[right[0]]] * rs
                    if score > cands.get(prog, 0.0):
                        cands[prog] = score
        items = list(cands.items())
        n_pick = min(pool_size, len(items))
        if rng is not None:
            w = np.array([s for _, s in items], dtype=np.float64)
            indices = rng.choice(len(items), size=n_pick, replace=False, p=w / w.sum())
            items = [items[i] for i in indices]
        pool[d] = dict(sorted(items, key=lambda kv: -kv[1])[:n_pick])
    return pool


def sample_primitives(
    pool: dict[int, dict[tuple[str, ...], float]],
    bigram: np.ndarray,
    primitives: list[str],
    depth: int,
    rng: np.random.Generator,
    p_direct: float = 0.5,
) -> list[str]:
    """Sample a primitive-name sequence of length `depth`.

    With probability `p_direct` (and only when `depth <= K`, the deepest
    non-empty pool level), picks directly from `pool[depth]`; otherwise
    concatenates boundary-weighted chunks from `pool[1..K-1]`. Dead-ends
    (empty pool level or zero-weight continuation) raise `SamplerDeadEnd`.
    """
    _require_unique(primitives)
    idx = {p: i for i, p in enumerate(primitives)}
    nonempty = [d for d, entries in pool.items() if entries]
    if not nonempty:
        raise SamplerDeadEnd("pool has no non-empty levels")
    K = max(nonempty)

    def pick(d: int, prev: str | None) -> tuple[str, ...]:
        items = list(pool[d].items())
        if not items:
            raise SamplerDeadEnd(f"pool[{d}] is empty")
        w = np.array([s for _, s in items], dtype=np.float64)
        if prev is not None:
            w *= bigram[idx[prev], [idx[c[0]] for c, _ in items]]
        if w.sum() == 0:
            raise SamplerDeadEnd(f"zero-weight continuation at prev={prev!r}, pool[{d}]")
        return items[rng.choice(len(items), p=w / w.sum())][0]

    if depth == 1 or (depth <= K and rng.random() < p_direct):
        return list(pick(depth, None))

    out, prev, rem = [], None, depth
    while rem > 0:
        d = int(rng.integers(1, min(rem, max(1, K - 1)) + 1))
        out.extend(pick(d, prev))
        prev, rem = out[-1], rem - d
    return out


def sample_bigram(
    bigram: np.ndarray,
    primitives: list[str],
    depth: int,
    rng: np.random.Generator,
    primitive_weights: dict[str, float] | None = None,
) -> list[str]:
    """Pool-free sampler. First primitive drawn from `primitive_weights`
    (uniform when `None`); each subsequent draw uses `bigram[prev, :]`
    column-scaled by the same weight vector — mirroring how `build_pool`
    propagates `primitive_weights` through every level of the dominant
    pool. Raises `SamplerDeadEnd` if a step has no positive mass.
    """
    _require_unique(primitives)
    n = len(primitives)
    w = primitive_weights or {}
    weight_vec = np.array([w.get(p, 1.0) for p in primitives], dtype=np.float64)
    probs = weight_vec
    out: list[str] = []
    for _ in range(depth):
        total = probs.sum()
        if total <= 0:
            raise SamplerDeadEnd(f"bigram row all-zero at prev={out[-1] if out else None!r}")
        j = int(rng.choice(n, p=probs / total))
        out.append(primitives[j])
        probs = bigram[j] * weight_vec
    return out
