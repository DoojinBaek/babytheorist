"""Evaluation transfer: compute the set of valid query outputs Y_q.

When a model sees one demo pair (x, y) and is asked to predict on a query
input x_q, the program is not always uniquely identified — different
selectors may pick the same shapes on x but disagree on x_q. Y_q is the
set of outputs consistent with that ambiguity:

    Y_q = { π'(x_q) : π' agrees with π on x }

Public surface (built up in this module):
    enumerate_selector_pool(types, palette) -> list[Selector]
    alternatives_at_step(step, scene, pool) -> list[Step]
    compute_valid_query_outputs(program, demo, query, world_size, pool, max_outputs=None) -> list[Grid] | None
    compute_valid_query_outputs_from_supports(...) -> list[Grid] | None
"""

from __future__ import annotations

import numpy as np

from .core import Program, Selector, SINGLE_SELECTORS, Shape, Step
from .engine import _rebuild_grid, apply_program_with_scenes, apply_step
from .selectors import select


def enumerate_selector_pool(
    selector_types: tuple[str, ...],
    palette: tuple[int, ...],
    n_shapes: int | None = None,
) -> list[Selector]:
    """Expand `LevelConfig.selectors` into concrete `Selector` objects.

    `by_color` expands to one Selector per palette color; `by_index` expands
    to `by_index(1)..by_index(n_shapes)` and therefore requires `n_shapes`.
    All other types take no param and appear as-is. Raises if `by_index`
    appears without `n_shapes` rather than silently omitting it.
    """
    if "by_index" in selector_types and n_shapes is None:
        raise ValueError("enumerate_selector_pool: by_index requires n_shapes")
    pool: list[Selector] = []
    for t in selector_types:
        if t == "by_color":
            for c in palette:
                pool.append(Selector("by_color", param=int(c)))
        elif t == "by_index":
            for k in range(1, n_shapes + 1):
                pool.append(Selector("by_index", param=k))
        else:
            pool.append(Selector(t))
    return pool


def alternatives_at_step(
    step: Step,
    scene: list[Shape],
    pool: list[Selector],
) -> list[Step]:
    """Return all Step variants whose selector(s) pick the same indices on `scene`.

    For transform / movement primitives (single selector), every pool entry
    that resolves to the same indices is returned. For relational primitives
    (single src + single tgt), the result is the Cartesian product of
    same-src and same-tgt single-selectors.

    The canonical `step` is always included — explicitly inserted if the
    pool happens not to contain its selector(s) (defensive: keeps the
    function robust to caller-supplied pools).

    Known limitation (relational): the single-selector filter is
    type-based, so multi-selectors (`by_color`, `is_hollow`) that happen
    to match exactly one shape on the scene are excluded. This mirrors
    `Step.__post_init__`'s current restriction and under-estimates Y_q
    in those cases. Planned follow-up: relax the Step constructor to
    runtime-arity checking and drop the `SINGLE_SELECTORS` filter here.
    """
    src_indices = sorted(select(step.selector, scene))
    if step.target is None:
        alt_srcs = [s for s in pool if sorted(select(s, scene)) == src_indices]
        result = [Step(s, step.primitive) for s in alt_srcs]
    else:
        tgt_indices = sorted(select(step.target, scene))
        single_pool = [s for s in pool if s.type in SINGLE_SELECTORS]
        alt_srcs = [s for s in single_pool if sorted(select(s, scene)) == src_indices]
        alt_tgts = [s for s in single_pool if sorted(select(s, scene)) == tgt_indices]
        result = [
            Step(src, step.primitive, target=tgt)
            for src in alt_srcs
            for tgt in alt_tgts
        ]
    if step not in result:
        result.insert(0, step)
    return result


def alternatives_at_step_across_supports(
    step: Step,
    scenes: list[list[Shape]],
    pool: list[Selector],
) -> list[Step]:
    """Return Step variants equivalent to ``step`` on every support scene.

    For transform / movement primitives, every pool entry that resolves to
    the same indices on all support scenes is returned. For relational
    primitives, the result is the Cartesian product of same-src and same-tgt
    single-selectors across all support scenes.

    Known limitation (relational): the single-selector filter is type-based,
    so multi-selectors that happen to match exactly one shape on every support
    scene are excluded. This mirrors ``Step.__post_init__``'s current
    restriction and under-estimates Y_q in those cases.
    """
    if not scenes:
        raise ValueError("alternatives_at_step_across_supports requires support scenes")

    if step.target is None:
        canonical = [sorted(select(step.selector, scene)) for scene in scenes]
        alt_srcs = [
            s for s in pool
            if all(sorted(select(s, scene)) == canonical[i]
                   for i, scene in enumerate(scenes))
        ]
        result = [Step(s, step.primitive) for s in alt_srcs]
    else:
        src_canonical = [sorted(select(step.selector, scene)) for scene in scenes]
        tgt_canonical = [sorted(select(step.target, scene)) for scene in scenes]
        single_pool = [s for s in pool if s.type in SINGLE_SELECTORS]
        alt_srcs = [
            s for s in single_pool
            if all(sorted(select(s, scene)) == src_canonical[i]
                   for i, scene in enumerate(scenes))
        ]
        alt_tgts = [
            s for s in single_pool
            if all(sorted(select(s, scene)) == tgt_canonical[i]
                   for i, scene in enumerate(scenes))
        ]
        result = [
            Step(src, step.primitive, target=tgt)
            for src in alt_srcs
            for tgt in alt_tgts
        ]
    if step not in result:
        result.insert(0, step)
    return result


def compute_valid_query_outputs_from_supports(
    program: Program,
    support_shapes: list[list[Shape]],
    query_shapes: list[Shape],
    world_size: int,
    pool: list[Selector],
    max_outputs: int | None = None,
) -> list[np.ndarray] | None:
    """Compute Y_q from a support set.

    A selector substitution remains valid only if it picks the same object
    indices as the canonical program on every support scene at every step.
    Returns ``None`` under strict compatibility semantics: if any support
    example fails the canonical program, or if any support-consistent
    alternative branch fails on the query.

    When ``max_outputs`` is set, DFS stops once that many distinct query
    outputs have been found. That form is intentionally partial and should
    only be used by callers that treat "at least max_outputs distinct
    outputs" as enough information, such as singleton-Y_q rejection.
    """
    if not support_shapes:
        raise ValueError("compute_valid_query_outputs_from_supports requires support")
    if max_outputs is not None and max_outputs < 1:
        raise ValueError(f"max_outputs must be >= 1 when set, got {max_outputs}")

    support_scenes_per_step: list[list[list[Shape]]] = []
    for shapes in support_shapes:
        _, scenes = apply_program_with_scenes(program, shapes, world_size)
        if len(scenes) < len(program):
            return None
        support_scenes_per_step.append(scenes)

    alts_by_step = [
        alternatives_at_step_across_supports(
            program[step_idx],
            [scenes[step_idx] for scenes in support_scenes_per_step],
            pool,
        )
        for step_idx in range(len(program))
    ]

    results: dict[bytes, np.ndarray] = {}
    all_ok = [True]
    enough_outputs = [False]

    def dfs(step_idx: int, current: list[Shape]) -> None:
        if not all_ok[0] or enough_outputs[0]:
            return
        if step_idx == len(program):
            grid = _rebuild_grid(current, world_size)
            if grid is None:
                all_ok[0] = False
                return
            results[grid.tobytes()] = grid
            if max_outputs is not None and len(results) >= max_outputs:
                enough_outputs[0] = True
            return
        for alt in alts_by_step[step_idx]:
            if not all_ok[0] or enough_outputs[0]:
                return
            next_state = apply_step(alt, current, world_size)
            if next_state is None:
                all_ok[0] = False
                return
            dfs(step_idx + 1, next_state)

    dfs(0, [s.copy() for s in query_shapes])
    if not all_ok[0]:
        return None
    return list(results.values())


def compute_valid_query_outputs(
    program: Program,
    demo_shapes: list[Shape],
    query_shapes: list[Shape],
    world_size: int,
    pool: list[Selector],
    max_outputs: int | None = None,
) -> list[np.ndarray] | None:
    """Compute Y_q: every output reachable on `query_shapes` via alternative
    selector substitutions that agree with `program` on `demo_shapes`.

    Strict compatibility: the query scene must accommodate *every*
    alternative program consistent with the demo. If any branch fails on
    the query (collision, precondition, OOB, or unrasterisable final
    scene), returns `None` — the caller should drop this query pair and
    retry. This matches the eval semantics in which any hypothesis the
    model could legitimately form from the demo must yield some grid on
    the query.

    Returns:
        list[np.ndarray]: deduplicated grids (insertion-ordered) if all
            alternative branches succeed. Always non-empty in this case
            because the canonical branch is guaranteed to reach a leaf.
        None: at least one branch died → query is incompatible with the
            full alternative set.

    ``max_outputs`` is an early-stop for callers that only need to know
    whether the answer set is singleton. It may return a partial output set.
    """
    return compute_valid_query_outputs_from_supports(
        program, [demo_shapes], query_shapes, world_size, pool,
        max_outputs=max_outputs,
    )
