"""Brute-force shortcut oracle — reliability analysis only.

Enumerates every Q of length ``< P.depth`` in the level's (vocab,
selector_types) language. For each Q, runs
:func:`babytheorist.engine.apply_program` on the task's input scene. If any
Q's output equals P's output, the task has a shortcut and we count
one violation.

Serial Python loop, exhaustive and authoritative. Feasible up to
depth ~4 on the filter vocab.

This is the ground-truth check for :func:`babytheorist.filter.generate.generate_task`.
It is NOT used in the generation path (that would defeat the point
of the symbolic filter). See
``memory/feedback_brute_force_oracle_only.md``.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from babytheorist.core import ALL, Program, RELATIONAL_PRIMITIVE_NAMES, Selector, Shape, Step
from babytheorist.engine import apply_program


__all__ = [
    "enumerate_shorter_programs",
    "scene_has_shortcut",
]


def _enumerate_step_options(
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    *,
    palette: tuple[int, ...] | None = None,
) -> list[Step]:
    """Every Step legal at this level: (primitive, selector[, target])."""
    opts: list[Step] = []
    selectors: list[Selector] = []
    if "ALL" in selector_types:
        selectors.append(ALL)
    if "by_index" in selector_types:
        for k in range(1, n_shapes + 1):
            selectors.append(Selector("by_index", k))
    if "is_hollow" in selector_types:
        selectors.append(Selector("is_hollow"))
    if "by_color" in selector_types:
        if palette is None:
            raise ValueError(
                "by_color in selector_types but palette not provided to oracle"
            )
        for c in sorted(palette):
            selectors.append(Selector("by_color", int(c)))
    for sel_type in ("topmost", "bottommost", "leftmost", "rightmost"):
        if sel_type in selector_types:
            selectors.append(Selector(sel_type))

    # Relational primitives need SINGLE_SELECTORS (defined in core). by_index,
    # topmost, bottommost, leftmost, rightmost are single-select; is_hollow /
    # by_color are multi-select and cannot fill src/tgt.
    _SINGLE_SEL_TYPES = frozenset({
        "by_index", "topmost", "bottommost", "leftmost", "rightmost",
    })
    single_selectors = [s for s in selectors if s.type in _SINGLE_SEL_TYPES]

    for prim in sorted(vocab):
        if prim in RELATIONAL_PRIMITIVE_NAMES:
            for src in single_selectors:
                for tgt in single_selectors:
                    if src == tgt:
                        continue
                    opts.append(Step(selector=src, primitive=prim, target=tgt))
        else:
            for sel in selectors:
                opts.append(Step(selector=sel, primitive=prim))
    return opts


def enumerate_shorter_programs(
    depth: int,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None = None,
) -> Iterator[Program]:
    """Yield every program of length ``0..depth-1``."""
    options = _enumerate_step_options(
        vocab, selector_types, n_shapes, palette=palette,
    )

    def extend(prefix: list[Step], remaining: int) -> Iterator[Program]:
        yield list(prefix)
        if remaining == 0:
            return
        for step in options:
            prefix.append(step)
            yield from extend(prefix, remaining - 1)
            prefix.pop()

    yield from extend([], depth - 1)


def scene_has_shortcut(
    p_output: np.ndarray,
    input_shapes: list[Shape],
    world_size: int,
    depth: int,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None = None,
) -> Optional[Program]:
    """Search for any Q of length < ``depth`` producing ``p_output``.

    Returns the offending Q (first encountered) or ``None`` if no
    shortcut exists.
    """
    for q in enumerate_shorter_programs(
        depth, vocab=vocab, selector_types=selector_types,
        n_shapes=n_shapes, palette=palette,
    ):
        q_out, _ = apply_program(q, input_shapes, world_size)
        if q_out is None:
            continue
        if np.array_equal(q_out, p_output):
            return q
    return None
