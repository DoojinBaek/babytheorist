"""End-to-end reliability test for the full 14-primitive vocab.

Phase 10' — adds relational primitives (swap_colors, align_row,
align_col) to the Phase 9' NON_RELATIONAL_VOCAB. n_shapes=1 cases
are dropped because relational needs ≥2 shapes; depth=3 with n=2
stays the most stress-testing point for relational interactions.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.sampler import FULL_VOCAB
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)
_VOCAB = FULL_VOCAB
_SELECTORS = frozenset({"ALL", "by_index"})


def _factory() -> ShapeFactory:
    return ShapeFactory(generators=(solid,), size_range=(10, 20))


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (1, 2, 60),
    (2, 2, 60),
    (3, 2, 60),
])
def test_no_shortcuts_full_vocab(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 10_000 + n_shapes + 13)
    factory = _factory()

    violations = []
    generated = 0
    for _ in range(n_tasks * 8):
        if generated >= n_tasks:
            break
        task = generate_task(
            depth=depth, n_shapes=n_shapes, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
            vocab=_VOCAB, selector_types=_SELECTORS,
        )
        if task is None:
            continue
        generated += 1

        offender = scene_has_shortcut(
            task.output_grid, task.input_shapes, world_size=24,
            depth=depth, vocab=_VOCAB, selector_types=_SELECTORS,
            n_shapes=n_shapes,
        )
        if offender is not None:
            violations.append({
                "program": task.program,
                "shortcut": offender,
                "depth": depth,
                "n_shapes": n_shapes,
            })

    assert generated >= n_tasks * 0.4, (
        f"generated only {generated}/{n_tasks} — sampler too leaky"
    )
    assert not violations, (
        f"{len(violations)}/{generated} shortcut violations "
        f"at depth={depth} n_shapes={n_shapes}: first={violations[0]}"
    )
