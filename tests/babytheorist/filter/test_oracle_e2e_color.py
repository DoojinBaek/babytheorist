"""End-to-end reliability test for the D4 + translate + next_color vocab.

Phase 8' — next_color adds an abelian Z/9Z axis. Trace-identity +
R1 (cyclic_order 9 → max 8) + R3 already cover the color axis; this
test confirms no new shortcut family opens up when the axis is
added.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.sampler import D4_TRANSLATE_COLOR_VOCAB
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)
_VOCAB = D4_TRANSLATE_COLOR_VOCAB
_SELECTORS = frozenset({"ALL", "by_index"})


def _factory() -> ShapeFactory:
    return ShapeFactory(generators=(solid,), size_range=(3, 6))


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (1, 1, 80),
    (2, 1, 80),
    (3, 1, 80),
    (1, 2, 80),
    (2, 2, 80),
    (3, 2, 80),
])
def test_no_shortcuts_d4_translate_color(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 10_000 + n_shapes)
    factory = _factory()

    violations = []
    generated = 0
    for _ in range(n_tasks * 6):
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

    assert generated >= n_tasks * 0.5, (
        f"generated only {generated}/{n_tasks} — sampler too leaky"
    )
    assert not violations, (
        f"{len(violations)}/{generated} shortcut violations "
        f"at depth={depth} n_shapes={n_shapes}: first={violations[0]}"
    )
