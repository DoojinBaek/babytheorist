"""End-to-end reliability test — D4-only vocab, Phase 6'.

Generates tasks through the full pipeline (sampler → scene_sampler →
engine) and brute-forces every shorter Q to verify 0 shortcut
violations. Covers depth ∈ {1, 2, 3}, n ∈ {1, 2}.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)
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
def test_no_shortcuts_d4(depth, n_shapes, n_tasks):
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
            vocab=D4_PRIMITIVE_VOCAB, selector_types=_SELECTORS,
        )
        if task is None:
            continue
        generated += 1

        offender = scene_has_shortcut(
            task.output_grid, task.input_shapes, world_size=24,
            depth=depth, vocab=D4_PRIMITIVE_VOCAB,
            selector_types=_SELECTORS, n_shapes=n_shapes,
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
