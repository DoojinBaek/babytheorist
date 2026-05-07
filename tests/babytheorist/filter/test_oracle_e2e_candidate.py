"""Small production-filter end-to-end shortcut-oracle checks."""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.generate import generate_task
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)
_SELECTORS = frozenset({"ALL", "by_index"})


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (1, 1, 8),
    (2, 1, 8),
    (3, 1, 6),
    (2, 2, 8),
])
def test_filter_generated_d4_tasks_have_no_shorter_scene_shortcut(
    depth, n_shapes, n_tasks,
):
    rng = np.random.default_rng(depth * 10_000 + n_shapes)
    factory = ShapeFactory(generators=(solid,), size_range=(3, 6))

    generated = 0
    violations = []
    for _ in range(n_tasks * 8):
        if generated >= n_tasks:
            break
        task = generate_task(
            depth=depth,
            n_shapes=n_shapes,
            factory=factory,
            palette=_PALETTE,
            world_size=24,
            rng=rng,
            vocab=D4_PRIMITIVE_VOCAB,
            selector_types=_SELECTORS,
        )
        if task is None:
            continue
        generated += 1

        offender = scene_has_shortcut(
            task.output_grid,
            task.input_shapes,
            world_size=24,
            depth=depth,
            vocab=D4_PRIMITIVE_VOCAB,
            selector_types=_SELECTORS,
            n_shapes=n_shapes,
        )
        if offender is not None:
            violations.append((task.program, offender))

    assert generated >= max(1, n_tasks // 2)
    assert not violations
