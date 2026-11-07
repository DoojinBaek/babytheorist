"""End-to-end reliability test for the non-relational vocab.

Phase 9' — adds hollow / fill_interior / add_border to the Phase 8'
vocab. Covers every non-relational primitive.

Initial grid_shape is assumed solid (matches the ``solid``
ShapeFactory generator used here and ``FilterState.initial``'s
default). fill_interior requires a prior hollow to be applicable.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.sampler import NON_RELATIONAL_VOCAB
from babytheorist.shapes import ShapeFactory, solid


_PALETTE = (1, 2, 3, 4, 5)
_VOCAB = NON_RELATIONAL_VOCAB
_SELECTORS = frozenset({"ALL", "by_index"})


def _factory() -> ShapeFactory:
    # Larger size_range needed to reliably yield shapes with min_dim >= 3
    # and non-square bbox (both required once add_border + translate land).
    return ShapeFactory(generators=(solid,), size_range=(10, 20))


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (1, 1, 60),
    (2, 1, 60),
    (3, 1, 60),
    (1, 2, 60),
    (2, 2, 60),
    (3, 2, 60),
])
def test_no_shortcuts_non_relational(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 10_000 + n_shapes + 7)
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
