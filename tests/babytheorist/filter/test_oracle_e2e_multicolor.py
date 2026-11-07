"""T2-2 — multi-color shape attribute audit.

Runs the full default shape factory (``solid, bordered, hollow,
patterned``) through the filter with both ``by_color`` and
``is_hollow`` selectors enabled, against the D4 + translate +
next_color + shape-axis vocab. Goal: confirm that multi-color
generators (``bordered``, ``patterned``) don't open any shortcut
families the filter can't see — in particular:

- ``primary_color`` detection on bordered / patterned shapes matches
  the symbolic ``ColorVal.base``.
- ``next_color`` shifts every color (not just primary) at the scene
  level, but R7' / I1 consume only primary — any leak surfaces here.
- ``add_border`` on multi-color shapes uses adjacent source colors,
  so the shape's primary stays stable; verified indirectly via the
  oracle cross-check.
- ``by_color`` commitments correctly scope via ``primary_color``
  under multi-color shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.sampler import NON_RELATIONAL_VOCAB
from babytheorist.shapes import ShapeFactory, bordered, hollow, patterned, solid


_PALETTE = (1, 2, 3, 4, 5)
_VOCAB = NON_RELATIONAL_VOCAB
_SELECTORS = frozenset({"ALL", "by_index", "is_hollow", "by_color"})


def _default_factory() -> ShapeFactory:
    return ShapeFactory(
        generators=(solid, bordered, hollow, patterned),
        size_range=(10, 20),
    )


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (2, 2, 30),
    (3, 2, 20),
])
def test_no_shortcuts_multicolor(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 1000 + n_shapes + 41)
    factory = _default_factory()

    violations = []
    generated = 0
    for _ in range(n_tasks * 10):
        if generated >= n_tasks:
            break
        task = generate_task(
            depth=depth, n_shapes=n_shapes, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
            vocab=_VOCAB, selector_types=_SELECTORS,
            initial_hollow_prob=0.3,
        )
        if task is None:
            continue
        generated += 1

        offender = scene_has_shortcut(
            task.output_grid, task.input_shapes, world_size=24,
            depth=depth, vocab=_VOCAB, selector_types=_SELECTORS,
            n_shapes=n_shapes, palette=_PALETTE,
        )
        if offender is not None:
            violations.append({
                "program": task.program,
                "shortcut": offender,
                "depth": depth,
                "n_shapes": n_shapes,
            })

    assert generated >= n_tasks * 0.2, (
        f"generated only {generated}/{n_tasks} — sampler too leaky"
    )
    assert not violations, (
        f"{len(violations)}/{generated} shortcut violations "
        f"at depth={depth} n_shapes={n_shapes}: first={violations[0]}"
    )
