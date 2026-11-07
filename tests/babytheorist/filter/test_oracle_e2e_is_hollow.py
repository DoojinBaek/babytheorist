"""End-to-end reliability with ``is_hollow`` in selector_types (T1-1α).

Mirrors ``test_oracle_e2e_shape_axis`` but adds ``is_hollow`` to the
selector scope for both the filter and the oracle. ``initial_hollow_prob``
is non-zero so hollow-initial scenes (and ``fill_interior``-first programs)
can appear.
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
_SELECTORS = frozenset({"ALL", "by_index", "is_hollow"})


def _factory() -> ShapeFactory:
    return ShapeFactory(
        generators=(solid, bordered, hollow, patterned),
        size_range=(10, 20),
    )


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (2, 2, 40),
    (3, 2, 40),
])
def test_no_shortcuts_is_hollow(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 1000 + n_shapes + 7)
    factory = _factory()

    violations = []
    generated = 0
    used_is_hollow = 0
    for _ in range(n_tasks * 10):
        if generated >= n_tasks:
            break
        task = generate_task(
            depth=depth, n_shapes=n_shapes, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
            vocab=_VOCAB, selector_types=_SELECTORS,
            initial_hollow_prob=0.5,
        )
        if task is None:
            continue
        generated += 1

        if any(s.selector.type == "is_hollow" for s in task.program):
            used_is_hollow += 1

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

    assert generated >= n_tasks * 0.3, (
        f"generated only {generated}/{n_tasks} — sampler too leaky"
    )
    assert not violations, (
        f"{len(violations)}/{generated} shortcut violations "
        f"at depth={depth} n_shapes={n_shapes}: first={violations[0]}"
    )
    # Sanity: at least one task actually used is_hollow. Without it
    # this test merely duplicates shape_axis coverage.
    assert used_is_hollow > 0, (
        f"no task used is_hollow at depth={depth} n_shapes={n_shapes} — "
        "sampler not exercising the new selector"
    )
