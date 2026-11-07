"""End-to-end reliability with ``by_color`` in selector_types (T1-1β).

by_color commits are scene-time: the filter picks ``(c, S)`` at
sampling time and records per-shape ``primary_color`` requirements.
The scene sampler enforces them; the oracle enumerates every
``by_color(c)`` selector for ``c ∈ palette``.

Oracle cost is ``(|vocab| × (|{ALL,by_index}| + |palette|))^(depth-1)``
— grows fast. Tests are kept at depth 2 to stay in the seconds-per-task
regime.
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
_SELECTORS = frozenset({"ALL", "by_index", "by_color"})


def _factory() -> ShapeFactory:
    return ShapeFactory(generators=(solid,), size_range=(10, 20))


@pytest.mark.parametrize("depth,n_shapes,n_tasks", [
    (2, 2, 30),
])
def test_no_shortcuts_by_color(depth, n_shapes, n_tasks):
    rng = np.random.default_rng(depth * 101 + n_shapes + 3)
    factory = _factory()

    violations = []
    generated = 0
    used_by_color = 0
    for _ in range(n_tasks * 10):
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

        if any(s.selector.type == "by_color" for s in task.program):
            used_by_color += 1

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
        f"{len(violations)}/{generated} shortcut violations: "
        f"first={violations[0]}"
    )
    assert used_by_color > 0, (
        "no task used by_color — sampler not exercising the new selector"
    )
