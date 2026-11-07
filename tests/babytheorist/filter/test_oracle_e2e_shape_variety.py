"""T2-1 — filter works with the full default shape factory.

Drops the solid-only factory used by the other e2e tests and swaps in
``(solid, bordered, hollow, patterned)`` to prove that:

1. ``initial_hollow_prob=0.0`` (baseline) still produces scenes whose
   actual ``grid_shape`` is always ``"solid"`` — the explicit
   ``required_grid_shape="solid"`` constraint blocks hollow-generated
   shapes at the scene sampler.
2. ``initial_hollow_prob=0.5`` lets the sampler produce programs that
   begin with ``fill_interior`` — something the baseline cannot do —
   and the oracle still reports zero shortcut violations.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.filter.generate import generate_task
from babytheorist.filter.oracle import scene_has_shortcut
from babytheorist.filter.sampler import NON_RELATIONAL_VOCAB
from babytheorist.filter.scene_sampler import grid_shape_of
from babytheorist.shapes import ShapeFactory, bordered, hollow, patterned, solid


_PALETTE = (1, 2, 3, 4, 5)
_VOCAB = NON_RELATIONAL_VOCAB
_SELECTORS = frozenset({"ALL", "by_index"})


def _default_factory() -> ShapeFactory:
    return ShapeFactory(
        generators=(solid, bordered, hollow, patterned),
        size_range=(10, 20),
    )


@pytest.mark.parametrize("depth", [2, 3])
def test_default_factory_solid_only_enforced(depth):
    """Baseline: ``initial_hollow_prob=0.0`` must still reject
    hollow-generated shapes via the ``required_grid_shape="solid"``
    constraint."""
    rng = np.random.default_rng(depth * 7 + 11)
    factory = _default_factory()

    violations = []
    nonsolid = 0
    generated = 0
    for _ in range(80 * 4):
        if generated >= 80:
            break
        task = generate_task(
            depth=depth, n_shapes=2, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
            vocab=_VOCAB, selector_types=_SELECTORS,
            initial_hollow_prob=0.0,
        )
        if task is None:
            continue
        generated += 1
        for s in task.input_shapes:
            if grid_shape_of(s) != "solid":
                nonsolid += 1
        offender = scene_has_shortcut(
            task.output_grid, task.input_shapes, world_size=24,
            depth=depth, vocab=_VOCAB, selector_types=_SELECTORS,
            n_shapes=2,
        )
        if offender is not None:
            violations.append(offender)

    assert generated >= 40, f"sampler too leaky: {generated}/80"
    assert nonsolid == 0, (
        f"{nonsolid} shapes had non-solid grid_shape under hollow_prob=0.0"
    )
    assert not violations, (
        f"{len(violations)} shortcut violations at depth={depth}"
    )


def test_initial_hollow_prob_unlocks_fill_interior_first():
    """With hollow-initial enabled, at least one sampled program
    must start with ``fill_interior``."""
    rng = np.random.default_rng(5)
    factory = _default_factory()
    first_fills = 0
    violations = 0
    generated = 0
    for _ in range(400):
        if generated >= 200:
            break
        task = generate_task(
            depth=3, n_shapes=2, factory=factory,
            palette=_PALETTE, world_size=24, rng=rng,
            vocab=_VOCAB, selector_types=_SELECTORS,
            initial_hollow_prob=0.5,
        )
        if task is None:
            continue
        generated += 1
        if task.program[0].primitive == "fill_interior":
            first_fills += 1
        offender = scene_has_shortcut(
            task.output_grid, task.input_shapes, world_size=24,
            depth=3, vocab=_VOCAB, selector_types=_SELECTORS,
            n_shapes=2,
        )
        if offender is not None:
            violations += 1

    assert generated >= 100, f"sampler too leaky: {generated}/200"
    assert first_fills > 0, (
        "hollow_prob=0.5 did not produce any fill_interior-first programs"
    )
    assert violations == 0, f"{violations} oracle violations"
