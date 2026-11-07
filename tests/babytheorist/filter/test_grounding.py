from __future__ import annotations

import numpy as np
import pytest

from babytheorist.dominant import AbstractRef, AbstractStep
from babytheorist.filter.grounding import (
    GROUNDING_SYSTEM_VERSION,
    SlotRequirement,
    ground_abstract_step,
    infer_slot_requirements,
    slot_grounding_options,
)
from babytheorist.filter.state import FilterState
from babytheorist.program import SamplerDeadEnd


def test_grounding_version_is_stable():
    assert GROUNDING_SYSTEM_VERSION == "filter.abstract_grounding.v2"


def test_slot_requirement_filters_single_slot_by_initial_grid_shape():
    ref = AbstractRef(0, "A", "single")
    requirement = SlotRequirement(ref, required_grid_shape="hollow")

    options = slot_grounding_options(
        requirement,
        selector_types=frozenset({"by_index"}),
        n_shapes=3,
        palette=None,
        initial_grid_shapes=["solid", "hollow", "solid"],
    )

    assert [(selector.type, selector.param, override) for selector, override, _ in options] == [
        ("by_index", 2, None),
    ]


def test_slot_requirement_rejects_missing_required_grid_shape():
    ref = AbstractRef(0, "A", "single")
    requirement = SlotRequirement(ref, required_grid_shape="hollow")

    with pytest.raises(SamplerDeadEnd):
        slot_grounding_options(
            requirement,
            selector_types=frozenset({"by_index"}),
            n_shapes=2,
            palette=None,
            initial_grid_shapes=["solid", "solid"],
        )


def test_any_slot_grounding_balances_single_and_group_families():
    ref = AbstractRef(0, "A", "any")
    requirement = SlotRequirement(ref)

    options = slot_grounding_options(
        requirement,
        selector_types=frozenset({"ALL", "by_index"}),
        n_shapes=2,
        palette=None,
        initial_grid_shapes=None,
    )

    single_weight = sum(
        weight for selector, _override, weight in options
        if selector.type == "by_index"
    )
    group_weight = sum(
        weight for selector, _override, weight in options
        if selector.type == "ALL"
    )
    assert single_weight == pytest.approx(0.5)
    assert group_weight == pytest.approx(0.5)


def test_any_slot_grounding_reuses_selector_binding():
    ref = AbstractRef(0, "A", "any")
    abstract = (
        AbstractStep(ref, "translate_up"),
        AbstractStep(ref, "translate_right"),
    )
    requirements = infer_slot_requirements(abstract)
    bindings = {}

    first, _ = ground_abstract_step(
        abstract[0],
        bindings,
        selector_types=frozenset({"ALL", "by_index"}),
        n_shapes=2,
        palette=None,
        fs=FilterState.initial(2),
        rng=np.random.default_rng(0),
        requirements=requirements,
        initial_grid_shapes=None,
    )
    second, _ = ground_abstract_step(
        abstract[1],
        bindings,
        selector_types=frozenset({"ALL", "by_index"}),
        n_shapes=2,
        palette=None,
        fs=FilterState.initial(2),
        rng=np.random.default_rng(1),
        requirements=requirements,
        initial_grid_shapes=None,
    )

    assert first.selector == second.selector
    assert bindings[ref][0] == first.selector


def test_relational_refs_force_by_index_before_relational_step():
    src = AbstractRef(0, "A", "single")
    tgt = AbstractRef(0, "B", "single")
    abstract = (
        AbstractStep(src, "rot90"),
        AbstractStep(src, "align_row", tgt),
    )
    requirements = infer_slot_requirements(abstract)

    assert requirements[src].force_by_index
    assert requirements[tgt].force_by_index

    bindings = {}
    step, override = ground_abstract_step(
        abstract[0],
        bindings,
        selector_types=frozenset({"by_index", "by_color"}),
        n_shapes=3,
        palette=(1, 2, 3),
        fs=FilterState.initial(3),
        rng=np.random.default_rng(0),
        requirements=requirements,
        initial_grid_shapes=None,
    )

    assert step.selector.type == "by_index"
    assert override is None
    assert bindings[src][0].type == "by_index"
