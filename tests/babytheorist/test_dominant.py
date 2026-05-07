from __future__ import annotations

import numpy as np
from babytheorist.core import RELATIONAL_PRIMITIVE_NAMES
from babytheorist.dominant import (
    AbstractRef,
    AbstractStep,
    abstract_provenance_entry_ids,
    abstract_provenance_from_jsonable,
    abstract_provenance_to_jsonable,
    abstract_pool_from_jsonable,
    abstract_pool_to_jsonable,
    build_abstract_pool,
    build_append_abstract_pool,
    sample_abstract_program,
    validate_canonical,
)
from babytheorist.filter.sampler import sample_program_with_constraint
from babytheorist.program import build_bigram


def test_abstract_pool_entries_match_depth_and_validate():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    vocab = frozenset(primitives)
    selectors = frozenset({"ALL", "by_index"})
    bigram = build_bigram(primitives)
    pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=3,
        n_shapes=3,
        selector_types=selectors,
        vocab=vocab,
        pool_size=8,
        palette=(1, 2, 3),
        sample_count=100,
    )

    assert set(pool) == {1, 2, 3}
    assert pool[3]
    for depth, entries in pool.items():
        for program in entries:
            assert len(program) == depth
            assert validate_canonical(
                program,
                n_shapes=3,
                vocab=vocab,
                selector_types=selectors,
                initial_grid_shapes=None,
                palette=(1, 2, 3),
            )


def test_abstract_pool_backfills_after_validation_rejects():
    primitives = [
        "rot90", "flip_h", "flip_v", "next_color",
        "translate_up", "translate_down", "translate_left", "translate_right",
    ]
    bigram = build_bigram(primitives)

    pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=10,
        n_shapes=1,
        selector_types=frozenset({"by_index"}),
        vocab=frozenset(primitives),
        pool_size=10,
        rng=np.random.default_rng(123),
        palette=(1, 2, 3),
        sample_count=100,
    )

    assert len(pool[1]) == len(primitives)
    assert all(len(pool[depth]) == 10 for depth in range(2, 11))
    for entries in pool.values():
        for program in entries:
            assert validate_canonical(
                program,
                n_shapes=1,
                vocab=frozenset(primitives),
                selector_types=frozenset({"by_index"}),
                initial_grid_shapes=None,
                palette=(1, 2, 3),
            )


def test_abstract_pool_depth_one_has_one_uniform_entry_per_primitive():
    primitives = [
        "rot90", "flip_h", "flip_v", "next_color",
        "translate_up", "translate_down", "translate_left", "translate_right",
        "swap_colors", "align_row", "align_col",
    ]
    bigram = build_bigram(primitives)

    pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=2,
        n_shapes=2,
        selector_types=frozenset({
            "ALL", "by_index", "by_color", "is_hollow",
            "topmost", "bottommost", "leftmost", "rightmost",
        }),
        vocab=frozenset(primitives),
        pool_size=10,
        rng=np.random.default_rng(123),
        palette=(1, 2, 3),
        sample_count=100,
    )

    non_rel = [p for p in primitives if p not in RELATIONAL_PRIMITIVE_NAMES]
    relational = [p for p in primitives if p in RELATIONAL_PRIMITIVE_NAMES]
    assert len(pool[1]) == len(non_rel) + len(relational)
    assert {program[0].primitive for program in pool[1]} == set(primitives)
    assert set(pool[1].values()) == {1.0}
    kinds_by_primitive: dict[str, set[str]] = {}
    for program in pool[1]:
        step = program[0]
        kinds_by_primitive.setdefault(step.primitive, set()).add(step.source.kind)
        if step.primitive in RELATIONAL_PRIMITIVE_NAMES:
            assert step.source.kind == "single"
            assert step.target is not None
            assert step.target.kind == "single"
        else:
            assert step.source.kind == "any"
            assert step.target is None
    for primitive in non_rel:
        assert kinds_by_primitive[primitive] == {"any"}
    for primitive in relational:
        assert kinds_by_primitive[primitive] == {"single"}


def test_abstract_pool_rejects_opposing_translate_on_same_slot():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "translate_up"),
        AbstractStep(AbstractRef(0, "A", "single"), "rot90"),
        AbstractStep(AbstractRef(0, "A", "single"), "translate_down"),
    )

    assert not validate_canonical(
        abstract,
        n_shapes=1,
        vocab=frozenset({"translate_up", "translate_down", "rot90"}),
        selector_types=frozenset({"by_index"}),
        initial_grid_shapes=None,
        palette=(1, 2, 3),
    )


def test_abstract_pool_allows_opposing_translate_on_different_slots():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "translate_up"),
        AbstractStep(AbstractRef(1, "A", "single"), "translate_down"),
    )

    assert validate_canonical(
        abstract,
        n_shapes=2,
        vocab=frozenset({"translate_up", "translate_down"}),
        selector_types=frozenset({"by_index"}),
        initial_grid_shapes=None,
        palette=(1, 2, 3),
    )


def test_abstract_pool_json_roundtrip():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    vocab = frozenset(primitives)
    selectors = frozenset({"ALL", "by_index"})
    bigram = build_bigram(primitives)
    pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=3,
        n_shapes=3,
        selector_types=selectors,
        vocab=vocab,
        pool_size=8,
        palette=(1, 2, 3),
        sample_count=100,
    )

    restored = abstract_pool_from_jsonable(abstract_pool_to_jsonable(pool))

    assert restored == pool
    assert any(
        step.source.kind == "any"
        for entries in restored.values()
        for program in entries
        for step in program
        if step.target is None
    )


def test_abstract_pool_tracks_entry_ids_and_parent_counts():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    bigram = build_bigram(primitives)

    build = build_abstract_pool(
        bigram,
        primitives,
        max_depth=3,
        n_shapes=2,
        selector_types=frozenset({"ALL", "by_index"}),
        vocab=frozenset(primitives),
        pool_size=5,
        palette=(1, 2, 3),
        sample_count=80,
        track_provenance=True,
        id_prefix="base:",
    )

    assert build.pool[3]
    ids = abstract_provenance_entry_ids(build.provenance)
    for depth, entries in build.pool.items():
        assert set(ids[depth]) == set(entries)
        for record in build.provenance[depth].values():
            assert record["id"].startswith(f"base:d{depth}.")
            assert record["depth"] == depth

    for program, score in build.pool[3].items():
        record = build.provenance[3][program]
        assert record["count"] == sum(parent["count"] for parent in record["parents"])
        assert record["score"] == score
        assert record["parents"]
        for parent in record["parents"]:
            assert parent["left_id"].startswith("base:d")
            assert parent["right_id"].startswith("base:d")
            assert parent["mode"] in {"reuse", "fresh"}

    restored = abstract_provenance_from_jsonable(
        abstract_provenance_to_jsonable(build.provenance)
    )
    assert restored == build.provenance


def test_append_pool_entries_use_introduced_primitive_once_and_cap_depth():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    selectors = frozenset({"ALL", "by_index"})
    bigram = build_bigram(primitives)
    base_pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=3,
        n_shapes=2,
        selector_types=selectors,
        vocab=frozenset(primitives),
        pool_size=10,
        palette=(1, 2, 3),
        sample_count=100,
    )

    append_pool = build_append_abstract_pool(
        base_pool,
        "hollow",
        max_depth=3,
        n_shapes=2,
        selector_types=selectors,
        vocab=frozenset(primitives + ["hollow"]),
        pool_size=5,
        palette=(1, 2, 3),
        sample_count=100,
    )

    assert len(append_pool[1]) == 1
    assert {program[0].primitive for program in append_pool[1]} == {"hollow"}
    assert {program[0].source.kind for program in append_pool[1]} == {"any"}
    assert append_pool[3]
    for depth, entries in append_pool.items():
        assert len(entries) <= 5
        for program in entries:
            assert len(program) == depth
            assert sum(step.primitive == "hollow" for step in program) == 1


def test_append_pool_tracks_append_and_inherited_parent_ids():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    selectors = frozenset({"ALL", "by_index"})
    bigram = build_bigram(primitives)
    base_build = build_abstract_pool(
        bigram,
        primitives,
        max_depth=3,
        n_shapes=2,
        selector_types=selectors,
        vocab=frozenset(primitives),
        pool_size=6,
        palette=(1, 2, 3),
        sample_count=80,
        track_provenance=True,
        id_prefix="base:",
    )

    append_build = build_append_abstract_pool(
        base_build.pool,
        "hollow",
        max_depth=3,
        n_shapes=2,
        selector_types=selectors,
        vocab=frozenset(primitives + ["hollow"]),
        pool_size=5,
        palette=(1, 2, 3),
        sample_count=80,
        inherited_entry_ids=abstract_provenance_entry_ids(base_build.provenance),
        track_provenance=True,
        id_prefix="append:hollow:",
    )

    assert append_build.pool[3]
    for program in append_build.pool[3]:
        record = append_build.provenance[3][program]
        assert record["id"].startswith("append:hollow:d3.")
        assert record["count"] == sum(parent["count"] for parent in record["parents"])
        assert record["parents"]
        for parent in record["parents"]:
            assert parent["source"] in {"append_left", "append_right"}
            parent_ids = (parent["left_id"], parent["right_id"])
            assert any(pid.startswith("append:hollow:") for pid in parent_ids)
            assert any(pid.startswith("base:") for pid in parent_ids)


def test_fill_append_pool_infers_targeted_hollow_input_condition():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    selectors = frozenset({"by_index"})
    bigram = build_bigram(primitives)
    base_pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=2,
        n_shapes=1,
        selector_types=selectors,
        vocab=frozenset(primitives),
        pool_size=10,
        palette=(1, 2, 3),
        sample_count=100,
    )

    append_pool = build_append_abstract_pool(
        base_pool,
        "fill_interior",
        max_depth=2,
        n_shapes=1,
        selector_types=selectors,
        vocab=frozenset(primitives + ["fill_interior"]),
        pool_size=5,
        initial_grid_shapes=["solid"],
        palette=(1, 2, 3),
        sample_count=100,
    )

    assert append_pool[1]
    for entries in append_pool.values():
        for program in entries:
            assert sum(step.primitive == "fill_interior" for step in program) == 1
            assert not validate_canonical(
                program,
                n_shapes=1,
                vocab=frozenset(primitives + ["fill_interior"]),
                selector_types=selectors,
                initial_grid_shapes=["solid"],
                palette=(1, 2, 3),
            )
            assert validate_canonical(
                program,
                n_shapes=1,
                vocab=frozenset(primitives + ["fill_interior"]),
                selector_types=selectors,
                initial_grid_shapes=["hollow"],
                palette=(1, 2, 3),
            )


def test_abstract_sample_walking_uses_fresh_scopes():
    primitives = ["rot90", "flip_h", "translate_up", "translate_right"]
    vocab = frozenset(primitives)
    selectors = frozenset({"ALL", "by_index"})
    bigram = build_bigram(primitives)
    pool = build_abstract_pool(
        bigram,
        primitives,
        max_depth=2,
        n_shapes=3,
        selector_types=selectors,
        vocab=vocab,
        pool_size=8,
        palette=(1, 2, 3),
        sample_count=100,
    )

    program = sample_abstract_program(
        pool, bigram, primitives, depth=4, rng=np.random.default_rng(0), p_direct=0.0,
    )
    scopes = {
        ref.scope
        for step in program
        for ref in ((step.source,) if step.target is None else (step.source, step.target))
    }
    assert len(program) == 4
    assert len(scopes) >= 2


def test_abstract_sample_walking_can_use_max_pool_level_chunks():
    p1 = (AbstractStep(AbstractRef(0, "A", "single"), "rot90"),)
    p2 = (
        AbstractStep(AbstractRef(0, "A", "single"), "rot90"),
        AbstractStep(AbstractRef(0, "A", "single"), "flip_h"),
    )
    pool = {1: {p1: 1.0}, 2: {p2: 1.0}}
    bigram = np.ones((2, 2), dtype=np.float64) / 2.0

    program = sample_abstract_program(
        pool,
        bigram,
        ["rot90", "flip_h"],
        depth=4,
        rng=np.random.default_rng(0),
        p_direct=0.0,
    )

    assert [step.source.scope for step in program] == [0, 0, 1, 1]


def test_concrete_grounding_reuses_slot_selector():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "translate_up"),
        AbstractStep(AbstractRef(0, "A", "single"), "translate_right"),
    )
    program, constraint = sample_program_with_constraint(
        depth=2,
        n_shapes=3,
        vocab=frozenset({"translate_up", "translate_right"}),
        selector_types=frozenset({"by_index"}),
        rng=np.random.default_rng(1),
        palette=(1, 2, 3),
        abstract_program=abstract,
    )

    assert [step.selector for step in program] == [program[0].selector] * 2
    assert not constraint.is_infeasible()


def test_relational_abstract_slots_ground_to_distinct_indices():
    relational = sorted(RELATIONAL_PRIMITIVE_NAMES)[0]
    abstract = (
        AbstractStep(
            AbstractRef(0, "A", "single"),
            relational,
            AbstractRef(0, "B", "single"),
        ),
    )
    program, _ = sample_program_with_constraint(
        depth=1,
        n_shapes=3,
        vocab=frozenset({relational}),
        selector_types=frozenset({"by_index"}),
        rng=np.random.default_rng(2),
        palette=(1, 2, 3),
        abstract_program=abstract,
    )

    assert program[0].selector.type == "by_index"
    assert program[0].target.type == "by_index"
    assert program[0].selector.param != program[0].target.param


def test_relational_slot_is_prebound_to_by_index():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "next_color"),
        AbstractStep(
            AbstractRef(0, "A", "single"),
            "align_row",
            AbstractRef(0, "B", "single"),
        ),
    )

    program, _ = sample_program_with_constraint(
        depth=2,
        n_shapes=3,
        vocab=frozenset({"next_color", "align_row"}),
        selector_types=frozenset({"by_index", "by_color"}),
        rng=np.random.default_rng(4),
        palette=(1, 2, 3),
        abstract_program=abstract,
    )

    assert program[0].selector.type == "by_index"
    assert program[1].selector == program[0].selector
    assert program[1].target.type == "by_index"


def test_abstract_grounding_supports_positional_selectors():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "next_color"),
        AbstractStep(AbstractRef(0, "A", "single"), "next_color"),
    )

    program, constraint = sample_program_with_constraint(
        depth=2,
        n_shapes=3,
        vocab=frozenset({"next_color"}),
        selector_types=frozenset({"topmost"}),
        rng=np.random.default_rng(0),
        palette=(1, 2, 3),
        initial_grid_shapes=["solid", "solid", "solid"],
        abstract_program=abstract,
    )

    assert [step.selector.type for step in program] == ["topmost", "topmost"]
    assert program[0].selector == program[1].selector
    assert len(constraint.positional_orderings) == 2


def test_abstract_grounding_preserves_v055_positional_reuse_semantics():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "translate_down"),
        AbstractStep(AbstractRef(0, "A", "single"), "translate_down"),
    )

    program, constraint = sample_program_with_constraint(
        depth=2,
        n_shapes=3,
        vocab=frozenset({"translate_down"}),
        selector_types=frozenset({"topmost"}),
        rng=np.random.default_rng(0),
        palette=(1, 2, 3),
        initial_grid_shapes=["solid", "solid", "solid"],
        abstract_program=abstract,
    )

    assert [step.selector.type for step in program] == ["topmost", "topmost"]
    assert program[0].selector == program[1].selector
    assert len(constraint.positional_orderings) == 2


def test_repeated_by_color_slot_reuses_selector_commitment():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "translate_up"),
        AbstractStep(AbstractRef(0, "A", "single"), "translate_right"),
    )

    program, constraint = sample_program_with_constraint(
        depth=2,
        n_shapes=3,
        vocab=frozenset({"translate_up", "translate_right"}),
        selector_types=frozenset({"by_color"}),
        rng=np.random.default_rng(5),
        palette=(1, 2, 3),
        initial_grid_shapes=["solid", "solid", "solid"],
        abstract_program=abstract,
    )

    assert program[0].selector == program[1].selector
    required = [
        shape.required_primary_color
        for shape in constraint.shapes
        if shape.required_primary_color is not None
    ]
    assert required == [program[0].selector.param]


def test_canonical_validation_respects_initial_hollow_state():
    abstract = (AbstractStep(AbstractRef(0, "A", "single"), "fill_interior"),)
    kwargs = dict(
        n_shapes=1,
        vocab=frozenset({"fill_interior"}),
        selector_types=frozenset({"by_index"}),
        palette=(1, 2, 3),
    )

    assert not validate_canonical(
        abstract, initial_grid_shapes=["solid"], **kwargs,
    )
    assert validate_canonical(
        abstract, initial_grid_shapes=["hollow"], **kwargs,
    )


def test_abstract_fill_ref_requires_hollow_grounding():
    abstract = (
        AbstractStep(AbstractRef(0, "A", "single"), "fill_interior"),
    )

    program, constraint = sample_program_with_constraint(
        depth=1,
        n_shapes=2,
        vocab=frozenset({"fill_interior"}),
        selector_types=frozenset({"by_index"}),
        rng=np.random.default_rng(0),
        palette=(1, 2, 3),
        initial_grid_shapes=["hollow", "solid"],
        abstract_program=abstract,
        ref_initial_grid_shapes={AbstractRef(0, "A", "single"): "hollow"},
    )

    assert program[0].selector.param == 1
    assert constraint.shapes[0].required_grid_shape == "hollow"
    assert constraint.shapes[1].required_grid_shape == "solid"


def test_dominant_filter_generation_passes_initial_hollow_to_pool():
    from babytheorist.generation import LevelConfig, TaskConfig, generate_filter_dataset

    level = LevelConfig(
        level=1,
        primitives=["fill_interior"],
        selectors=("by_index",),
        fixed_depth=1,
        use_dominant=True,
    )
    task_cfg = TaskConfig(
        mode="train",
        n_shapes_range=(1, 1),
        shape_size_range=(4, 8),
        max_dataset_attempts_per_task=10,
    )

    tasks = generate_filter_dataset(
        level,
        task_cfg,
        n_tasks=1,
        seed=0,
        initial_hollow_prob=1.0,
        pool_seed=123,
    )

    assert tasks[0].program[0].primitive == "fill_interior"


def test_pool_initial_grid_shapes_only_splits_all_hollow_cache():
    from babytheorist.generation.dominant import _pool_initial_grid_shapes

    assert _pool_initial_grid_shapes(["hollow", "hollow"]) == [
        "hollow", "hollow",
    ]
    assert _pool_initial_grid_shapes(["solid", "solid"]) is None
    assert _pool_initial_grid_shapes(["solid", "hollow"]) is None
