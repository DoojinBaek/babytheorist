"""Unit tests for R7' trace-identity rule."""

from __future__ import annotations

from babytheorist.core import ALL, Selector, Step
from babytheorist.filter.state import FilterState
from babytheorist.filter.trace_identity import (
    per_shape_write_trace,
    check_trace_identity,
)


def _build_fs(program, n_shapes):
    fs = FilterState.initial(n_shapes)
    for step in program:
        fs = fs.apply(step)
    return fs


def _step_all(prim):
    return Step(selector=ALL, primitive=prim)


def _step_idx(prim, k):
    return Step(selector=Selector("by_index", k), primitive=prim)


# === per_shape_write_trace ================================================


def test_trace_all_selector_writes_every_shape():
    program = [_step_all("rot90"), _step_all("flip_h")]
    fs = _build_fs(program, 2)
    traces = per_shape_write_trace(program, fs)
    assert traces[0] == [("rot90", "self"), ("flip_h", "self")]
    assert traces[1] == [("rot90", "self"), ("flip_h", "self")]


def test_trace_by_index_restricts_writes():
    program = [_step_idx("rot90", 1), _step_idx("flip_h", 2)]
    fs = _build_fs(program, 2)
    traces = per_shape_write_trace(program, fs)
    assert traces[0] == [("rot90", "self")]
    assert traces[1] == [("flip_h", "self")]


def test_trace_untouched_shape_is_empty():
    program = [_step_idx("rot90", 1)]
    fs = _build_fs(program, 3)
    traces = per_shape_write_trace(program, fs)
    assert traces[0] == [("rot90", "self")]
    assert traces[1] == []
    assert traces[2] == []


# === check_trace_identity: violations =====================================


def test_violation_pure_by_index_collapses_to_all():
    # [by_idx(1).rot90, by_idx(2).rot90] — both shapes same trace,
    # program depth 2 > trace length 1 → violation.
    program = [_step_idx("rot90", 1), _step_idx("rot90", 2)]
    fs = _build_fs(program, 2)
    v = check_trace_identity(program, fs)
    assert v is not None
    assert v.rule == "R7'"


def test_violation_mixed_all_plus_split_by_index():
    # [ALL.rot90, by_idx(1).flip_h, by_idx(2).flip_h] — both shapes share
    # trace [(rot90), (flip_h)] of length 2; program depth 3 → violation.
    program = [
        _step_all("rot90"),
        _step_idx("flip_h", 1),
        _step_idx("flip_h", 2),
    ]
    fs = _build_fs(program, 2)
    v = check_trace_identity(program, fs)
    assert v is not None
    assert v.rule == "R7'"


def test_violation_three_shapes_via_by_index():
    program = [
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
        _step_idx("rot90", 3),
    ]
    fs = _build_fs(program, 3)
    v = check_trace_identity(program, fs)
    assert v is not None


# === check_trace_identity: passes =========================================


def test_pass_when_traces_differ_by_order():
    # [by_idx(1).rot90, ALL.flip_h, by_idx(2).rot90] —
    # shape 1 trace = [(rot90), (flip_h)], shape 2 = [(flip_h), (rot90)]
    # Distinct traces → not aliasable.
    program = [
        _step_idx("rot90", 1),
        _step_all("flip_h"),
        _step_idx("rot90", 2),
    ]
    fs = _build_fs(program, 2)
    assert check_trace_identity(program, fs) is None


def test_pass_all_selector_program_has_length_equals_trace():
    # [ALL.rot90, ALL.flip_h] — trace length 2, depth 2, no excess.
    program = [_step_all("rot90"), _step_all("flip_h")]
    fs = _build_fs(program, 2)
    assert check_trace_identity(program, fs) is None


def test_pass_single_shape_scene_skipped():
    # n_shapes = 1 — rule is a no-op.
    program = [_step_all("rot90"), _step_idx("flip_h", 1)]
    fs = _build_fs(program, 1)
    assert check_trace_identity(program, fs) is None


def test_pass_subset_group_not_collapsible_in_all_plus_by_index():
    # [by_idx(1).rot90, by_idx(2).rot90, by_idx(3).flip_h] n=3.
    # Shapes 1 and 2 share trace, shape 3 differs. With only ALL+by_index,
    # no selector picks {1, 2} as a subset — cannot collapse. Pass.
    program = [
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
        _step_idx("flip_h", 3),
    ]
    fs = _build_fs(program, 3)
    assert check_trace_identity(program, fs) is None


# === check_trace_identity: is_hollow subset expressibility ================


def test_violation_is_hollow_collapses_two_hollow_shapes():
    # n=3, shapes 0 and 1 initially hollow, shape 2 solid.
    # P = [by_idx(1).rot90, by_idx(2).rot90, by_idx(3).flip_h]
    # shapes 0 and 1 share trace [(rot90, self)]; subset is expressible
    # as is_hollow. With is_hollow in selector_types, R7' must fire.
    program = [
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
        _step_idx("flip_h", 3),
    ]
    fs = FilterState.initial(3, grid_shapes=["hollow", "hollow", "solid"])
    for step in program:
        fs = fs.apply(step)
    v = check_trace_identity(
        program, fs,
        selector_types=frozenset({"ALL", "by_index", "is_hollow"}),
    )
    assert v is not None, "should flag is_hollow-aliased subgroup"
    assert v.rule == "R7'"


def test_pass_is_hollow_requires_matching_hollow_subset():
    # Same program but shape 1 is solid (shape 0 hollow only). The
    # trace-equivalent subgroup {0, 1} does NOT match the is_hollow
    # resolution {0}, so R7' should NOT fire.
    program = [
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
        _step_idx("flip_h", 3),
    ]
    fs = FilterState.initial(3, grid_shapes=["hollow", "solid", "solid"])
    for step in program:
        fs = fs.apply(step)
    assert check_trace_identity(
        program, fs,
        selector_types=frozenset({"ALL", "by_index", "is_hollow"}),
    ) is None


def test_pass_is_hollow_disabled_in_selector_types():
    # Even though {0, 1} matches hollow and share trace, is_hollow is
    # not available in selector_types — no collapse path exists.
    program = [
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
        _step_idx("flip_h", 3),
    ]
    fs = FilterState.initial(3, grid_shapes=["hollow", "hollow", "solid"])
    for step in program:
        fs = fs.apply(step)
    assert check_trace_identity(
        program, fs,
        selector_types=frozenset({"ALL", "by_index"}),
    ) is None


# === check_trace_identity: by_color subset expressibility =================


def test_violation_by_color_collapses_subgroup_under_constraint():
    # n=3. by_color(c=3) at step 0 committed to {0, 1} pins primaries
    # 0,1=3, forbids 3 on shape 2. Shapes 0 and 1 share trace on rot90
    # written through by_index in a follow-up step pattern.
    # Here we construct:
    #   P = [by_color(3).rot90 committed to {0,1},
    #        by_idx(1).flip_h, by_idx(2).flip_h, by_idx(3).rot90]
    # Shapes 0, 1 share trace [(rot90, self), (flip_h, self)],
    # shape 2 has [(rot90, self)].
    # Under the accumulated constraint, by_color(3) at step 1 (between
    # rot90 and flip_h) resolves to {0, 1} — enabling collapse of the
    # two flip_h steps.
    from babytheorist.filter.rules_input import accumulate_constraint
    from babytheorist.filter.constraint import InputConstraint

    palette = (1, 2, 3, 4, 5)
    sel_types = frozenset({"ALL", "by_index", "by_color"})

    step_by_color = Step(
        selector=Selector("by_color", 3), primitive="rot90",
    )
    step_flip1 = _step_idx("flip_h", 1)
    step_flip2 = _step_idx("flip_h", 2)
    step_rot_3 = _step_idx("rot90", 3)

    fs = FilterState.initial(3)
    fs = fs.apply(step_by_color, override_indices=[0, 1])
    fs = fs.apply(step_flip1)
    fs = fs.apply(step_flip2)
    fs = fs.apply(step_rot_3)
    program = [step_by_color, step_flip1, step_flip2, step_rot_3]

    constraint = accumulate_constraint(
        program, fs, palette=palette,
    )

    v = check_trace_identity(
        program, fs, selector_types=sel_types,
        constraint=constraint, palette=palette,
    )
    assert v is not None
    assert v.rule == "R7'"


def test_pass_by_color_disabled_no_collapse():
    """Without by_color in selector_types, the subgroup is not
    collapsible even when constraints would support it."""
    from babytheorist.filter.rules_input import accumulate_constraint

    palette = (1, 2, 3, 4, 5)

    step_by_color = Step(
        selector=Selector("by_color", 3), primitive="rot90",
    )
    step_flip1 = _step_idx("flip_h", 1)
    step_flip2 = _step_idx("flip_h", 2)

    fs = FilterState.initial(3)
    fs = fs.apply(step_by_color, override_indices=[0, 1])
    fs = fs.apply(step_flip1)
    fs = fs.apply(step_flip2)
    program = [step_by_color, step_flip1, step_flip2]

    constraint = accumulate_constraint(program, fs, palette=palette)

    # selector_types lacks by_color → can't express {0, 1} as a subset.
    # Also n=3, so ALL would need the full set. Therefore no collapse.
    assert check_trace_identity(
        program, fs,
        selector_types=frozenset({"ALL", "by_index"}),
        constraint=constraint, palette=palette,
    ) is None


def test_pass_is_hollow_changes_mid_program_breaks_expressibility():
    # n=3, shapes 0, 1 hollow initially, shape 2 solid. At step 0 the
    # is_hollow resolution is {0, 1}. Step 0 applies fill_interior to
    # is_hollow → shapes 0, 1 become solid. At step 1 the hollow set
    # is now empty.
    #
    # P = [is_hollow.fill_interior, by_idx(1).rot90, by_idx(2).rot90]
    # shapes 0 and 1 share trace [(fill_interior, self), (rot90, self)],
    # but at step 2 the hollow subset is {} — {0, 1} is no longer
    # expressible as is_hollow throughout. R7' must pass.
    program = [
        Step(selector=Selector("is_hollow"), primitive="fill_interior"),
        _step_idx("rot90", 1),
        _step_idx("rot90", 2),
    ]
    fs = FilterState.initial(3, grid_shapes=["hollow", "hollow", "solid"])
    for step in program:
        fs = fs.apply(step)
    assert check_trace_identity(
        program, fs,
        selector_types=frozenset({"ALL", "by_index", "is_hollow"}),
    ) is None
