"""Tests for babytheorist.filter.rules_input (I1)."""

from __future__ import annotations

import pytest

from babytheorist.core import ALL, Selector, Step
from babytheorist.filter.constraint import InputConstraint
from babytheorist.filter.constraint import PrimaryColorCoincidenceForbid
from babytheorist.filter.d4_tables import (
    D4_IDENTITY,
    D4_SUBGROUPS,
    stabilizers_allowed_at_depth,
)
from babytheorist.filter.constraint import InfeasibleConstraint
from babytheorist.filter.rules_input import (
    accumulate_constraint,
    apply_align_coincidence_forbids,
    apply_by_color_commitments,
    apply_execution_safety_constraints,
    apply_rule_i1,
    apply_swap_next_color_forbids,
    apply_translate_align_coincidence_forbids,
    apply_vocab_requirements,
)
from babytheorist.filter.state import FilterState
from babytheorist.symbolic import D4, LinExpr


_ROT = Step(selector=ALL, primitive="rot90")
_FLIP_H = Step(selector=ALL, primitive="flip_h")
_FLIP_V = Step(selector=ALL, primitive="flip_v")


def _fs(program, n_shapes: int) -> FilterState:
    fs = FilterState.initial(n_shapes)
    for step in program:
        fs = fs.apply(step)
    return fs


def _by_idx(k, prim):
    return Step(selector=Selector("by_index", k), primitive=prim)


# === I1 ===================================================================


def test_i1_depth_one_rot90_forbids_stabilizers_containing_r():
    """[rot90] on 1 shape: net = r, depth = 1. Allowed stabilizers are
    those NOT containing r."""
    program = [_ROT]
    fs = _fs(program, 1)
    c = apply_rule_i1(program, fs, InputConstraint.empty(1))
    allowed = c.shapes[0].allowed_stabilizers
    r = D4(1, False)
    for H in D4_SUBGROUPS:
        if r in H:
            assert H not in allowed
        else:
            assert H in allowed


def test_i1_depth_three_rrr_allows_only_trivial_stabilizer():
    """[rot90, rot90, rot90] on 1 shape: net = r³ (word length 3),
    depth = 3. Only trivial {e} works — any non-trivial H either
    contains r, r², r³, or some non-rotation which would shorten.

    Verify by cross-checking against the lookup table.
    """
    program = [_ROT, _ROT, _ROT]
    fs = _fs(program, 1)
    c = apply_rule_i1(program, fs, InputConstraint.empty(1))
    allowed = c.shapes[0].allowed_stabilizers
    assert allowed == stabilizers_allowed_at_depth(D4(3, False), 3)


def test_i1_untouched_shape_not_restricted():
    """by_index(1) rot90 on 2-shape scene: shape 0 touched, shape 1 not.

    Shape 1's stabilizer remains the full set of 10 subgroups.
    """
    program = [_by_idx(1, "rot90")]
    fs = _fs(program, 2)
    c = apply_rule_i1(program, fs, InputConstraint.empty(2))
    assert c.shapes[1].allowed_stabilizers == frozenset(D4_SUBGROUPS)


def test_i1_accumulates_across_two_shapes():
    """Each shape gets its own per-shape I1 restriction."""
    program = [_by_idx(1, "rot90"), _by_idx(2, "flip_h")]
    fs = _fs(program, 2)
    c = apply_rule_i1(program, fs, InputConstraint.empty(2))
    r = D4(1, False)
    s = D4(0, True)
    for H in D4_SUBGROUPS:
        assert (H in c.shapes[0].allowed_stabilizers) == (r not in H)
        assert (H in c.shapes[1].allowed_stabilizers) == (s not in H)


def test_i1_preserves_prior_restrictions():
    """If caller pre-restricted a shape, I1 intersects rather than
    overwrites."""
    program = [_ROT]
    fs = _fs(program, 1)
    trivial = frozenset({D4_IDENTITY})
    c0 = InputConstraint.empty(1).restrict_stabilizers(0, {trivial})
    c1 = apply_rule_i1(program, fs, c0)
    assert c1.shapes[0].allowed_stabilizers == {trivial}  # unchanged


def test_i1_zero_d4_steps_no_constraint():
    """Empty program → no shape touched → stabilizer unconstrained."""
    fs = FilterState.initial(1)
    c = apply_rule_i1([], fs, InputConstraint.empty(1))
    assert c.shapes[0].allowed_stabilizers == frozenset(D4_SUBGROUPS)


# === Accumulator ==========================================================


def test_accumulate_constraint_matches_i1_under_d4_vocab():
    """Under D4-only vocab, accumulate_constraint keeps I1 restrictions."""
    program = [_ROT, _FLIP_H]
    fs = _fs(program, 1)
    c_acc = accumulate_constraint(program, fs)
    c_i1 = apply_rule_i1(program, fs, InputConstraint.empty(1))
    assert c_acc.shapes == c_i1.shapes
    assert c_acc.row_forbids == c_i1.row_forbids
    assert c_acc.col_forbids == c_i1.col_forbids
    assert c_acc.primary_forbids == c_i1.primary_forbids
    assert c_acc.linear_forbids == c_i1.linear_forbids
    assert c_acc.box_bounds


def test_accumulate_constraint_shape_count_matches_fs():
    fs = FilterState.initial(3)
    c = accumulate_constraint([], fs)
    assert c.n_shapes == 3
    assert len(c.shapes) == 3


def test_vocab_requirements_apply_structural_preconditions_globally():
    c = apply_vocab_requirements(
        frozenset({"rot90", "translate_up", "add_border"}),
        InputConstraint.empty(2),
    )

    assert all(s.require_non_square for s in c.shapes)
    assert all(s.min_bbox_dim == 3 for s in c.shapes)


def test_vocab_requirements_do_not_fire_without_risky_vocab_pairs():
    c = apply_vocab_requirements(
        frozenset({"rot90", "add_border"}),
        InputConstraint.empty(2),
    )

    assert not any(s.require_non_square for s in c.shapes)
    assert all(s.min_bbox_dim == 1 for s in c.shapes)


def test_execution_safety_bounds_only_bbox_mutating_resolutions():
    program = [
        _by_idx(1, "translate_up"),
        _by_idx(2, "add_border"),
        _by_idx(1, "rot90"),
    ]
    fs = _fs(program, 2)
    c = apply_execution_safety_constraints(
        program, fs, InputConstraint.empty(2),
    )

    assert len(c.box_bounds) == 2
    assert {bound.shape_idx for bound in c.box_bounds} == {0, 1}


def test_execution_safety_leaves_movement_to_engine():
    program = [_by_idx(1, "translate_up")]
    fs = _fs(program, 1)
    c = apply_execution_safety_constraints(
        program, fs, InputConstraint.empty(1),
    )

    assert not c.box_bounds


def test_swap_then_next_color_forbids_shortcut_color_coincidence():
    program = [
        Step(
            selector=Selector("by_index", 2),
            primitive="swap_colors",
            target=Selector("by_index", 1),
        ),
        _by_idx(1, "next_color"),
    ]
    fs = _fs(program, 2)
    c = apply_swap_next_color_forbids(
        program, fs, InputConstraint.empty(2),
    )

    assert c.primary_forbids == {
        PrimaryColorCoincidenceForbid(1, 0, 1),
    }


def test_next_color_then_swap_forbids_equal_initial_colors():
    program = [
        _by_idx(2, "next_color"),
        Step(
            selector=Selector("by_index", 1),
            primitive="swap_colors",
            target=Selector("by_index", 2),
        ),
    ]
    fs = _fs(program, 2)
    c = apply_swap_next_color_forbids(
        program, fs, InputConstraint.empty(2),
    )

    assert c.primary_forbids == {
        PrimaryColorCoincidenceForbid(0, 1, 0),
    }


def test_align_translate_coincidence_forbid_records_size_dependent_shortcut():
    """align_row target at r_1 - h_1 must not equal source translate_up."""
    program = [
        _by_idx(2, "translate_up"),
        _by_idx(1, "hollow"),
        Step(
            selector=Selector("by_index", 1),
            primitive="align_row",
            target=Selector("by_index", 2),
        ),
    ]
    fs = _fs(program, 2)
    c = apply_align_coincidence_forbids(
        program, fs, InputConstraint.empty(2),
        vocab=frozenset({"translate_up", "hollow", "align_row"}),
    )
    expected_left = LinExpr.var("r_0") - LinExpr.var("h_0")
    expected_right = LinExpr.var("r_1") - LinExpr.var("h_1")
    assert any(
        f.left == expected_left and f.right == expected_right
        for f in c.linear_forbids
    )


def test_align_translate_coincidence_forbid_records_opposite_translate_shortcut():
    """align_col followed by left can collapse to initial right on a scene."""
    program = [
        Step(
            selector=Selector("by_index", 2),
            primitive="align_col",
            target=Selector("by_index", 1),
        ),
        _by_idx(2, "translate_left"),
    ]
    fs = _fs(program, 2)
    c = apply_align_coincidence_forbids(
        program, fs, InputConstraint.empty(2),
        vocab=frozenset({"translate_left", "translate_right", "align_col"}),
    )
    expected_left = LinExpr.var("c_1") + LinExpr.var("w_1")
    expected_right = LinExpr.var("c_0") - LinExpr.var("w_1")
    assert any(
        f.left == expected_left and f.right == expected_right
        for f in c.linear_forbids
    )


def test_translate_align_coincidence_forbid_records_col_alignment_shortcut():
    """translate_left can collapse to align_col when it lands on target col."""
    program = [
        _by_idx(1, "translate_left"),
    ]
    fs = _fs(program, 2)
    c = apply_translate_align_coincidence_forbids(
        program, fs, InputConstraint.empty(2),
        vocab=frozenset({"translate_left", "align_col"}),
    )
    expected_left = LinExpr.var("c_0") - LinExpr.var("w_0")
    expected_right = LinExpr.var("c_1")
    assert any(
        f.left == expected_left and f.right == expected_right
        for f in c.linear_forbids
    )


def test_translate_align_coincidence_forbid_records_row_alignment_shortcut():
    """translate_down can collapse to align_row when it lands on target row."""
    program = [
        _by_idx(2, "translate_down"),
    ]
    fs = _fs(program, 2)
    c = apply_translate_align_coincidence_forbids(
        program, fs, InputConstraint.empty(2),
        vocab=frozenset({"translate_down", "align_row"}),
    )
    expected_left = LinExpr.var("r_1") + LinExpr.var("h_1")
    expected_right = LinExpr.var("r_0")
    assert any(
        f.left == expected_left and f.right == expected_right
        for f in c.linear_forbids
    )


# === Sanity ===============================================================


def test_constraint_feasible_for_depth_1_rot():
    """Depth 1 rot90 leaves 5 allowed stabilizers (all H not containing r)."""
    program = [_ROT]
    fs = _fs(program, 1)
    c = accumulate_constraint(program, fs)
    assert not c.is_infeasible()
    # There should be exactly 6 subgroups not containing r: trivial,
    # {e,s}, {e,rs}, {e,r²s}, {e,r³s}, Klein_hv={e,r²,s,r²s}, Klein_dg={e,r²,rs,r³s}
    # …hmm, Klein groups contain r² (not r). Actually which of the 10
    # subgroups contain r = (1, False)?
    r = D4(1, False)
    expected = frozenset(H for H in D4_SUBGROUPS if r not in H)
    assert c.shapes[0].allowed_stabilizers == expected


# === by_color commitments (T1-1β) =========================================


def test_by_color_commit_records_required_primary():
    """by_color(c=3) committed to {0} at step 0 (offset=0) should pin
    shape 0 to primary=3 and forbid 3 on shape 1."""
    step = Step(selector=Selector("by_color", 3), primitive="rot90")
    fs = FilterState.initial(2)
    fs = fs.apply(step, override_indices=[0])
    c = apply_by_color_commitments(
        [step], fs, InputConstraint.empty(2),
        palette=(1, 2, 3, 4, 5),
    )
    assert c.shapes[0].required_primary_color == 3
    assert 3 in c.shapes[1].forbidden_primary_colors


def test_by_color_commit_next_color_shifts_target():
    """After one next_color on shape 0, by_color(3) must pin primary to
    (3 - 1) mod 9 = 2."""
    steps = [
        Step(selector=Selector("by_index", 1), primitive="next_color"),
        Step(selector=Selector("by_color", 3), primitive="rot90"),
    ]
    fs = FilterState.initial(2)
    fs = fs.apply(steps[0])
    fs = fs.apply(steps[1], override_indices=[0])
    c = apply_by_color_commitments(
        steps, fs, InputConstraint.empty(2),
        palette=(1, 2, 3, 4, 5),
    )
    assert c.shapes[0].required_primary_color == 2


def test_by_color_commit_conflict_is_infeasible():
    """Two by_color commits pinning shape 0 to incompatible primaries
    raise InfeasibleConstraint."""
    steps = [
        Step(selector=Selector("by_color", 1), primitive="rot90"),
        Step(selector=Selector("by_color", 2), primitive="flip_h"),
    ]
    fs = FilterState.initial(2)
    fs = fs.apply(steps[0], override_indices=[0])
    fs = fs.apply(steps[1], override_indices=[0])
    with pytest.raises(InfeasibleConstraint):
        apply_by_color_commitments(
            steps, fs, InputConstraint.empty(2),
            palette=(1, 2, 3, 4, 5),
        )


def test_by_color_commit_target_outside_palette_infeasible():
    """by_color(c=7) committed to a shape with offset=0 wants primary=7,
    outside palette (1..5) → infeasible."""
    step = Step(selector=Selector("by_color", 7), primitive="rot90")
    fs = FilterState.initial(2)
    fs = fs.apply(step, override_indices=[0])
    with pytest.raises(InfeasibleConstraint):
        apply_by_color_commitments(
            [step], fs, InputConstraint.empty(2),
            palette=(1, 2, 3, 4, 5),
        )


# === Positional commitments (T1-1γ) ========================================


def test_positional_topmost_commit_emits_lt_orderings():
    """topmost committed to shape 0 on n=3 must produce
    ``pos[0][row] < pos[j][row]`` for j ∈ {1, 2}."""
    from babytheorist.filter.rules_input import apply_positional_commitments
    step = Step(selector=Selector("topmost"), primitive="rot90")
    fs = FilterState.initial(3)
    fs = fs.apply(step, override_indices=[0])
    c = apply_positional_commitments([step], fs, InputConstraint.empty(3))
    from babytheorist.filter.constraint import PositionalOrdering
    assert c.positional_orderings == {
        PositionalOrdering(lt=0, gt=1, axis="row"),
        PositionalOrdering(lt=0, gt=2, axis="row"),
    }


def test_positional_bottommost_commit_emits_gt_orderings():
    from babytheorist.filter.rules_input import apply_positional_commitments
    from babytheorist.filter.constraint import PositionalOrdering
    step = Step(selector=Selector("bottommost"), primitive="rot90")
    fs = FilterState.initial(2)
    fs = fs.apply(step, override_indices=[1])
    c = apply_positional_commitments([step], fs, InputConstraint.empty(2))
    assert c.positional_orderings == {
        PositionalOrdering(lt=0, gt=1, axis="row"),
    }


def test_positional_leftmost_commit_emits_col_orderings():
    from babytheorist.filter.rules_input import apply_positional_commitments
    from babytheorist.filter.constraint import PositionalOrdering
    step = Step(selector=Selector("leftmost"), primitive="rot90")
    fs = FilterState.initial(2)
    fs = fs.apply(step, override_indices=[0])
    c = apply_positional_commitments([step], fs, InputConstraint.empty(2))
    assert c.positional_orderings == {
        PositionalOrdering(lt=0, gt=1, axis="col"),
    }


def test_positional_cycle_detection_infeasible():
    """``a < b`` then ``b < a`` on same axis must raise."""
    c = InputConstraint.empty(2)
    c = c.add_positional_ordering(lt=0, gt=1, axis="row")
    with pytest.raises(InfeasibleConstraint):
        c.add_positional_ordering(lt=1, gt=0, axis="row")


def test_positional_multi_hop_cycle_infeasible():
    """``0 < 1``, ``1 < 2``, adding ``2 < 0`` closes a 3-cycle."""
    c = InputConstraint.empty(3)
    c = c.add_positional_ordering(lt=0, gt=1, axis="row")
    c = c.add_positional_ordering(lt=1, gt=2, axis="row")
    with pytest.raises(InfeasibleConstraint):
        c.add_positional_ordering(lt=2, gt=0, axis="row")


def test_positional_different_axes_do_not_conflict():
    """Row ordering and col ordering are independent."""
    c = InputConstraint.empty(2)
    c = c.add_positional_ordering(lt=0, gt=1, axis="row")
    # Same pair but on col axis — must succeed.
    c = c.add_positional_ordering(lt=1, gt=0, axis="col")
    assert len(c.positional_orderings) == 2
