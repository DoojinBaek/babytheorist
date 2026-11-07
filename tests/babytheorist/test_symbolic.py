"""Tests for babytheorist.symbolic — symbolic state + per-primitive transitions."""

import sys; sys.path.insert(0, ".")
import pytest

from babytheorist.core import ALL, Selector, Step
from babytheorist.symbolic import (
    ColorVal,
    D4,
    D4_IDENTITY,
    LinExpr,
    PreconditionFail,
    apply_symbolic_program,
    apply_symbolic_step,
    current_height,
    current_width,
    initial_state,
    initial_states,
    resolve_symbolic,
)


# ---------------------------------------------------------------------------
# LinExpr
# ---------------------------------------------------------------------------


def test_linexpr_canonical_equality():
    a = LinExpr.from_dict({"x": 1, "y": 2}, const=3)
    b = LinExpr.from_dict({"y": 2, "x": 1}, const=3)
    assert a == b


def test_linexpr_zero_coeff_dropped():
    a = LinExpr.from_dict({"x": 1, "y": 0})
    b = LinExpr.from_dict({"x": 1})
    assert a == b


def test_linexpr_arithmetic():
    x = LinExpr.var("x")
    y = LinExpr.var("y")
    assert x + y - x == y
    assert x + x == LinExpr.from_dict({"x": 2})
    assert -x == LinExpr.from_dict({"x": -1})
    assert x - x == LinExpr.constant(0)


# ---------------------------------------------------------------------------
# D4
# ---------------------------------------------------------------------------


def test_d4_rot_four_is_identity():
    g = D4_IDENTITY
    for _ in range(4):
        g = g.apply_rot()
    assert g == D4_IDENTITY


def test_d4_flip_h_involution():
    assert D4_IDENTITY.apply_flip_h().apply_flip_h() == D4_IDENTITY


def test_d4_flip_v_involution():
    assert D4_IDENTITY.apply_flip_v().apply_flip_v() == D4_IDENTITY


def test_d4_flip_h_flip_v_equals_r_squared():
    """flip_h then flip_v = rot180."""
    g = D4_IDENTITY.apply_flip_h().apply_flip_v()
    assert g == D4(2, False)


def test_d4_rot_flip_rot_flip_is_identity():
    """(rot · flip)^2 = e — a canonical D4 composite identity."""
    g = D4_IDENTITY
    for _ in range(2):
        g = g.apply_rot().apply_flip_h()
    assert g == D4_IDENTITY


def test_d4_srs_equals_r_inverse():
    """The D4 relation s·r·s = r^{-1} = r^3."""
    g = D4_IDENTITY.apply_flip_h().apply_rot().apply_flip_h()
    assert g == D4(3, False)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_fields():
    s = initial_state(0)
    assert s.initial_h_var == "h_0"
    assert s.initial_w_var == "w_0"
    assert s.orientation == D4_IDENTITY
    assert s.border_count == 0
    assert s.row == LinExpr.var("r_0")
    assert s.col == LinExpr.var("c_0")
    assert s.color == ColorVal("0", 0)
    assert s.grid_shape == "solid"


def test_initial_states_distinct_per_shape():
    states = initial_states(3)
    assert states[0].row != states[1].row
    assert states[0].color != states[1].color


# ---------------------------------------------------------------------------
# Selector resolution (static only)
# ---------------------------------------------------------------------------


def test_resolve_symbolic_all():
    states = initial_states(3)
    assert resolve_symbolic(ALL, states) == [0, 1, 2]


def test_resolve_symbolic_by_index():
    states = initial_states(3)
    assert resolve_symbolic(Selector("by_index", 2), states) == [1]
    assert resolve_symbolic(Selector("by_index", 10), states) == []


def test_resolve_symbolic_dynamic_raises():
    states = initial_states(2)
    with pytest.raises(NotImplementedError):
        resolve_symbolic(Selector("topmost"), states)
    with pytest.raises(NotImplementedError):
        resolve_symbolic(Selector("by_color", param=3), states)


def test_resolve_symbolic_is_hollow_initial_all_solid():
    """All shapes initial grid_shape = 'solid' → is_hollow resolves to []."""
    states = initial_states(3)
    assert resolve_symbolic(Selector("is_hollow"), states) == []


def test_resolve_symbolic_is_hollow_initial_mixed():
    states = initial_states(3, grid_shapes=["solid", "hollow", "solid"])
    assert resolve_symbolic(Selector("is_hollow"), states) == [1]


def test_resolve_symbolic_is_hollow_after_hollow_all():
    """After hollow(ALL), is_hollow selects every shape."""
    from babytheorist.core import ALL, Step
    from babytheorist.symbolic import apply_symbolic_program
    traj = apply_symbolic_program([Step(ALL, "hollow")], n_shapes=3)
    assert resolve_symbolic(Selector("is_hollow"), traj[-1]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Per-primitive transitions — transform
# ---------------------------------------------------------------------------


def test_rot90_updates_orientation():
    s = initial_state(0)
    out = apply_symbolic_step(Step(ALL, "rot90"), [s])[0]
    assert out.orientation == D4(1, False)


def test_flip_h_updates_orientation():
    s = initial_state(0)
    out = apply_symbolic_step(Step(ALL, "flip_h"), [s])[0]
    assert out.orientation == D4(0, True)


def test_next_color_increments_offset():
    s = initial_state(0)
    out = apply_symbolic_step(Step(ALL, "next_color"), [s])[0]
    assert out.color == ColorVal("0", 1)


def test_next_color_wraps_mod_9():
    states = [initial_state(0)]
    for _ in range(9):
        states = apply_symbolic_step(Step(ALL, "next_color"), states)
    assert states[0].color == ColorVal("0", 0)


def test_hollow_sets_grid_shape():
    s = initial_state(0)
    out = apply_symbolic_step(Step(ALL, "hollow"), [s])[0]
    assert out.grid_shape == "hollow"


def test_fill_interior_requires_hollow():
    s = initial_state(0)  # solid
    with pytest.raises(PreconditionFail):
        apply_symbolic_step(Step(ALL, "fill_interior"), [s])


def test_hollow_then_fill_interior_returns_solid():
    states = [initial_state(0)]
    states = apply_symbolic_step(Step(ALL, "hollow"), states)
    states = apply_symbolic_step(Step(ALL, "fill_interior"), states)
    assert states[0].grid_shape == "solid"


def test_add_border_shifts_position_and_grows_size():
    s = initial_state(0)
    out = apply_symbolic_step(Step(ALL, "add_border"), [s])[0]
    assert out.border_count == 1
    assert out.row == LinExpr.var("r_0") - LinExpr.constant(1)
    assert out.col == LinExpr.var("c_0") - LinExpr.constant(1)
    # height = h_0 + 2
    assert current_height(out) == LinExpr.from_dict({"h_0": 1}, const=2)
    assert current_width(out) == LinExpr.from_dict({"w_0": 1}, const=2)


# ---------------------------------------------------------------------------
# Per-primitive transitions — movement
# ---------------------------------------------------------------------------


def test_translate_up_subtracts_height():
    states = [initial_state(0)]
    states = apply_symbolic_step(Step(ALL, "translate_up"), states)
    # row = r_0 - h_0
    assert states[0].row == LinExpr.from_dict({"r_0": 1, "h_0": -1})


def test_translate_up_twice():
    states = [initial_state(0)]
    for _ in range(2):
        states = apply_symbolic_step(Step(ALL, "translate_up"), states)
    # row = r_0 - 2*h_0
    assert states[0].row == LinExpr.from_dict({"r_0": 1, "h_0": -2})


def test_translate_up_down_cancels():
    states = [initial_state(0)]
    states = apply_symbolic_step(Step(ALL, "translate_up"), states)
    states = apply_symbolic_step(Step(ALL, "translate_down"), states)
    assert states[0].row == LinExpr.var("r_0")


def test_translate_under_rotation_uses_swapped_size():
    """After rot90, the shape's effective height is the initial width."""
    states = [initial_state(0)]
    states = apply_symbolic_step(Step(ALL, "rot90"), states)
    states = apply_symbolic_step(Step(ALL, "translate_up"), states)
    # row = r_0 - w_0 (not h_0)
    assert states[0].row == LinExpr.from_dict({"r_0": 1, "w_0": -1})


def test_translate_under_border_adds_two():
    states = [initial_state(0)]
    states = apply_symbolic_step(Step(ALL, "add_border"), states)
    states = apply_symbolic_step(Step(ALL, "translate_up"), states)
    # row after add_border: r_0 - 1
    # height after add_border: h_0 + 2
    # row after translate_up: r_0 - 1 - (h_0 + 2) = r_0 - h_0 - 3
    assert states[0].row == LinExpr.from_dict({"r_0": 1, "h_0": -1}, const=-3)


# ---------------------------------------------------------------------------
# Per-primitive transitions — relational
# ---------------------------------------------------------------------------


def _two_shapes_by_index():
    """Helper: two-shape scene selectable via by_index(1) and by_index(2)."""
    return initial_states(2), Selector("by_index", 1), Selector("by_index", 2)


def test_swap_colors_swaps():
    states, sel_a, sel_b = _two_shapes_by_index()
    out = apply_symbolic_step(
        Step(sel_a, "swap_colors", target=sel_b), states
    )
    assert out[0].color == ColorVal("1", 0)
    assert out[1].color == ColorVal("0", 0)


def test_swap_colors_twice_restores():
    states, sel_a, sel_b = _two_shapes_by_index()
    step = Step(sel_a, "swap_colors", target=sel_b)
    states = apply_symbolic_step(step, states)
    states = apply_symbolic_step(step, states)
    assert states[0].color == ColorVal("0", 0)
    assert states[1].color == ColorVal("1", 0)


def test_align_row_copies_target_row():
    states, sel_a, sel_b = _two_shapes_by_index()
    out = apply_symbolic_step(
        Step(sel_a, "align_row", target=sel_b), states
    )
    assert out[0].row == LinExpr.var("r_1")
    assert out[1].row == LinExpr.var("r_1")


def test_align_row_on_equal_rows_raises():
    """Symbolic precondition: if rows are already (symbolically) equal, fail."""
    states, sel_a, sel_b = _two_shapes_by_index()
    # Force A.row == B.row by aligning first; second align should fail.
    states = apply_symbolic_step(
        Step(sel_a, "align_row", target=sel_b), states
    )
    with pytest.raises(PreconditionFail):
        apply_symbolic_step(
            Step(sel_a, "align_row", target=sel_b), states
        )


def test_relational_requires_distinct_indices():
    states = initial_states(2)
    sel_a = Selector("by_index", 1)
    with pytest.raises(PreconditionFail):
        apply_symbolic_step(
            Step(sel_a, "swap_colors", target=sel_a), states
        )


# ---------------------------------------------------------------------------
# apply_symbolic_program trajectory
# ---------------------------------------------------------------------------


def test_apply_symbolic_program_returns_full_trajectory():
    program = [Step(ALL, "rot90"), Step(ALL, "rot90")]
    traj = apply_symbolic_program(program, n_shapes=1)
    assert len(traj) == 3
    assert traj[0][0].orientation == D4_IDENTITY
    assert traj[1][0].orientation == D4(1, False)
    assert traj[2][0].orientation == D4(2, False)


def test_apply_symbolic_program_rot_x4_returns_to_identity():
    program = [Step(ALL, "rot90")] * 4
    traj = apply_symbolic_program(program, n_shapes=1)
    assert traj[0][0].orientation == traj[4][0].orientation == D4_IDENTITY
