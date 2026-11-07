"""Tests for babytheorist.filter.constraint."""

from __future__ import annotations

import pytest

from babytheorist.symbolic import D4, D4_IDENTITY, LinExpr
from babytheorist.filter.constraint import (
    ColCoincidenceForbid,
    InfeasibleConstraint,
    InputConstraint,
    LinearCoincidenceForbid,
    PrimaryColorCoincidenceForbid,
    RowCoincidenceForbid,
    ShapeConstraint,
)
from babytheorist.filter.d4_tables import D4_SUBGROUPS


# === ShapeConstraint ======================================================


def test_default_shape_allows_all_subgroups():
    s = ShapeConstraint()
    assert s.allowed_stabilizers == frozenset(D4_SUBGROUPS)
    assert s.forbidden_primary_colors == frozenset()
    assert s.required_grid_shape is None
    assert not s.is_infeasible()


def test_restrict_stabilizers_shrinks():
    s = ShapeConstraint()
    trivial = frozenset({D4_IDENTITY})
    s2 = s.restrict_stabilizers({trivial})
    assert s2.allowed_stabilizers == {trivial}
    assert not s2.is_infeasible()


def test_restrict_stabilizers_idempotent():
    s = ShapeConstraint()
    trivial = frozenset({D4_IDENTITY})
    s2 = s.restrict_stabilizers({trivial})
    s3 = s2.restrict_stabilizers({trivial})
    assert s2 == s3


def test_restrict_stabilizers_empty_is_infeasible():
    s = ShapeConstraint()
    s2 = s.restrict_stabilizers([])  # empty → intersection empty
    assert s2.is_infeasible()


def test_forbid_primary_accumulates():
    s = ShapeConstraint()
    s2 = s.forbid_primary(3)
    s3 = s2.forbid_primary(5)
    assert s3.forbidden_primary_colors == {3, 5}


def test_require_grid_shape_then_same_is_ok():
    s = ShapeConstraint().require_grid_shape("hollow")
    s2 = s.require_grid_shape("hollow")
    assert s2.required_grid_shape == "hollow"


def test_require_grid_shape_conflict_raises():
    s = ShapeConstraint().require_grid_shape("hollow")
    with pytest.raises(InfeasibleConstraint):
        s.require_grid_shape("solid")


# === InputConstraint ======================================================


def test_empty_constructs_with_n_shapes():
    c = InputConstraint.empty(2)
    assert c.n_shapes == 2
    assert len(c.shapes) == 2
    assert c.row_forbids == frozenset()
    assert not c.is_infeasible()


def test_restrict_stabilizers_on_single_shape():
    c = InputConstraint.empty(2)
    trivial = frozenset({D4_IDENTITY})
    c2 = c.restrict_stabilizers(0, {trivial})
    assert c2.shapes[0].allowed_stabilizers == {trivial}
    assert c2.shapes[1].allowed_stabilizers == frozenset(D4_SUBGROUPS)


def test_add_row_forbid():
    c = InputConstraint.empty(2)
    c2 = c.add_row_forbid(0, 1, offset=3)
    assert c2.row_forbids == {RowCoincidenceForbid(0, 1, 3)}


def test_add_col_forbid():
    c = InputConstraint.empty(2)
    c2 = c.add_col_forbid(0, 1, offset=-2)
    assert c2.col_forbids == {ColCoincidenceForbid(0, 1, -2)}


def test_add_primary_forbid_takes_offset_mod_9():
    c = InputConstraint.empty(2)
    c2 = c.add_primary_forbid(0, 1, offset=11)
    assert c2.primary_forbids == {PrimaryColorCoincidenceForbid(0, 1, 2)}


def test_add_linear_forbid():
    c = InputConstraint.empty(2)
    left = LinExpr.var("r_0") - LinExpr.var("h_0")
    right = LinExpr.var("r_1") - LinExpr.var("h_1")
    c2 = c.add_linear_forbid(left, right)
    assert c2.linear_forbids == {LinearCoincidenceForbid(left, right)}


def test_add_linear_forbid_identical_expr_infeasible():
    c = InputConstraint.empty(2)
    expr = LinExpr.var("r_0")
    with pytest.raises(InfeasibleConstraint):
        c.add_linear_forbid(expr, expr)


def test_add_forbid_duplicate_no_op():
    c = InputConstraint.empty(2)
    c2 = c.add_row_forbid(0, 1, offset=3)
    c3 = c2.add_row_forbid(0, 1, offset=3)
    assert c3.row_forbids == c2.row_forbids


def test_self_coincidence_rejected():
    c = InputConstraint.empty(2)
    with pytest.raises(ValueError):
        c.add_row_forbid(0, 0, offset=0)
    with pytest.raises(ValueError):
        c.add_col_forbid(1, 1, offset=0)
    with pytest.raises(ValueError):
        c.add_primary_forbid(0, 0, offset=1)


def test_index_out_of_range():
    c = InputConstraint.empty(2)
    with pytest.raises(IndexError):
        c.restrict_stabilizers(5, frozenset())
    with pytest.raises(IndexError):
        c.add_row_forbid(0, 2, 0)


def test_immutable_update():
    """Mutator returns new; original unchanged."""
    c = InputConstraint.empty(2)
    c2 = c.forbid_primary(0, 3)
    assert 3 in c2.shapes[0].forbidden_primary_colors
    assert 3 not in c.shapes[0].forbidden_primary_colors


def test_is_infeasible_propagates_from_shape():
    c = InputConstraint.empty(2)
    c2 = c.restrict_stabilizers(0, [])
    assert c2.is_infeasible()


def test_is_infeasible_ignores_cross_shape():
    """Row/col/color forbids alone never flag infeasibility —
    scene sampler handles those."""
    c = InputConstraint.empty(2).add_row_forbid(0, 1, 5).add_primary_forbid(0, 1, 0)
    assert not c.is_infeasible()
