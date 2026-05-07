"""Tests for babytheorist.filter.state.FilterState."""

from __future__ import annotations

import pytest

from babytheorist.core import ALL, Selector, Step
from babytheorist.filter.state import FilterState, PreconditionFail
from babytheorist.symbolic import D4, D4_IDENTITY


# === Initial ==============================================================


def test_initial_one_shape():
    fs = FilterState.initial(1)
    assert fs.n_shapes == 1
    assert fs.n_steps_applied == 0
    assert len(fs.history) == 1
    # Initial orient is identity.
    assert fs.current[0].orientation == D4_IDENTITY


def test_initial_two_shapes_distinct_states():
    fs = FilterState.initial(2)
    assert fs.n_shapes == 2
    # Shapes share orient but have distinct row/col vars.
    assert fs.current[0].orientation == fs.current[1].orientation
    assert fs.current[0].row != fs.current[1].row
    assert fs.current[0].col != fs.current[1].col


def test_initial_grid_shapes():
    fs = FilterState.initial(2, grid_shapes=["hollow", "solid"])
    assert fs.current[0].grid_shape == "hollow"
    assert fs.current[1].grid_shape == "solid"


# === Apply ================================================================


def _step_rot90_all() -> Step:
    return Step(selector=ALL, primitive="rot90")


def test_apply_rot90_updates_orient():
    fs = FilterState.initial(1)
    fs1 = fs.apply(_step_rot90_all())
    assert fs1.current[0].orientation == D4(1, False)
    assert fs1.n_steps_applied == 1
    assert len(fs1.history) == 2


def test_apply_is_immutable():
    """Applying does not mutate the original state."""
    fs = FilterState.initial(1)
    _ = fs.apply(_step_rot90_all())
    assert fs.current[0].orientation == D4_IDENTITY
    assert fs.n_steps_applied == 0


def test_apply_chain_four_rot90_cycles_to_identity():
    fs = FilterState.initial(1)
    for _ in range(4):
        fs = fs.apply(_step_rot90_all())
    assert fs.current[0].orientation == D4_IDENTITY
    assert fs.n_steps_applied == 4
    assert len(fs.history) == 5


def test_apply_by_index_targets_single_shape():
    fs = FilterState.initial(2)
    step = Step(selector=Selector("by_index", 1), primitive="rot90")
    fs1 = fs.apply(step)
    assert fs1.current[0].orientation == D4(1, False)
    assert fs1.current[1].orientation == D4_IDENTITY


def test_apply_precondition_fail_propagates():
    fs = FilterState.initial(1)
    # fill_interior on solid shape raises PreconditionFail.
    step = Step(selector=ALL, primitive="fill_interior")
    with pytest.raises(PreconditionFail):
        fs.apply(step)


# === History =============================================================


def test_history_preserves_past_snapshots():
    fs = FilterState.initial(1)
    fs1 = fs.apply(_step_rot90_all())
    fs2 = fs1.apply(_step_rot90_all())
    # history[0] is initial, history[1] after first rot, history[2] after second.
    assert fs2.history[0][0].orientation == D4_IDENTITY
    assert fs2.history[1][0].orientation == D4(1, False)
    assert fs2.history[2][0].orientation == D4(2, False)


def test_shape_trajectory_order():
    fs = FilterState.initial(1)
    for _ in range(3):
        fs = fs.apply(_step_rot90_all())
    traj = fs.shape_trajectory(0)
    assert len(traj) == 4  # initial + 3 applies
    orients = [s.orientation for s in traj]
    assert orients == [D4_IDENTITY, D4(1, False), D4(2, False), D4(3, False)]


def test_shape_state_at_out_of_range():
    fs = FilterState.initial(1)
    with pytest.raises(IndexError):
        fs.shape_state_at(5, 0)
    with pytest.raises(IndexError):
        fs.shape_state_at(0, 7)


def test_history_snapshot_is_hashable():
    """History entries must be hashable for R3 revisit via set lookup."""
    fs = FilterState.initial(1)
    for _ in range(3):
        fs = fs.apply(_step_rot90_all())
    # Per-shape state tuples are frozen-dataclass based → hashable.
    seen = {fs.history[i] for i in range(len(fs.history))}
    assert len(seen) == 4
