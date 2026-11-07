"""Tests for babytheorist.program — simplification + dominant-program sampling."""

import sys; sys.path.insert(0, ".")
import numpy as np
import pytest

from babytheorist.core import Step, Selector, ALL, by_color, topmost, bottommost
from babytheorist.program import (
    SamplerDeadEnd, build_bigram, build_pool,
    sample_bigram, sample_primitives,
    simplify, is_valid, effective_depth,
    trajectory_distinct, no_dead_writes,
)


def _idx(k: int) -> Selector:
    """V1 trajectory_distinct uses static selectors only — by_index(k)."""
    return Selector("by_index", k)


# ---------------------------------------------------------------------------
# Involutions
# ---------------------------------------------------------------------------


def test_involution_flip_h():
    prog = [Step(ALL, "flip_h"), Step(ALL, "flip_h")]
    assert simplify(prog) == []


def test_involution_flip_v():
    prog = [Step(ALL, "flip_v"), Step(ALL, "flip_v")]
    assert simplify(prog) == []


def test_involution_different_selectors():
    """Different selectors → no simplification even if same action."""
    prog = [Step(by_color(1), "flip_h"), Step(by_color(2), "flip_h")]
    assert simplify(prog) == prog


def test_swap_colors_involution():
    """swap_colors is an involution (relational action with target)."""
    prog = [
        Step(topmost(), "swap_colors", target=bottommost()),
        Step(topmost(), "swap_colors", target=bottommost()),
    ]
    assert simplify(prog) == []


def test_swap_colors_different_targets():
    """Different targets → no simplification."""
    s1 = Step(topmost(), "swap_colors", target=bottommost())
    s2 = Step(topmost(), "swap_colors", target=Selector("leftmost"))
    prog = [s1, s2]
    assert simplify(prog) == prog


# ---------------------------------------------------------------------------
# Idempotents
# ---------------------------------------------------------------------------


def test_idempotent_hollow():
    prog = [Step(ALL, "hollow"), Step(ALL, "hollow")]
    assert simplify(prog) == [Step(ALL, "hollow")]


def test_idempotent_fill_interior():
    prog = [Step(ALL, "fill_interior"), Step(ALL, "fill_interior")]
    assert simplify(prog) == [Step(ALL, "fill_interior")]


def test_idempotent_triple():
    """(A, A, A) → (A, A) → (A)."""
    prog = [Step(ALL, "hollow")] * 3
    assert simplify(prog) == [Step(ALL, "hollow")]


def test_idempotent_different_selectors():
    """Different selectors → no simplification."""
    prog = [Step(by_color(1), "hollow"), Step(by_color(2), "hollow")]
    assert simplify(prog) == prog


def test_idempotent_align_row():
    """align_row is relational + idempotent; same context → reduce."""
    prog = [
        Step(topmost(), "align_row", target=bottommost()),
        Step(topmost(), "align_row", target=bottommost()),
    ]
    assert simplify(prog) == [Step(topmost(), "align_row", target=bottommost())]


# ---------------------------------------------------------------------------
# Canceling pairs
# ---------------------------------------------------------------------------


def test_canceling_translate_up_down():
    """translate_up then translate_down → cancel both."""
    prog = [Step(ALL, "translate_up"), Step(ALL, "translate_down")]
    assert simplify(prog) == []


def test_canceling_translate_left_right():
    """translate_left then translate_right → cancel both."""
    prog = [Step(ALL, "translate_left"), Step(ALL, "translate_right")]
    assert simplify(prog) == []


def test_canceling_translate_different_selectors():
    """Different selectors → no simplification for canceling pairs."""
    prog = [Step(by_color(1), "translate_up"), Step(by_color(2), "translate_down")]
    assert simplify(prog) == prog


def test_canceling_cascade():
    """Canceling in the middle of a program should cascade.

    [hollow, translate_up, translate_down, hollow] →
    cancel translate pair → [hollow, hollow] →
    idempotent → [hollow]
    """
    prog = [
        Step(ALL, "hollow"),
        Step(ALL, "translate_up"),
        Step(ALL, "translate_down"),
        Step(ALL, "hollow"),
    ]
    assert simplify(prog) == [Step(ALL, "hollow")]


# ---------------------------------------------------------------------------
# Cyclic
# ---------------------------------------------------------------------------


def test_cyclic_rot90_4():
    prog = [Step(ALL, "rot90")] * 4
    assert simplify(prog) == []


def test_cyclic_rot90_5():
    prog = [Step(ALL, "rot90")] * 5
    assert simplify(prog) == [Step(ALL, "rot90")]


def test_cyclic_rot90_3():
    """3 rot90 stays as 3 (no reduction)."""
    prog = [Step(ALL, "rot90")] * 3
    assert simplify(prog) == [Step(ALL, "rot90")] * 3


def test_cyclic_next_color_9():
    prog = [Step(ALL, "next_color")] * 9
    assert simplify(prog) == []


def test_cyclic_next_color_10():
    prog = [Step(ALL, "next_color")] * 10
    assert simplify(prog) == [Step(ALL, "next_color")]


def test_cyclic_different_selectors():
    """Different selectors break the cyclic run."""
    # 2x rot90 with selector A, then 2x rot90 with selector B → no reduction
    prog = [Step(by_color(1), "rot90")] * 2 + [Step(by_color(2), "rot90")] * 2
    assert simplify(prog) == prog


# ---------------------------------------------------------------------------
# Mixed / Combined
# ---------------------------------------------------------------------------


def test_mixed_involution_and_idempotent():
    """Mix of involution + idempotent in same program."""
    prog = [
        Step(ALL, "flip_h"),
        Step(ALL, "flip_h"),
        Step(ALL, "hollow"),
        Step(ALL, "hollow"),
    ]
    assert simplify(prog) == [Step(ALL, "hollow")]


def test_cascading_reduction():
    """After involution removal, new adjacent pair may simplify."""
    # [hollow, flip_h, flip_h, hollow] → [hollow, hollow] → [hollow]
    prog = [
        Step(ALL, "hollow"),
        Step(ALL, "flip_h"),
        Step(ALL, "flip_h"),
        Step(ALL, "hollow"),
    ]
    assert simplify(prog) == [Step(ALL, "hollow")]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_program():
    assert simplify([]) == []
    assert effective_depth([]) == 0
    assert is_valid([]) is True


def test_single_step():
    prog = [Step(ALL, "flip_h")]
    assert simplify(prog) == prog
    assert effective_depth(prog) == 1
    assert is_valid(prog) is True


# ---------------------------------------------------------------------------
# effective_depth and is_valid
# ---------------------------------------------------------------------------


def test_effective_depth():
    assert effective_depth([Step(ALL, "flip_h"), Step(ALL, "flip_h")]) == 0
    assert effective_depth([Step(ALL, "rot90")] * 4) == 0
    assert effective_depth([Step(ALL, "flip_h"), Step(ALL, "rot90")]) == 2


def test_is_valid_true():
    """Irreducible programs are valid."""
    prog = [Step(ALL, "flip_h"), Step(ALL, "rot90")]
    assert is_valid(prog) is True


def test_is_valid_false():
    """Reducible programs are invalid."""
    assert is_valid([Step(ALL, "flip_h"), Step(ALL, "flip_h")]) is False
    assert is_valid([Step(ALL, "hollow"), Step(ALL, "hollow")]) is False
    assert is_valid([Step(ALL, "rot90")] * 4) is False


# ---------------------------------------------------------------------------
# build_bigram
# ---------------------------------------------------------------------------


def test_build_bigram_rejects_single_involution():
    """['flip_h'] masks its only cell to 0 — reject at construction."""
    with pytest.raises(ValueError, match="no positive transitions"):
        build_bigram(["flip_h"])


def test_public_api_rejects_duplicate_primitives():
    """Duplicate primitives desync `idx` and the matrix — every public entry
    point must reject them, not just `build_bigram` (so callers that skip the
    bigram builder don't silently get malformed state)."""
    dup = ["flip_h", "flip_h"]
    with pytest.raises(ValueError, match="duplicate primitives"):
        build_bigram(dup)
    with pytest.raises(ValueError, match="duplicate primitives"):
        build_pool(np.ones((2, 2)), dup, max_depth=2)
    with pytest.raises(ValueError, match="duplicate primitives"):
        sample_primitives(
            {1: {("flip_h",): 1.0}}, np.ones((2, 2)), dup,
            depth=1, rng=np.random.default_rng(0),
        )


def test_build_bigram_row_stochastic_with_zeros():
    prims = ["rot90", "flip_h", "translate_up", "translate_down"]
    mat = build_bigram(prims)
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat.sum(axis=1), 1.0)
    idx = {p: i for i, p in enumerate(prims)}
    assert mat[idx["flip_h"], idx["flip_h"]] == 0.0          # involution
    assert mat[idx["translate_up"], idx["translate_down"]] == 0.0  # canceling
    assert mat[idx["translate_down"], idx["translate_up"]] == 0.0  # canceling


# ---------------------------------------------------------------------------
# build_pool — DP score is the full bigram product along the sequence
# ---------------------------------------------------------------------------


def test_build_bigram_pair_weights_boost_specific_cells():
    """`bigram_weights` multiplies pre-mask cells; zeros stay zero."""
    prims = ["rot90", "flip_h", "translate_up", "translate_down"]
    base = build_bigram(prims)
    boosted = build_bigram(
        prims,
        bigram_weights={
            ("rot90", "flip_h"): 4.0,
            ("translate_up", "translate_down"): 99.0,  # canceled, must stay 0
        },
    )
    idx = {p: i for i, p in enumerate(prims)}
    # rot90 row should now favor flip_h (4× boost vs base, after re-normalization)
    assert boosted[idx["rot90"], idx["flip_h"]] > base[idx["rot90"], idx["flip_h"]]
    # canceling pair stays zero (mask runs first)
    assert boosted[idx["translate_up"], idx["translate_down"]] == 0.0
    np.testing.assert_allclose(boosted.sum(axis=1), 1.0)


def test_build_pool_primitive_weights_propagate():
    """Per-primitive base scores propagate multiplicatively via DP."""
    prims = ["rot90", "flip_h"]
    bigram = build_bigram(prims)
    pool = build_pool(bigram, prims, max_depth=3, primitive_weights={"rot90": 5.0})
    assert pool[1][("rot90",)] == 5.0
    assert pool[1][("flip_h",)] == 1.0
    # Score of (rot90, rot90, rot90) = 5 · T · 5 · T · 5 = 125 · T²
    # Score of (flip_h, flip_h, flip_h) = 1 · 0 · 1 · 0 · 1 = 0  (involution)
    # → rot90-only entries dominate at deeper levels
    if ("rot90", "rot90", "rot90") in pool[3]:
        assert pool[3][("rot90", "rot90", "rot90")] == pytest.approx(
            5.0 * bigram[0, 0] * 5.0 * bigram[0, 0] * 5.0
        )


def test_build_pool_score_is_bigram_product():
    """Any kept program's score equals ∏ T[p_i, p_{i+1}], regardless of which
    split produced it — so the "max over duplicates" logic is a no-op but must
    never under-count."""
    prims = ["rot90", "flip_h", "translate_up", "translate_down"]
    bigram = build_bigram(prims)
    pool = build_pool(bigram, prims, max_depth=5, pool_size=25)
    idx = {p: i for i, p in enumerate(prims)}
    for d in range(2, 6):
        for prog, score in pool[d].items():
            expected = 1.0
            for a, b in zip(prog, prog[1:]):
                expected *= bigram[idx[a], idx[b]]
            assert score == pytest.approx(expected), (
                f"pool[{d}][{prog}]: score={score}, expected={expected}"
            )


# ---------------------------------------------------------------------------
# sample_primitives — dead-end paths raise, length invariant holds
# ---------------------------------------------------------------------------


def test_sample_primitives_length_invariant():
    prims = ["rot90", "flip_h", "translate_up", "translate_down"]
    bigram = build_bigram(prims)
    pool = build_pool(bigram, prims, max_depth=4, pool_size=20)
    rng = np.random.default_rng(0)
    for d in [1, 2, 3, 4, 7]:
        for _ in range(10):
            seq = sample_primitives(pool, bigram, prims, d, rng)
            assert len(seq) == d
            assert all(p in prims for p in seq)


def test_sample_primitives_raises_on_empty_pool_level():
    """Empty pool[d] in the direct path → raise SamplerDeadEnd (retryable)."""
    prims = ["rot90", "flip_h"]
    bigram = build_bigram(prims)
    pool = {
        1: {("rot90",): 1.0},
        2: {},                               # empty intermediate level
        3: {("rot90", "rot90", "rot90"): 0.5},
    }
    rng = np.random.default_rng(0)
    with pytest.raises(SamplerDeadEnd, match=r"pool\[2\] is empty"):
        sample_primitives(pool, bigram, prims, depth=2, rng=rng, p_direct=1.0)


def test_sample_primitives_raises_on_zero_weight_continuation():
    """Walk path hits a zero-weight boundary → raise instead of silently
    falling back to uniform (past silent bug)."""
    prims = ["flip_h", "flip_v"]  # both involutions
    bigram = build_bigram(prims)
    # Only flip_h is reachable from pool[1]; second chunk needs
    # T[flip_h, flip_h] = 0 — the walk must raise rather than paper over it.
    pool = {1: {("flip_h",): 1.0}}
    rng = np.random.default_rng(0)
    with pytest.raises(SamplerDeadEnd, match="zero-weight continuation"):
        sample_primitives(pool, bigram, prims, depth=2, rng=rng, p_direct=0.0)


# ---------------------------------------------------------------------------
# sample_bigram — pool-free fallback used when use_dominant=False
# ---------------------------------------------------------------------------


def test_sample_bigram_length_and_vocab():
    prims = ["rot90", "flip_h", "translate_up", "translate_down"]
    bigram = build_bigram(prims)
    rng = np.random.default_rng(0)
    for d in [1, 2, 3, 7]:
        for _ in range(10):
            seq = sample_bigram(bigram, prims, d, rng)
            assert len(seq) == d
            assert all(p in prims for p in seq)


def test_sample_bigram_raises_on_zero_row():
    """Zeroed bigram row → `SamplerDeadEnd` (caught by the task retry loop)."""
    prims = ["rot90", "flip_h"]
    bigram = build_bigram(
        prims,
        bigram_weights={("rot90", "rot90"): 0.0, ("rot90", "flip_h"): 0.0},
    )
    # rot90 row is now [0, 0]; depth=3 must re-enter it from either start.
    rng = np.random.default_rng(0)
    with pytest.raises(SamplerDeadEnd, match="bigram row all-zero"):
        sample_bigram(bigram, prims, depth=3, rng=rng)


# ---------------------------------------------------------------------------
# generate_task exception filtering — dead-ends retry, other errors propagate
# ---------------------------------------------------------------------------


def test_generate_task_propagates_non_dead_end_errors(monkeypatch):
    """A `ValueError` from random_program that isn't a `SamplerDeadEnd` must
    NOT be swallowed by the retry loop, or real bugs disappear into generic
    `None` returns."""
    import babytheorist.generation.rejection as rejection_module
    from babytheorist.generation.config import LevelConfig, TaskConfig
    from babytheorist.shapes import ShapeFactory

    def boom(*args, **kwargs):
        raise ValueError("unrelated bug")

    monkeypatch.setattr(rejection_module, "random_program_filtered", boom)
    with pytest.raises(ValueError, match="unrelated bug"):
        rejection_module.generate_task(
            LevelConfig(level=1, primitives=["rot90"], fixed_depth=1),
            TaskConfig(max_program_retries=3),
            ShapeFactory(size_range=(4, 10)),
            {1: {("rot90",): 1.0}},
            np.array([[1.0]]),
            np.random.default_rng(0),
        )


def test_generate_task_retries_on_sampler_dead_end(monkeypatch):
    """A `SamplerDeadEnd` from random_program should be caught and retried,
    not propagated. Retry exhaustion returns `None` (the existing contract)."""
    import babytheorist.generation.rejection as rejection_module
    from babytheorist.generation.config import LevelConfig, TaskConfig
    from babytheorist.shapes import ShapeFactory

    calls = {"n": 0}

    def always_dead(*args, **kwargs):
        calls["n"] += 1
        raise SamplerDeadEnd("forced")

    monkeypatch.setattr(rejection_module, "random_program_filtered", always_dead)
    result = rejection_module.generate_task(
        LevelConfig(level=1, primitives=["rot90"], fixed_depth=1),
        TaskConfig(max_program_retries=5),
        ShapeFactory(size_range=(4, 10)),
        {1: {("rot90",): 1.0}},
        np.array([[1.0]]),
        np.random.default_rng(0),
    )
    assert result is None
    assert calls["n"] == 5


# ---------------------------------------------------------------------------
# trajectory_distinct — Phase 1 Rule 1
# ---------------------------------------------------------------------------


def test_trajectory_distinct_empty_program():
    assert trajectory_distinct([], n_shapes=1) is True


def test_trajectory_distinct_single_rot90():
    assert trajectory_distinct([Step(ALL, "rot90")], n_shapes=1) is True


def test_trajectory_distinct_rot90_three_times_ok():
    prog = [Step(ALL, "rot90")] * 3
    assert trajectory_distinct(prog, n_shapes=1) is True


def test_trajectory_distinct_rot90_four_times_rejected():
    prog = [Step(ALL, "rot90")] * 4
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_flip_involution_rejected():
    prog = [Step(ALL, "flip_h"), Step(ALL, "flip_h")]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_d4_composite_identity_rejected():
    """(rot90; flip_h)^2 = e — D4 composite identity."""
    prog = [
        Step(ALL, "rot90"), Step(ALL, "flip_h"),
        Step(ALL, "rot90"), Step(ALL, "flip_h"),
    ]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_flip_h_flip_v_is_rot180():
    """flip_h; flip_v = rot180 — not identity, so accept."""
    prog = [Step(ALL, "flip_h"), Step(ALL, "flip_v")]
    assert trajectory_distinct(prog, n_shapes=1) is True


def test_trajectory_distinct_translate_cancel_rejected():
    prog = [Step(ALL, "translate_up"), Step(ALL, "translate_down")]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_translate_twice_accepted():
    """Two translations in the same direction give distinct positions."""
    prog = [Step(ALL, "translate_up"), Step(ALL, "translate_up")]
    assert trajectory_distinct(prog, n_shapes=1) is True


def test_trajectory_distinct_hollow_fill_round_trip_rejected():
    prog = [Step(ALL, "hollow"), Step(ALL, "fill_interior")]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_fill_on_solid_fails_precondition():
    """fill_interior on solid is symbolically infeasible → reject."""
    prog = [Step(ALL, "fill_interior")]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_next_color_nine_times_rejected():
    prog = [Step(ALL, "next_color")] * 9
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_next_color_eight_times_accepted():
    prog = [Step(ALL, "next_color")] * 8
    assert trajectory_distinct(prog, n_shapes=1) is True


def test_trajectory_distinct_cross_shape_rot_hidden_cycle_rejected():
    """`rot(A); trans(B); rot(A); rot(A); rot(A)` — simplify misses this
    because rot's aren't adjacent. But A's own trajectory cycles back."""
    prog = [
        Step(_idx(1), "rot90"),
        Step(_idx(2), "translate_up"),
        Step(_idx(1), "rot90"),
        Step(_idx(1), "rot90"),
        Step(_idx(1), "rot90"),
    ]
    assert trajectory_distinct(prog, n_shapes=2) is False


def test_trajectory_distinct_cross_shape_no_cycle_accepted():
    """`rot(A); trans(B)` — different shapes, not redundant."""
    prog = [
        Step(_idx(1), "rot90"),
        Step(_idx(2), "translate_up"),
    ]
    assert trajectory_distinct(prog, n_shapes=2) is True


def test_trajectory_distinct_swap_twice_precondition_fails():
    """swap(A,B); swap(A,B) — second swap sees identical colors → reject."""
    prog = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "swap_colors", target=_idx(2)),
    ]
    assert trajectory_distinct(prog, n_shapes=2) is False


def test_trajectory_distinct_three_cycle_swap_rejected():
    """swap(A,B); swap(B,C); swap(A,C) — A cycles back to its initial color."""
    prog = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(2), "swap_colors", target=_idx(3)),
        Step(_idx(1), "swap_colors", target=_idx(3)),
    ]
    assert trajectory_distinct(prog, n_shapes=3) is False


def test_trajectory_distinct_align_chain_passes_rule1():
    """Rule 1 does NOT catch overwrite dead-code. align(A,B); align(A,C)
    has distinct row expressions (r_B, r_C), so Rule 1 accepts. Rule 2
    (not yet implemented) is responsible for catching this."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "align_row", target=_idx(3)),
    ]
    assert trajectory_distinct(prog, n_shapes=3) is True


def test_trajectory_distinct_align_self_loop_rejected():
    """align(A,B); align(A,B) — second align fails precondition."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    assert trajectory_distinct(prog, n_shapes=2) is False


def test_trajectory_distinct_translate_under_rotation_no_false_cancel():
    """rot90; translate_up; rot90 × 3; translate_down — after 4 rot's A
    is back to e, then translate_down; but translate distance depends on
    orientation at each step, so positions don't trivially cancel. The
    rot × 4 itself causes rejection before we get there."""
    prog = [Step(ALL, "rot90")] * 4 + [Step(ALL, "translate_up")]
    assert trajectory_distinct(prog, n_shapes=1) is False


def test_trajectory_distinct_add_border_no_inverse():
    """add_border has no symbolic inverse — cannot revisit prior state."""
    prog = [Step(ALL, "add_border")] * 3
    assert trajectory_distinct(prog, n_shapes=1) is True


# ---------------------------------------------------------------------------
# no_dead_writes — Phase 1 Rule 2
# ---------------------------------------------------------------------------


def test_no_dead_writes_empty():
    assert no_dead_writes([], n_shapes=1) is True


def test_no_dead_writes_single_op():
    assert no_dead_writes([Step(ALL, "rot90")], n_shapes=1) is True


def test_dead_writes_align_chain():
    """align(A, B); align(A, C) — first align's row overwritten by
    second, with no intervening read of A.row."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "align_row", target=_idx(3)),
    ]
    assert no_dead_writes(prog, n_shapes=3) is False


def test_dead_writes_translate_then_align():
    """trans_up(A); align(A, X) — translate's row overwrite is dead
    (align doesn't read A.row computationally)."""
    prog = [
        Step(_idx(1), "translate_up"),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    assert no_dead_writes(prog, n_shapes=2) is False


def test_dead_writes_align_translate_align_middle_dead():
    """align(A,B); trans(A); align(A,C) — translate's write dead
    (second align overwrites without reading A.row computationally).
    First align is live via translate's computational read of A.row."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "translate_up"),
        Step(_idx(1), "align_row", target=_idx(3)),
    ]
    assert no_dead_writes(prog, n_shapes=3) is False


def test_no_dead_writes_align_then_translate():
    """align(A, B); translate_up(A) — translate reads A.row
    computationally; align's write is live. And translate's own write
    has no later step to overwrite it, so observable in final. Accept."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "translate_up"),
    ]
    assert no_dead_writes(prog, n_shapes=2) is True


def test_dead_writes_double_hollow():
    """hollow(A); hollow(A) — second hollow overwrites without reading;
    first hollow dead."""
    prog = [Step(ALL, "hollow"), Step(ALL, "hollow")]
    assert no_dead_writes(prog, n_shapes=1) is False


def test_no_dead_writes_hollow_then_fill():
    """hollow(A); fill_interior(A) — fill's computational read of
    grid_shape keeps hollow's write live."""
    prog = [Step(ALL, "hollow"), Step(ALL, "fill_interior")]
    assert no_dead_writes(prog, n_shapes=1) is True


def test_dead_writes_hollow_fill_hollow():
    """hollow(A); fill_interior(A); hollow(A) — fill's write to
    grid_shape is overwritten by second hollow which doesn't read."""
    prog = [
        Step(ALL, "hollow"),
        Step(ALL, "fill_interior"),
        Step(ALL, "hollow"),
    ]
    assert no_dead_writes(prog, n_shapes=1) is False


def test_no_dead_writes_rot_chain():
    """rot; rot; rot — each rot reads and writes orientation, all
    writes live via next step's read. No dead writes (but
    trajectory_distinct would reject if the total cycled back)."""
    prog = [Step(ALL, "rot90")] * 3
    assert no_dead_writes(prog, n_shapes=1) is True


def test_no_dead_writes_cross_shape():
    """rot(A); trans(B) — different shapes, no overwrite."""
    prog = [
        Step(_idx(1), "rot90"),
        Step(_idx(2), "translate_up"),
    ]
    assert no_dead_writes(prog, n_shapes=2) is True


def test_no_dead_writes_align_different_target_on_different_shape():
    """align(A, B); align(C, B) — writes to different shapes' rows.
    No overwrite, no dead."""
    prog = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(3), "align_row", target=_idx(2)),
    ]
    assert no_dead_writes(prog, n_shapes=3) is True


def test_no_dead_writes_swap_twice():
    """swap(A,B); swap(A,B) — each swap reads both colors; writes
    are live via next swap's read. No dead writes. (trajectory_distinct
    catches this as a cycle back to initial.)"""
    prog = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "swap_colors", target=_idx(2)),
    ]
    assert no_dead_writes(prog, n_shapes=2) is True


def test_no_dead_writes_swap_chain():
    """swap cycles — each swap reads both colors computationally."""
    prog = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(2), "swap_colors", target=_idx(3)),
    ]
    assert no_dead_writes(prog, n_shapes=3) is True


def test_dead_writes_add_border_then_align():
    """add_border(A); align_row(A, B) — border writes border_count, row,
    col. align overwrites row without reading. But border_count and col
    writes are still observable in final → not all dead → step live."""
    prog = [
        Step(_idx(1), "add_border"),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    assert no_dead_writes(prog, n_shapes=2) is True


def test_no_dead_writes_hollow_on_different_shape_doesnt_help():
    """hollow(A); hollow(B) — different shapes, no overwrite."""
    prog = [
        Step(_idx(1), "hollow"),
        Step(_idx(2), "hollow"),
    ]
    assert no_dead_writes(prog, n_shapes=2) is True
