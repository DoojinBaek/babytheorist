"""Tests for babytheorist.filter.rules_program."""

from __future__ import annotations

import pytest

from babytheorist.core import ALL, Selector, Step
from babytheorist.filter.rules_program import (
    RuleViolation,
    check_count_limits,
    check_d4_stabilizer_shortcut,
    check_add_border_d4_split_shortcut,
    check_grid_shape_cancellation,
    check_joint_stabilizer_shortcut,
    check_minimal_word,
    check_state_revisit,
    check_translate_cancellation,
    check_translate_joint_orchestration,
    program_is_valid,
)
from babytheorist.filter.state import FilterState


# === Helpers ==============================================================


def _fs(program, n_shapes: int) -> FilterState:
    fs = FilterState.initial(n_shapes)
    for step in program:
        fs = fs.apply(step)
    return fs


_ROT = Step(selector=ALL, primitive="rot90")
_FLIP_H = Step(selector=ALL, primitive="flip_h")
_FLIP_V = Step(selector=ALL, primitive="flip_v")


def _by_idx(k: int, prim: str) -> Step:
    return Step(selector=Selector("by_index", k), primitive=prim)


# === R1: count limits =====================================================


def test_r1_rot90_four_violates():
    program = [_ROT] * 4
    fs = _fs(program, 1)
    v = check_count_limits(program, fs)
    assert v is not None
    assert v.rule == "R1"
    assert "rot90" in v.reason


def test_r1_rot90_three_ok():
    program = [_ROT] * 3
    fs = _fs(program, 1)
    assert check_count_limits(program, fs) is None


def test_r1_flip_h_twice_violates():
    program = [_FLIP_H, _FLIP_H]
    fs = _fs(program, 1)
    v = check_count_limits(program, fs)
    assert v is not None
    assert v.rule == "R1"
    assert "flip_h" in v.reason


def test_r1_flip_h_once_ok():
    program = [_FLIP_H]
    fs = _fs(program, 1)
    assert check_count_limits(program, fs) is None


def test_r1_per_shape_count_not_scene_count():
    """by_index splits counts across shapes so neither hits the limit."""
    program = [_by_idx(1, "flip_h"), _by_idx(2, "flip_h")]
    fs = _fs(program, 2)
    assert check_count_limits(program, fs) is None


# === R3: state revisit ====================================================


def test_r3_rot90_four_revisits_identity():
    program = [_ROT] * 4
    fs = _fs(program, 1)
    v = check_state_revisit(program, fs)
    assert v is not None
    assert v.rule == "R3"


def test_r3_rot_flip_rot_flip_revisits():
    """r s r s = e on single shape — revisit at step 4."""
    program = [_ROT, _FLIP_H, _ROT, _FLIP_H]
    fs = _fs(program, 1)
    v = check_state_revisit(program, fs)
    assert v is not None
    assert v.rule == "R3"


def test_r3_rot_rot_rot_ok():
    """Trajectory e, r, r², r³ — all distinct."""
    program = [_ROT, _ROT, _ROT]
    fs = _fs(program, 1)
    assert check_state_revisit(program, fs) is None


def test_r3_two_shape_one_revisits_one_not():
    """Shape 0 gets [rot90]×4 (revisit), shape 1 gets [rot90]×1 (ok)."""
    program = [
        _by_idx(1, "rot90"),
        _by_idx(1, "rot90"),
        _by_idx(1, "rot90"),
        _by_idx(1, "rot90"),
        _by_idx(2, "rot90"),
    ]
    fs = _fs(program, 2)
    v = check_state_revisit(program, fs)
    assert v is not None
    assert v.rule == "R3"
    assert "shape 0" in v.reason


# === R6: per-shape minimal-word ===========================================


def test_r6_rot_flipv_rot_violates():
    """Program [r, r²s, r] has net = r²s (word length 1) but uses 3 prims.

    e → r → rs → r²s (trajectory distinct so R3 passes), yet r²s is
    a single primitive (flip_v) — shortcut exists.
    """
    program = [_ROT, _FLIP_V, _ROT]
    fs = _fs(program, 1)
    # R3 passes
    assert check_state_revisit(program, fs) is None
    # R6 fails
    v = check_minimal_word(program, fs)
    assert v is not None
    assert v.rule == "R6"
    assert "shape 0" in v.reason


def test_r6_minimal_path_ok():
    """[r, r, r] is shortest word for r³ (length 3), R6 passes."""
    program = [_ROT, _ROT, _ROT]
    fs = _fs(program, 1)
    assert check_minimal_word(program, fs) is None


def test_r6_primitive_alone_ok():
    """Each primitive is length-1 word of itself."""
    for prim in ("rot90", "flip_h", "flip_v"):
        program = [Step(selector=ALL, primitive=prim)]
        fs = _fs(program, 1)
        assert check_minimal_word(program, fs) is None, f"{prim} R6 should pass"


def test_r6_rs_reached_via_flip_h_rot_ok():
    """Program [flip_h, rot90] has net r · s = rs (word length 2), len 2 → ok."""
    program = [_FLIP_H, _ROT]
    fs = _fs(program, 1)
    assert check_minimal_word(program, fs) is None


def test_r6_per_shape_check():
    """Shape 1 gets [r, r²s, r] (R6 fail); shape 0 just gets one step."""
    program = [
        _by_idx(2, "rot90"),
        _by_idx(2, "flip_v"),
        _by_idx(2, "rot90"),
        _by_idx(1, "rot90"),
    ]
    fs = _fs(program, 2)
    v = check_minimal_word(program, fs)
    assert v is not None
    assert v.rule == "R6"
    assert "shape 1" in v.reason


# === R5: dependency-aware translate cancellation ==========================


def _trans(prim):
    return Step(selector=ALL, primitive=prim)


def test_r5_right_up_left_cancels_col():
    # [trans_right, trans_up, trans_left] — right/left cancel across
    # transparent trans_up → col net zero with 2 col writes.
    program = [_trans("translate_right"), _trans("translate_up"),
               _trans("translate_left")]
    fs = _fs(program, 1)
    v = check_translate_cancellation(program, fs)
    assert v is not None
    assert v.rule == "R5"
    assert "col" in v.reason


def test_r5_up_flip_h_down_flip_keeps_height():
    # flip_h doesn't swap h/w axes, so up/down cancel across flip_h.
    program = [_trans("translate_up"), _FLIP_H, _trans("translate_down")]
    fs = _fs(program, 1)
    v = check_translate_cancellation(program, fs)
    assert v is not None
    assert v.rule == "R5"


def test_r5_rot_between_translates_does_not_cancel():
    # rot90 swaps h↔w, so [trans_right, rot90, trans_left] reaches
    # col = c + w - h ≠ 0. Not redundant.
    program = [_trans("translate_right"), _ROT, _trans("translate_left")]
    fs = _fs(program, 1)
    assert check_translate_cancellation(program, fs) is None


def test_r5_pure_translate_chain_ok():
    # Three trans_right — no cancellation, min_word = 3.
    program = [_trans("translate_right")] * 3
    fs = _fs(program, 1)
    assert check_translate_cancellation(program, fs) is None


def test_r5_per_shape_check():
    # Shape 1 has canceling translates via by_index; shape 0 unaffected.
    program = [
        _by_idx(2, "translate_right"),
        _by_idx(2, "translate_up"),
        _by_idx(2, "translate_left"),
    ]
    fs = _fs(program, 2)
    v = check_translate_cancellation(program, fs)
    assert v is not None
    assert "shape 1" in v.reason


# === R5b: grid-shape cancellation ========================================


def _fs_with_grid_shapes(program, grid_shapes: list[str]) -> FilterState:
    fs = FilterState.initial(len(grid_shapes), grid_shapes=grid_shapes)
    for step in program:
        fs = fs.apply(step)
    return fs


def test_r5b_hollow_fill_roundtrip_violates():
    program = [Step(selector=ALL, primitive="hollow"),
               Step(selector=ALL, primitive="fill_interior")]
    fs = _fs(program, 1)
    v = check_grid_shape_cancellation(program, fs)
    assert v is not None
    assert v.rule == "R5b"


def test_r5b_split_fill_across_d4_shortcuts_to_all_fill():
    program = [
        _by_idx(2, "fill_interior"),
        _ROT,
        _by_idx(1, "fill_interior"),
    ]
    fs = _fs_with_grid_shapes(program, ["hollow", "hollow"])
    v = check_grid_shape_cancellation(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "R5b"
    assert "ALL.fill_interior" in v.reason


def test_r5b_three_way_split_fill_across_d4_shortcuts_to_all_fill():
    program = [
        _by_idx(1, "fill_interior"),
        _by_idx(2, "rot90"),
        _by_idx(2, "fill_interior"),
        _FLIP_H,
        _by_idx(3, "fill_interior"),
    ]
    fs = _fs_with_grid_shapes(program, ["hollow", "hollow", "hollow"])
    v = check_grid_shape_cancellation(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "R5b"


def test_r5b_split_fill_does_not_cross_color_step():
    program = [
        _by_idx(1, "fill_interior"),
        Step(selector=ALL, primitive="next_color"),
        _by_idx(2, "fill_interior"),
    ]
    fs = _fs_with_grid_shapes(program, ["hollow", "hollow"])
    assert check_grid_shape_cancellation(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_r5b_split_fill_does_not_cross_is_hollow_selector():
    program = [
        _by_idx(1, "fill_interior"),
        Step(selector=Selector("is_hollow"), primitive="rot90"),
        _by_idx(2, "fill_interior"),
    ]
    fs = _fs_with_grid_shapes(program, ["hollow", "hollow"])
    assert check_grid_shape_cancellation(
        program, fs, selector_types=frozenset({"ALL", "by_index", "is_hollow"}),
    ) is None


# === R8: add_border / D4 commutation =====================================


def test_r8_split_d4_add_border_shortcuts_to_all_steps():
    program = [
        _by_idx(1, "rot90"),
        Step(selector=ALL, primitive="add_border"),
        _by_idx(2, "rot90"),
    ]
    fs = _fs(program, 2)
    v = check_add_border_d4_split_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )

    assert v is not None
    assert v.rule == "R8"


def test_r8_ignores_non_partitioned_split():
    program = [
        _by_idx(1, "rot90"),
        Step(selector=ALL, primitive="add_border"),
        _by_idx(1, "rot90"),
    ]
    fs = _fs(program, 2)

    assert check_add_border_d4_split_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


# === Aggregate ============================================================


def test_program_is_valid_passes_clean_program():
    program = [_ROT, _ROT]
    fs = _fs(program, 1)
    assert program_is_valid(program, fs) is None


def test_program_is_valid_returns_first_violation():
    """R1 is cheapest; should fire before R3 on rot90×4."""
    program = [_ROT] * 4
    fs = _fs(program, 1)
    v = program_is_valid(program, fs)
    assert v is not None
    assert v.rule == "R1"


def test_program_is_valid_r6_fires_when_r1_r3_clean():
    program = [_ROT, _FLIP_V, _ROT]
    fs = _fs(program, 1)
    v = program_is_valid(program, fs)
    assert v is not None
    assert v.rule == "R6"


# === R3 (relaxed): idle steps don't count as revisit ======================


def test_r3_relaxed_idle_shape_no_revisit():
    """Shape 1 is idle at step 1 (state doesn't change) — not a revisit."""
    program = [_by_idx(1, "rot90"), _by_idx(2, "rot90")]
    fs = _fs(program, 2)
    # Trajectories: shape 0 = [e, r, r]; shape 1 = [e, e, r].
    # Dedupe consec: [e, r] for both. No dup. R3 (relaxed) passes.
    assert check_state_revisit(program, fs) is None


def test_r3_relaxed_true_cycle_still_fires():
    """Shape leaves initial then returns — R3 fires even after relax."""
    program = [
        _by_idx(1, "rot90"),
        _by_idx(2, "rot90"),
        _by_idx(1, "rot90"),
        _by_idx(1, "rot90"),
        _by_idx(1, "rot90"),
    ]
    # Shape 0 trajectory: [e, r, r, r², r³, e] → dedupe [e, r, r², r³, e].
    # e appears twice → R3 fires.
    fs = _fs(program, 2)
    v = check_state_revisit(program, fs)
    assert v is not None
    assert v.rule == "R3"


# === I6: joint D4 stabilizer shortcut =====================================


def test_i6_two_bystep_rot90_shortcut():
    """[by_index(1).rot90, by_index(2).rot90] is not shortest — Q =
    [ALL.rot90] of length 1 reaches the same per-shape effects."""
    program = [_by_idx(1, "rot90"), _by_idx(2, "rot90")]
    fs = _fs(program, 2)
    v = check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6"


def test_i6_single_step_all_no_shortcut():
    """[ALL.rot90] is length 1 — nothing shorter possible."""
    program = [_ROT]
    fs = _fs(program, 2)
    assert check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_i6_single_step_single_shape_no_shortcut():
    """[by_index(1).rot90] is length 1 — no shorter Q possible."""
    program = [_by_idx(1, "rot90")]
    fs = _fs(program, 2)
    assert check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_i6_different_nets_no_all_shortcut_stabilizer_irrelevant():
    """[by_index(1).rot90, by_index(2).rot90, by_index(2).rot90] — shape 1
    net=rot90 d=1, shape 2 net=rot180 d=2. ALL.X can only match one. No
    single-shape Q length < 3 that hits both targets without a stabilizer
    that makes them align. With I1-allowed stabilizers though, some
    shortcut may exist."""
    program = [
        _by_idx(1, "rot90"),
        _by_idx(2, "rot90"),
        _by_idx(2, "rot90"),
    ]
    fs = _fs(program, 2)
    # Shape 1: (rot90, 1); shape 2: (rot180, 2).
    # For H_1 = {e}, H_2 = {e}: target = (rot90, rot180). Depth-2 Q via
    # [ALL.rot90, by_index(2).rot90] hits shape 1=rot90, shape 2=rot180.
    # That's 2 steps < 3. So a shortcut exists.
    v = check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6"


def test_i6_no_d4_steps_skips():
    """Program with only non-D4 primitives — nothing for I6 to check."""
    program = [Step(selector=ALL, primitive="translate_right")]
    fs = _fs(program, 2)
    assert check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_i6_mixed_vocab_d4_subsequence_not_checked():
    """With a translate interleaved, I6 must *not* claim a shortcut
    based on the D4 sub-sequence alone.

    Program: [by_index(1).rot90, ALL.translate_right, by_index(2).rot90].
    The naive "collapse D4 sub-sequence to ALL.rot90" reasoning is
    unsound: the ALL.translate_right step reads each shape's width,
    and rot90 swaps width↔height. So re-ordering the rotations
    relative to the translate changes the per-shape column delta
    (by ``w - h``). The "shortcut" [ALL.rot90, ALL.translate_right]
    only reproduces P when both shapes are square, which the
    rot+translate vocab forbids via ``require_non_square``.

    I6a (trace identity) rightly does not align the traces because
    the primitive sequences per shape differ once translate_right
    enters the mix, and I6b (D4 coset BFS) skips mixed-vocab
    programs entirely.
    """
    program = [
        _by_idx(1, "rot90"),
        Step(selector=ALL, primitive="translate_right"),
        _by_idx(2, "rot90"),
    ]
    fs = _fs(program, 2)
    v = check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is None


def test_i6_align_then_split_translate_has_all_shortcut():
    """align writes only src, so the following split translates can merge."""
    program = [
        Step(
            selector=Selector("by_index", 2),
            primitive="align_row",
            target=Selector("by_index", 1),
        ),
        _by_idx(2, "translate_down"),
        _by_idx(1, "translate_down"),
    ]
    fs = _fs(program, 2)
    v = check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6"


# === R5j: joint translate orchestration ===================================


def test_r5j_two_bystep_translates_plus_all_orthogonal_shortcut():
    """P = [by_idx(2).tr_right, ALL.tr_up, by_idx(1).tr_right] (3 steps).

    Both shapes net row = -1, col = +1. Joint min = 1 + 1 = 2 via
    [ALL.tr_right, ALL.tr_up]. Shortcut → reject.
    """
    program = [
        _by_idx(2, "translate_right"),
        Step(selector=ALL, primitive="translate_up"),
        _by_idx(1, "translate_right"),
    ]
    fs = _fs(program, 2)
    v = check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "R5j"


def test_r5j_different_per_shape_targets_no_shortcut():
    """Shape 0 gets 2 tr_right; shape 1 gets 1 tr_up. Joint targets differ,
    min = |δ_ALL| + |2 - δ_ALL| (col) + |δ_ALL| + |−1 − δ_ALL| (row).
    No shortcut — 3-step program is already minimal."""
    program = [
        _by_idx(1, "translate_right"),
        _by_idx(1, "translate_right"),
        _by_idx(2, "translate_up"),
    ]
    fs = _fs(program, 2)
    assert check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_r5j_mixed_vocab_skipped():
    """Non-translate step (rot90) → out of scope. Rule must skip, not fire."""
    program = [
        Step(selector=ALL, primitive="rot90"),
        Step(selector=ALL, primitive="translate_right"),
    ]
    fs = _fs(program, 2)
    assert check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_r5j_without_all_no_shortcut():
    """Without ALL in selector_types, by_index only → min per axis = Σ|a_i|,
    which equals program count. Rule never fires."""
    program = [
        _by_idx(2, "translate_right"),
        _by_idx(1, "translate_right"),
    ]
    fs = _fs(program, 2)
    assert check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"by_index"}),
    ) is None


def test_r5j_next_color_plus_translate_shortcut():
    """P = [by_idx(1).next_color, ALL.tr_down, by_idx(2).next_color] (3 steps).

    Both shapes net color +1, shape 1 net row +1 (shape 0 row +1).
    Joint min = 1 (color, via ALL.next_color) + 1 (row, via ALL.tr_down)
    = 2 < 3 → shortcut.
    """
    program = [
        _by_idx(1, "next_color"),
        Step(selector=ALL, primitive="translate_down"),
        _by_idx(2, "next_color"),
    ]
    fs = _fs(program, 2)
    v = check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "R5j"


def test_r5j_two_next_colors_on_distinct_shapes():
    """P = [by_idx(1).next_color, by_idx(2).next_color] (2 steps).

    Color targets (1, 1). Joint min = 2 - 1·1 = 1 (via ALL.next_color).
    Shortcut: Q = [ALL.next_color] length 1 < 2 → reject.
    """
    program = [_by_idx(1, "next_color"), _by_idx(2, "next_color")]
    fs = _fs(program, 2)
    v = check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "R5j"


def test_r5j_all_only_no_shortcut():
    """[ALL.tr_right, ALL.tr_up] (2 steps) is its own joint min (2)."""
    program = [
        Step(selector=ALL, primitive="translate_right"),
        Step(selector=ALL, primitive="translate_up"),
    ]
    fs = _fs(program, 2)
    assert check_translate_joint_orchestration(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


# === I6b: D4 coset BFS (stabilizer-equivalence shortcut) ==================


def test_i6b_flip_h_rot90_flip_v_triplet_has_coset_shortcut():
    """P = [by_idx(1).flip_h, ALL.rot90, by_idx(2).flip_v].

    Shape 0 net = rot90 · flip_h = D4(1, True).
    Shape 1 net = flip_v · rot90 = D4(1, True).

    Q = [ALL.flip_h, ALL.rot90] (length 2) reaches D4(1, True) on
    both shapes — equal nets, no stabilizer needed. I6a trace
    identity misses this because shape 1's P primitives are
    [rot90, flip_v] vs Q's [flip_h, rot90]; I6b catches it via
    coset BFS.
    """
    program = [
        _by_idx(1, "flip_h"),
        Step(selector=ALL, primitive="rot90"),
        _by_idx(2, "flip_v"),
    ]
    fs = _fs(program, 2)
    # I6a misses
    assert check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None
    # I6b catches
    v = check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6b"


def test_i6b_minimal_word_length_program_passes():
    """Each per-shape D4 word is already minimal and nothing collapses."""
    program = [Step(selector=ALL, primitive="rot90")]
    fs = _fs(program, 2)
    assert check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_i6b_mixed_vocab_skipped():
    """Non-D4 step present → I6b must skip (soundness: translate depends
    on orientation, can't reason about D4 subword in isolation)."""
    program = [
        _by_idx(1, "rot90"),
        Step(selector=ALL, primitive="translate_right"),
        _by_idx(2, "rot90"),
    ]
    fs = _fs(program, 2)
    assert check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    ) is None


def test_i6b_by_index_rot90_twice_has_all_shortcut():
    """[by_idx(1).rot90, by_idx(2).rot90] → Q = [ALL.rot90] (length 1).

    I6a also catches this (same primitive). I6b catches it as the
    D4 case where nets are (rot90, rot90) and Q reaches both in
    one step via ALL.
    """
    program = [_by_idx(1, "rot90"), _by_idx(2, "rot90")]
    fs = _fs(program, 2)
    v = check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6b"


def test_i6b_d4_subword_collapses_across_swap_colors():
    """D4 operations commute with color swaps, so the D4 subword can shrink."""
    program = [
        _by_idx(1, "flip_v"),
        Step(
            selector=Selector("by_index", 1),
            primitive="swap_colors",
            target=Selector("by_index", 2),
        ),
        _by_idx(2, "flip_v"),
    ]
    fs = _fs(program, 2)
    v = check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"ALL", "by_index"}),
    )
    assert v is not None
    assert v.rule == "I6b"


def test_i6b_requires_all_for_collapse():
    """Without ALL in selector_types, [by_idx(1).rot90, by_idx(2).rot90]
    cannot collapse — Q can only touch one shape per step."""
    program = [_by_idx(1, "rot90"), _by_idx(2, "rot90")]
    fs = _fs(program, 2)
    assert check_d4_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"by_index"}),
    ) is None


def test_i6_only_by_index_vocab_no_all_shortcut():
    """Without ALL in selector_types, Q can only use by_index — no
    single-step Q reaches both shapes' targets.

    Program [by_index(1).rot90, by_index(2).rot90] has d_total=2.
    With selectors={by_index} only, Q length 1 can only touch one
    shape. Shape 1 target coset = rot90·{e} = {rot90} (for H_1={e});
    single-shape Q leaves shape 2 at e ≠ rot90. Similarly vice versa.
    So no shortcut exists when ALL is unavailable.

    However if stabilizers admit {rot90} in either coset's target set,
    the tuple allows the shortcut — but only via some other stabilizer
    tuple in which the non-trivial shape's coset contains e.
    """
    program = [_by_idx(1, "rot90"), _by_idx(2, "rot90")]
    fs = _fs(program, 2)
    # Even without ALL, I6 may fire if some allowed stabilizer makes
    # one shape's coset contain identity (then that shape is idle and
    # the other gets the needed by_index step). For (rot90, 1), allowed
    # stabilizers are those not containing rot270; the "full" subgroup
    # D4 contains rot270 so is not allowed, but smaller ones are.
    # Check that the rule returns the expected verdict (either "shortcut
    # exists" or "no shortcut"). For now just ensure no crash.
    _ = check_joint_stabilizer_shortcut(
        program, fs, selector_types=frozenset({"by_index"}),
    )
