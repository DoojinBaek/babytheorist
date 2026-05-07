"""Tests for babytheorist.canonical — Tier 1 minimality analysis."""

import sys; sys.path.insert(0, ".")
from itertools import product

import pytest

from babytheorist.core import ALL, Selector, Step
from babytheorist.canonical import (
    color_joint_bfs_table,
    d4_min_depths,
    is_minimal_tier1,
    is_minimal_tier2,
    is_minimal_tier3,
    is_minimal_tier4_swap,
    is_minimal_tier5_align,
    is_minimal_multi_shape_by_index,
    min_depth_border,
    min_depth_color,
    min_depth_grid,
    min_depth_orient,
    orient_border_position_bfs_table,
    orient_position_bfs_table,
    row_col_source_joint_bfs_table,
)
from babytheorist.symbolic import D4, D4_IDENTITY, PreconditionFail, apply_symbolic_program


# ---------------------------------------------------------------------------
# d4_min_depths
# ---------------------------------------------------------------------------


def test_d4_min_depths_rot_only():
    table = d4_min_depths(frozenset({"rot90"}))
    assert table == {
        D4(0, False): 0,
        D4(1, False): 1,
        D4(2, False): 2,
        D4(3, False): 3,
    }


def test_d4_min_depths_flip_h_only():
    table = d4_min_depths(frozenset({"flip_h"}))
    assert table == {D4(0, False): 0, D4(0, True): 1}


def test_d4_min_depths_full_vocab_flip_v_shortcut():
    """With flip_v in vocab, (2, T) = r²s is reachable in 1 step.
    Without it, (2, T) needs 3 steps (rot·rot·flip_h or flip_h·rot·rot)."""
    table_with = d4_min_depths(frozenset({"rot90", "flip_h", "flip_v"}))
    table_without = d4_min_depths(frozenset({"rot90", "flip_h"}))
    assert table_with[D4(2, True)] == 1
    assert table_without[D4(2, True)] == 3


def test_d4_min_depths_full_vocab_all_reachable():
    table = d4_min_depths(frozenset({"rot90", "flip_h", "flip_v"}))
    assert len(table) == 8  # all of D4


def test_d4_min_depths_empty_vocab():
    table = d4_min_depths(frozenset())
    assert table == {D4_IDENTITY: 0}


# ---------------------------------------------------------------------------
# per-dimension min_depth
# ---------------------------------------------------------------------------


def test_min_depth_color_next_color_count():
    vocab = frozenset({"next_color"})
    assert min_depth_color(0, vocab) == 0
    assert min_depth_color(1, vocab) == 1
    assert min_depth_color(8, vocab) == 8
    assert min_depth_color(9, vocab) == 0  # wraps mod 9
    assert min_depth_color(10, vocab) == 1


def test_min_depth_grid_identity():
    assert min_depth_grid("solid", "solid", frozenset()) == 0
    assert min_depth_grid("hollow", "hollow", frozenset()) == 0


def test_min_depth_grid_transitions():
    assert min_depth_grid("solid", "hollow", frozenset({"hollow"})) == 1
    assert min_depth_grid("hollow", "solid", frozenset({"fill_interior"})) == 1


def test_min_depth_border():
    assert min_depth_border(0, frozenset()) == 0
    assert min_depth_border(3, frozenset({"add_border"})) == 3


# ---------------------------------------------------------------------------
# is_minimal_tier1 — explicit known-gap cases
# ---------------------------------------------------------------------------


def test_tier1_catches_gap1_d4_alternative_path():
    """[rot, rot, flip_h] reaches r²s = flip_v. With flip_v in universe
    vocab, shorter [flip_v] reaches same in 1 step → redundant."""
    P = [Step(ALL, "rot90"), Step(ALL, "rot90"), Step(ALL, "flip_h")]
    vocab_full = frozenset({"rot90", "flip_h", "flip_v"})
    vocab_no_flipv = frozenset({"rot90", "flip_h"})
    # With flip_v available as alternative: non-minimal.
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab_full) is False
    # Without flip_v: r²s needs 3 orient-steps, P is minimal.
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab_no_flipv) is True


def test_tier1_catches_rot_cycle():
    """[rot90]×4 returns to identity — net orient = e with depth 0."""
    P = [Step(ALL, "rot90")] * 4
    assert is_minimal_tier1(P, n_shapes=1, vocab=frozenset({"rot90"})) is False


def test_tier1_accepts_rot_minimal():
    for d in (1, 2, 3):
        P = [Step(ALL, "rot90")] * d
        assert is_minimal_tier1(P, n_shapes=1, vocab=frozenset({"rot90"})) is True


def test_tier1_catches_gap3_hollow_rot_fill():
    """[hollow, rot90, fill_interior]: net (orient=r, grid=solid).
    Min depth = 1 (rot90 alone). Program depth 3 → redundant."""
    P = [Step(ALL, "hollow"), Step(ALL, "rot90"), Step(ALL, "fill_interior")]
    vocab = frozenset({"rot90", "hollow", "fill_interior"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is False


def test_tier1_catches_cross_dim_cycle_rot_nextcolor():
    """[next_color × 9, rot90] — color cycles back, orient=r.
    Min depth = 1 (just rot90). Depth 10 → redundant."""
    P = [Step(ALL, "next_color")] * 9 + [Step(ALL, "rot90")]
    vocab = frozenset({"rot90", "next_color"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is False


def test_tier1_accepts_rot_plus_color_independent():
    """[rot90, next_color]: net (r, 1). Min 1+1=2 = depth. Minimal."""
    P = [Step(ALL, "rot90"), Step(ALL, "next_color")]
    vocab = frozenset({"rot90", "next_color"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is True


def test_tier1_accepts_border_add_only():
    P = [Step(ALL, "add_border")] * 3
    assert is_minimal_tier1(
        P, n_shapes=1, vocab=frozenset({"add_border"})
    ) is True


def test_tier1_defers_on_translate():
    P = [Step(ALL, "translate_up"), Step(ALL, "translate_down")]
    vocab = frozenset({"translate_up", "translate_down"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is True


def test_tier1_defers_on_relational():
    P = [
        Step(Selector("by_index", 1), "swap_colors", target=Selector("by_index", 2)),
    ]
    vocab = frozenset({"swap_colors"})
    assert is_minimal_tier1(P, n_shapes=2, vocab=vocab) is True


def test_tier1_multi_shape_with_all_selector():
    """Multi-shape programs using only ALL selector are analysed via
    shape 0 — each shape evolves identically under ALL, so shape 0's
    verdict applies to the whole program. [rot90] × 4 is redundant on
    all shapes simultaneously."""
    P = [Step(ALL, "rot90")] * 4
    assert is_minimal_tier1(
        P, n_shapes=3, vocab=frozenset({"rot90"})
    ) is False


def test_tier1_defers_multi_shape_by_index():
    """Multi-shape programs with non-ALL selectors still defer (not yet
    supported)."""
    P = [Step(Selector("by_index", 1), "rot90")] * 4
    assert is_minimal_tier1(
        P, n_shapes=3, vocab=frozenset({"rot90"})
    ) is True


def test_tier1_hollow_init_shape():
    """[hollow] on initially-hollow: dead. [fill_interior] on hollow: minimal."""
    vocab = frozenset({"hollow", "fill_interior"})
    P = [Step(ALL, "hollow")]
    assert is_minimal_tier1(
        P, n_shapes=1, vocab=vocab, grid_shapes=["hollow"]
    ) is False
    Q = [Step(ALL, "fill_interior")]
    assert is_minimal_tier1(
        Q, n_shapes=1, vocab=vocab, grid_shapes=["hollow"]
    ) is True


def test_tier1_infeasible_rejected():
    P = [Step(ALL, "fill_interior")]
    vocab = frozenset({"fill_interior"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is False


# ---------------------------------------------------------------------------
# Oracle cross-check — Tier 1 scope vocabs must match brute-force
# ---------------------------------------------------------------------------


def _enum_steps_singleshape(primitives):
    return [Step(ALL, p) for p in primitives]


def _enum_programs(steps, depth):
    if depth == 0:
        return [[]]
    return [list(c) for c in product(steps, repeat=depth)]


def _final_state(program, *, grid_shapes=None):
    try:
        traj = apply_symbolic_program(program, 1, grid_shapes=grid_shapes)
    except PreconditionFail:
        return None
    return tuple(traj[-1])


def _collect_shorter_finals(programs_by_depth, depth, *, grid_shapes=None):
    finals = set()
    for d in range(depth):
        for Q in programs_by_depth[d]:
            f = _final_state(Q, grid_shapes=grid_shapes)
            if f is not None:
                finals.add(f)
    return finals


def _run_oracle_tier1(primitives, *, max_depth, grid_shapes=None):
    """Assert is_minimal_tier1 matches oracle for every program."""
    steps = _enum_steps_singleshape(primitives)
    vocab = frozenset(primitives)
    programs_by_depth = {
        d: _enum_programs(steps, d) for d in range(max_depth + 1)
    }

    mismatches = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(
            programs_by_depth, d, grid_shapes=grid_shapes
        )
        for P in programs_by_depth[d]:
            final = _final_state(P, grid_shapes=grid_shapes)
            oracle_rejects = final is None or final in shorter_finals
            rule_accepts = is_minimal_tier1(
                P, 1, vocab=vocab, grid_shapes=grid_shapes
            )
            if rule_accepts == oracle_rejects:
                mismatches.append((P, rule_accepts, oracle_rejects))

    assert not mismatches, _format_mismatches(mismatches)


def _format_mismatches(items):
    lines = [f"{len(items)} rule/oracle mismatches:"]
    for P, rule_accepts, oracle_rejects in items[:10]:
        desc = "[" + ", ".join(s.primitive for s in P) + "]"
        lines.append(
            f"  {desc}  rule_accepts={rule_accepts}, oracle_rejects={oracle_rejects}"
        )
    if len(items) > 10:
        lines.append(f"  ... and {len(items) - 10} more")
    return "\n".join(lines)


def test_oracle_tier1_d4_full_vocab():
    """{rot, flip_h, flip_v} — Tier 1 must catch all D4 alternative-path
    shortcuts (Gap 1)."""
    _run_oracle_tier1(["rot90", "flip_h", "flip_v"], max_depth=4)


def test_oracle_tier1_rot_and_next_color():
    """Independent dims — rot cycles around next_color caught by Tier 1."""
    _run_oracle_tier1(["rot90", "next_color"], max_depth=4)


def test_oracle_tier1_rot_hollow_fill():
    """Cross-dim cycle [hollow, rot, fill] = [rot] caught by Tier 1."""
    _run_oracle_tier1(["rot90", "hollow", "fill_interior"], max_depth=3)


def test_oracle_tier1_hollow_rot_add_border():
    """Orient + grid + border — all non-interacting, Tier 1 complete."""
    _run_oracle_tier1(
        ["rot90", "hollow", "fill_interior", "add_border"], max_depth=3,
    )


def test_oracle_tier1_full_nontranslate_vocab():
    """Full Tier 1 vocab on shallow depth."""
    _run_oracle_tier1(
        ["rot90", "flip_h", "next_color", "hollow", "fill_interior"],
        max_depth=3,
    )


# ---------------------------------------------------------------------------
# Tier 2 — orient + translate joint BFS
# ---------------------------------------------------------------------------


def test_orient_position_bfs_rot_only():
    """rot alone: 4 orient states, position never changes."""
    table = orient_position_bfs_table(frozenset({"rot90"}))
    # Position always (0, 0, 0, 0); only orient cycles.
    positions_seen = {(s[1], s[2], s[3], s[4]) for s in table}
    assert positions_seen == {(0, 0, 0, 0)}
    # 4 distinct orient values.
    orients = {s[0] for s in table}
    assert len(orients) == 4


def test_orient_position_bfs_translate_up_only():
    """trans_up alone: position keeps going up, orient stays e."""
    table = orient_position_bfs_table(frozenset({"translate_up"}))
    # All states have orient=e.
    for s in table:
        assert s[0] == D4_IDENTITY
    # rh grows strictly negative.
    rhs = sorted(s[1] for s in table)
    assert rhs[0] == -20  # up to depth cap
    assert rhs[-1] == 0


def test_orient_position_bfs_rot_plus_trans():
    """{rot, trans_up}: orient and row-coefs both vary."""
    table = orient_position_bfs_table(frozenset({"rot90", "translate_up"}))
    # Starting state is at distance 0.
    assert table[(D4_IDENTITY, 0, 0, 0, 0)] == 0
    # [rot, trans_up]: orient=r, rh=0 but rw=-1 (rot-odd phase at trans).
    assert table[(D4(1, False), 0, -1, 0, 0)] == 2
    # [trans_up, rot]: orient=r, rh=-1 (rot-even at trans).
    assert table[(D4(1, False), -1, 0, 0, 0)] == 2


def test_tier2_catches_split_rot_around_translate():
    """Known Gap 2: [rot, rot, trans_up, rot, rot] reaches (e, rh=-1) in
    depth 5, but [trans_up] reaches it in depth 1. Tier 2 must reject."""
    P = [
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "translate_up"),
        Step(ALL, "rot90"), Step(ALL, "rot90"),
    ]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is False


def test_tier2_accepts_minimal_translate():
    P = [Step(ALL, "translate_up")]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


def test_tier2_accepts_rot_then_trans():
    """[rot, translate_up]: orient=r, rh=0, rw=-1. Minimal at depth 2."""
    P = [Step(ALL, "rot90"), Step(ALL, "translate_up")]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


def test_tier2_accepts_rot_trans_rot3_trans_down_non_redundant():
    """[rot, trans_up, rot^3, trans_down] — rot cycle but translates at
    different phases produce w_A - h_A row delta, depth 6 minimal."""
    P = [
        Step(ALL, "rot90"),
        Step(ALL, "translate_up"),
        Step(ALL, "rot90"), Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "translate_down"),
    ]
    vocab = frozenset({"rot90", "translate_up", "translate_down"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


def test_tier2_defers_on_add_border():
    """add_border present → Tier 2 defers (scheduling-dependent const)."""
    P = [Step(ALL, "translate_up"), Step(ALL, "add_border")]
    vocab = frozenset({"translate_up", "add_border"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


def test_tier2_defers_on_relational():
    P = [Step(Selector("by_index", 1), "align_row",
              target=Selector("by_index", 2))]
    vocab = frozenset({"align_row", "translate_up"})
    assert is_minimal_tier2(P, n_shapes=2, vocab=vocab) is True


def test_tier2_multi_shape_all_selector():
    """Multi-shape ALL-selector programs analysed on shape 0."""
    # [trans_up, trans_up] is minimal (row_coef_h=-2, minimal depth 2).
    P = [Step(ALL, "translate_up")] * 2
    vocab = frozenset({"translate_up"})
    assert is_minimal_tier2(P, n_shapes=3, vocab=vocab) is True
    # [rot, rot, trans_up, rot, rot] redundant (= [trans_up]).
    Q = [
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "translate_up"),
        Step(ALL, "rot90"), Step(ALL, "rot90"),
    ]
    vocab_q = frozenset({"rot90", "translate_up"})
    assert is_minimal_tier2(Q, n_shapes=3, vocab=vocab_q) is False


def test_tier2_defers_multi_shape_non_all():
    """Multi-shape with non-ALL selector defers."""
    P = [Step(Selector("by_index", 1), "translate_up")] * 2
    vocab = frozenset({"translate_up"})
    assert is_minimal_tier2(P, n_shapes=3, vocab=vocab) is True


def test_tier2_defers_no_translate():
    """No translate → Tier 1 is the right check, Tier 2 defers."""
    P = [Step(ALL, "rot90")] * 4  # would be redundant, but Tier 2 defers
    vocab = frozenset({"rot90", "flip_h"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


def test_tier2_color_and_grid_additive():
    """[rot, translate_up, next_color]: orient=r, rh=0, rw=-1, color_offset=1.
    min_depth_orient_position = 2 + min_depth_color = 1 → total 3. Minimal."""
    P = [Step(ALL, "rot90"), Step(ALL, "translate_up"), Step(ALL, "next_color")]
    vocab = frozenset({"rot90", "translate_up", "next_color"})
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is True


# ---------------------------------------------------------------------------
# Oracle cross-check: Tier 2 agrees with brute-force on translate vocabs
# ---------------------------------------------------------------------------


def _run_oracle_tier2(primitives, *, max_depth):
    """Enumerate programs; check is_minimal_tier2 matches oracle."""
    steps = _enum_steps_singleshape(primitives)
    vocab = frozenset(primitives)
    programs_by_depth = {d: _enum_programs(steps, d) for d in range(max_depth + 1)}

    mismatches = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(programs_by_depth, d)
        for P in programs_by_depth[d]:
            final = _final_state(P)
            oracle_rejects = final is None or final in shorter_finals
            rule_accepts = is_minimal_tier2(P, 1, vocab=vocab)
            if rule_accepts == oracle_rejects:
                mismatches.append((P, rule_accepts, oracle_rejects))
    assert not mismatches, _format_mismatches(mismatches)


def test_oracle_tier2_rot_translate():
    """{rot90, translate_up} — split-cyclic phase gap closed."""
    _run_oracle_tier2(["rot90", "translate_up"], max_depth=4)


def test_oracle_tier2_rot_translate_bidirectional():
    """{rot90, translate_up, translate_down} — cancel pairs + phase nav."""
    _run_oracle_tier2(["rot90", "translate_up", "translate_down"], max_depth=3)


def test_oracle_tier2_rot_translate_all_four():
    """{rot90, all translates} — 2D walk with rotation."""
    _run_oracle_tier2(
        ["rot90", "translate_up", "translate_down", "translate_left", "translate_right"],
        max_depth=3,
    )


def test_oracle_tier2_rot_flip_translate():
    """Full D4 + translate."""
    _run_oracle_tier2(
        ["rot90", "flip_h", "flip_v", "translate_up"],
        max_depth=3,
    )


def test_oracle_tier2_translate_with_color():
    """Tier 2 + color orthogonal dimension."""
    _run_oracle_tier2(
        ["rot90", "translate_up", "next_color"],
        max_depth=3,
    )


# ---------------------------------------------------------------------------
# Tier 3 — add_border + translate joint BFS
# ---------------------------------------------------------------------------


def test_orient_border_position_bfs_add_border_alone():
    """add_border only: border grows monotone, row/col const shift by -1 each."""
    table = orient_border_position_bfs_table(frozenset({"add_border"}))
    # [add_border × 3]: border=3, const_shift=-3 each.
    target = (D4_IDENTITY, 3, 0, 0, 0, 0, -3, -3)
    assert table[target] == 3


def test_orient_border_position_bfs_border_then_trans():
    """[add_border, translate_up]: border=1, row = -(h+2) as coef+const."""
    vocab = frozenset({"add_border", "translate_up"})
    table = orient_border_position_bfs_table(vocab)
    target = (D4_IDENTITY, 1, -1, 0, 0, 0, -3, -1)  # rc = -1 (border) + -2 (trans at border=1)
    assert table[target] == 2


def test_orient_border_position_bfs_trans_then_border():
    """[translate_up, add_border]: border=1, row = -h - 1 (coef -1, const -1)."""
    vocab = frozenset({"add_border", "translate_up"})
    table = orient_border_position_bfs_table(vocab)
    # trans at border=0: coef -1, const 0. Then add_border: border=1, row -= 1, col -= 1.
    target = (D4_IDENTITY, 1, -1, 0, 0, 0, -1, -1)
    assert table[target] == 2


def test_orient_border_position_bfs_orderings_distinct():
    """[add_border, translate_up] and [translate_up, add_border] reach
    different states (different row_const)."""
    vocab = frozenset({"add_border", "translate_up"})
    table = orient_border_position_bfs_table(vocab)
    # Both reach border=1, coef -1, but different const.
    state_a = (D4_IDENTITY, 1, -1, 0, 0, 0, -3, -1)
    state_b = (D4_IDENTITY, 1, -1, 0, 0, 0, -1, -1)
    assert state_a in table
    assert state_b in table
    assert state_a != state_b


def test_tier3_catches_add_border_rot_cycle_with_translate():
    """[add_border, rot×3, translate_up, add_border, rot] with net identity
    cycling — would require checking shorter equivalent exists. Construct
    a specific redundant program."""
    # Depth 4 program: [add_border, rot, rot, rot, rot, add_border, translate_up]
    # Actually simpler: [rot, rot, rot, rot, add_border, translate_up] depth 6.
    # orient cycles back to e, final border=1, row=-1-(h+2)=-h-3.
    # Shorter: [add_border, translate_up] depth 2. Same effect.
    # trajectory_distinct would catch rot cycle; but Tier 3 is what analyses
    # add_border+translate interaction. Ensure Tier 3 also rejects.
    P = [
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "add_border"), Step(ALL, "translate_up"),
    ]
    vocab = frozenset({"rot90", "add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is False


def test_tier3_accepts_minimal_border_translate():
    """[add_border, translate_up]: depth 2, minimal (no shorter equivalent)."""
    P = [Step(ALL, "add_border"), Step(ALL, "translate_up")]
    vocab = frozenset({"rot90", "add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


def test_tier3_accepts_trans_then_border():
    """Different ordering, still minimal — distinct effect."""
    P = [Step(ALL, "translate_up"), Step(ALL, "add_border")]
    vocab = frozenset({"rot90", "add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


def test_tier3_defers_without_add_border():
    """No add_border → Tier 2 handles, Tier 3 defers."""
    P = [Step(ALL, "rot90"), Step(ALL, "translate_up")]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


def test_tier3_defers_without_translate():
    """No translate → Tier 1 handles."""
    P = [Step(ALL, "rot90"), Step(ALL, "add_border")]
    vocab = frozenset({"rot90", "add_border"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


def test_tier3_defers_on_relational():
    P = [Step(Selector("by_index", 1), "swap_colors",
              target=Selector("by_index", 2))]
    vocab = frozenset({"swap_colors", "add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=2, vocab=vocab) is True


def test_tier3_multi_shape_all_selector():
    """Multi-shape ALL-selector Tier 3 analysed on shape 0."""
    # Minimal program: [add_border, translate_up] depth 2.
    P = [Step(ALL, "add_border"), Step(ALL, "translate_up")]
    vocab = frozenset({"add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=3, vocab=vocab) is True
    # Redundant: rot cycle wrapping simple border+trans.
    Q = [
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "add_border"), Step(ALL, "translate_up"),
    ]
    vocab_q = frozenset({"rot90", "add_border", "translate_up"})
    assert is_minimal_tier3(Q, n_shapes=3, vocab=vocab_q) is False


def test_tier3_defers_multi_shape_non_all():
    P = [
        Step(Selector("by_index", 1), "add_border"),
        Step(Selector("by_index", 1), "translate_up"),
    ]
    vocab = frozenset({"add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=3, vocab=vocab) is True


def test_tier3_defers_beyond_bfs_cap():
    """Programs longer than BFS cap return True (cannot conclude)."""
    P = [Step(ALL, "translate_up")] * 11  # depth 11 > cap 10
    vocab = frozenset({"add_border", "translate_up"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


def test_tier3_color_and_grid_additive():
    """Tier 3 + color + grid orthogonal."""
    P = [
        Step(ALL, "add_border"), Step(ALL, "translate_up"),
        Step(ALL, "next_color"),
    ]
    vocab = frozenset({"add_border", "translate_up", "next_color"})
    assert is_minimal_tier3(P, n_shapes=1, vocab=vocab) is True


# ---------------------------------------------------------------------------
# Oracle cross-check Tier 3
# ---------------------------------------------------------------------------


def _run_oracle_tier3(primitives, *, max_depth):
    steps = _enum_steps_singleshape(primitives)
    vocab = frozenset(primitives)
    programs_by_depth = {d: _enum_programs(steps, d) for d in range(max_depth + 1)}

    mismatches = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(programs_by_depth, d)
        for P in programs_by_depth[d]:
            final = _final_state(P)
            oracle_rejects = final is None or final in shorter_finals
            rule_accepts = is_minimal_tier3(P, 1, vocab=vocab)
            if rule_accepts == oracle_rejects:
                mismatches.append((P, rule_accepts, oracle_rejects))
    assert not mismatches, _format_mismatches(mismatches)


def test_oracle_tier3_border_translate_up():
    """{add_border, translate_up} — simplest Tier 3 vocab."""
    _run_oracle_tier3(["add_border", "translate_up"], max_depth=4)


def test_oracle_tier3_border_translate_bidir():
    """{add_border, translate_up, translate_down}."""
    _run_oracle_tier3(
        ["add_border", "translate_up", "translate_down"], max_depth=3,
    )


def test_oracle_tier3_rot_border_translate():
    """{rot90, add_border, translate_up} — orient + border + translate."""
    _run_oracle_tier3(
        ["rot90", "add_border", "translate_up"], max_depth=3,
    )


def test_oracle_tier3_rot_border_trans_all_directions():
    """Full orient + border + all translates."""
    _run_oracle_tier3(
        ["rot90", "add_border", "translate_up", "translate_down"], max_depth=3,
    )


# ---------------------------------------------------------------------------
# Multi-shape by_index slicing
# ---------------------------------------------------------------------------


def _idx(k):
    return Selector("by_index", k)


def test_multishape_by_index_rejects_per_shape_redundant():
    """[rot(1), rot(1), rot(1), rot(1), rot(2)]: shape 1's sub-program
    is [rot]×4 (redundant); shape 2's is [rot] (minimal). Joint
    redundant → reject."""
    P = [
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(2), "rot90"),
    ]
    vocab = frozenset({"rot90"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_multishape_by_index_accepts_independent_minimal():
    """Different shapes, different effects, each minimal."""
    P = [
        Step(_idx(1), "rot90"),
        Step(_idx(2), "rot90"),
        Step(_idx(2), "rot90"),
    ]
    vocab = frozenset({"rot90"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_multishape_by_index_rejects_translate_shortcut():
    """Shape 1: rot×2 + trans_up + rot×2 reachable in 1 step (trans_up)."""
    P = [
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(1), "translate_up"),
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(2), "rot90"),
    ]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_multishape_by_index_defers_on_all_selector():
    """ALL selector → existing per-level checks apply; this function
    defers (returns True) to avoid double-counting."""
    P = [Step(ALL, "rot90")] * 4
    vocab = frozenset({"rot90"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is True  # defers — single-shape Tier 1 handles this via ALL branch


def test_multishape_by_index_defers_on_relational_in_program():
    """Relational step in program → slicing returns None → defer."""
    P = [Step(_idx(1), "swap_colors", target=_idx(2))]
    vocab = frozenset({"swap_colors"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_multishape_by_index_handles_relational_in_vocab():
    """Relational IN VOCAB but not in program: analyse anyway via the
    non-relational subvocab (sound, may miss relational shortcuts)."""
    P = [
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),  # redundant rot×4
        Step(_idx(2), "rot90"),
    ]
    # vocab includes swap_colors but program doesn't use it.
    vocab = frozenset({"rot90", "swap_colors"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_multishape_by_index_defers_single_shape():
    P = [Step(_idx(1), "rot90")] * 4
    vocab = frozenset({"rot90"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=1, vocab=vocab,
    ) is True  # single-shape — existing rules handle


def test_multishape_by_index_empty_slice():
    """If one shape has no sub-program, its slice is trivially minimal."""
    P = [Step(_idx(1), "rot90")]  # only shape 1 targeted
    vocab = frozenset({"rot90"})
    assert is_minimal_multi_shape_by_index(
        P, n_shapes=2, vocab=vocab,
    ) is True


# ---------------------------------------------------------------------------
# Tier 4 — multi-shape swap_colors joint BFS
# ---------------------------------------------------------------------------


def test_color_joint_bfs_two_shapes_swap():
    """{swap_colors, next_color} on 2 shapes. swap exchanges colors."""
    table = color_joint_bfs_table(
        frozenset({"swap_colors", "next_color"}), n_shapes=2,
    )
    # initial state reachable in 0 steps
    from babytheorist.symbolic import ColorVal
    initial = (ColorVal("0", 0), ColorVal("1", 0))
    assert table[initial] == 0
    # After swap: swapped.
    swapped = (ColorVal("1", 0), ColorVal("0", 0))
    assert table[swapped] == 1
    # After next_color(ALL): both offsets += 1.
    shifted = (ColorVal("0", 1), ColorVal("1", 1))
    assert table[shifted] == 1


def test_tier4_catches_swap_redundant():
    """[swap(1,2), swap(1,2)] — returns to initial, redundant vs empty."""
    P = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "swap_colors", target=_idx(2)),
    ]
    vocab = frozenset({"swap_colors"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_tier4_accepts_swap_once():
    P = [Step(_idx(1), "swap_colors", target=_idx(2))]
    vocab = frozenset({"swap_colors"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier4_accepts_three_swap_cycle():
    """[swap(1,2), swap(2,3), swap(1,3)]: A cycles back to initial.
    Net effect: shapes 2 and 3 swapped. Equivalent to [swap(2,3)] depth 1.
    Redundant."""
    P = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(2), "swap_colors", target=_idx(3)),
        Step(_idx(1), "swap_colors", target=_idx(3)),
    ]
    vocab = frozenset({"swap_colors"})
    assert is_minimal_tier4_swap(
        P, n_shapes=3, vocab=vocab,
    ) is False


def test_tier4_swap_plus_rot_non_redundant():
    """[swap(1,2), rot(1)]: color swapped + shape 1 rotated.
    Min depth: 1 (color) + 1 (rot) = 2. Program len 2. Minimal."""
    P = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "rot90"),
    ]
    vocab = frozenset({"swap_colors", "rot90"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier4_swap_plus_rot_redundant():
    """[rot(1), rot(1), rot(1), rot(1), swap(1,2)]: rot×4 on shape 1 = identity.
    Color swapped. Shorter: [swap(1,2)] depth 1. Redundant."""
    P = [
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(1), "rot90"), Step(_idx(1), "rot90"),
        Step(_idx(1), "swap_colors", target=_idx(2)),
    ]
    vocab = frozenset({"swap_colors", "rot90"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_tier4_defers_on_align_in_program():
    """Align step in the program itself → defer (row dim coupled)."""
    P = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "swap_colors", target=_idx(2)),
    ]
    vocab = frozenset({"swap_colors", "align_row"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier4_accepts_align_in_vocab_not_program():
    """Align in vocab but not in program → analyze (align in shorter Q
    can't shortcut color dim since align only moves rows)."""
    P = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "swap_colors", target=_idx(2)),  # redundant
    ]
    vocab = frozenset({"swap_colors", "align_row"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_tier4_defers_on_all_selector():
    """ALL selector → defer (T4 slicing is by_index only)."""
    P = [Step(ALL, "next_color")]
    vocab = frozenset({"swap_colors", "next_color"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier4_defers_no_swap_in_vocab():
    """Without swap_colors, T4 isn't applicable."""
    P = [Step(_idx(1), "rot90")]
    vocab = frozenset({"rot90"})
    assert is_minimal_tier4_swap(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier4_defers_single_shape():
    P = [Step(_idx(1), "swap_colors", target=_idx(1))]  # invalid anyway
    vocab = frozenset({"swap_colors"})
    assert is_minimal_tier4_swap(
        P, n_shapes=1, vocab=vocab,
    ) is True


# ---------------------------------------------------------------------------
# Tier 5 — multi-shape align_row/align_col joint BFS
# ---------------------------------------------------------------------------


def test_row_col_source_bfs_align_row_only():
    table = row_col_source_joint_bfs_table(
        frozenset({"align_row"}), n_shapes=3,
    )
    # Initial state
    initial = ((0, 1, 2), (0, 1, 2))
    assert table[initial] == 0
    # After align_row(1, 2): shape 0 row source = 1 (index of shape 2 in 0-based).
    # Wait — align_row(by_index(1), by_index(2)) means src=shape 0, tgt=shape 1.
    # Hmm, let me re-check. by_index is 1-based, so by_index(1) = shape 0,
    # by_index(2) = shape 1.
    # The BFS table uses 0-based shape indices internally.
    # After align(0, 1): rows[0] = rows[1] = 1.
    after_0_1 = ((1, 1, 2), (0, 1, 2))
    assert table[after_0_1] == 1
    # Both aligning to same source.
    after_all_1 = ((1, 1, 1), (0, 1, 2))
    # rows[0]=1, rows[2]=1 requires two aligns: align(0,1) + align(2,1)
    assert table[after_all_1] == 2


def test_row_col_source_bfs_both_axes():
    table = row_col_source_joint_bfs_table(
        frozenset({"align_row", "align_col"}), n_shapes=3,
    )
    # Row and col align independently.
    state = ((1, 1, 2), (2, 1, 2))  # shape 0 row from 1, col from 2
    # Requires: align_row(0, 1) + align_col(0, 2) = depth 2.
    assert table[state] == 2


def test_tier5_catches_align_chain():
    """[align_row(1, 2), align_row(1, 3)]: shape 0's row source goes
    1 → 2. Depth 1 [align_row(1, 3)] reaches same. Redundant."""
    P = [
        Step(_idx(1), "align_row", target=_idx(2)),
        Step(_idx(1), "align_row", target=_idx(3)),
    ]
    vocab = frozenset({"align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=3, vocab=vocab,
    ) is False


def test_tier5_catches_flip_cycle_around_align():
    """Regression: the diagnostic-discovered case.
    [flip_v(1), next_color(1), flip_v(1), align_col(1, 2)]:
    flip_v² cancels; net = color+1 + align_col. Shorter:
    [next_color(1), align_col(1, 2)] depth 2. Redundant."""
    P = [
        Step(_idx(1), "flip_v"),
        Step(_idx(1), "next_color"),
        Step(_idx(1), "flip_v"),
        Step(_idx(1), "align_col", target=_idx(2)),
    ]
    vocab = frozenset({"flip_v", "next_color", "align_col"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_tier5_accepts_minimal_align():
    P = [Step(_idx(1), "align_row", target=_idx(2))]
    vocab = frozenset({"align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=3, vocab=vocab,
    ) is True


def test_tier5_accepts_align_with_rot_non_redundant():
    """[rot(1), align_row(1, 2)]: rot on shape 0 + align. Non-redundant."""
    P = [
        Step(_idx(1), "rot90"),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    vocab = frozenset({"rot90", "align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier5_catches_rot_cycle_around_align():
    """[rot(1)×4, align_row(1, 2)]: rot×4 = e on shape 0. Shorter =
    [align_row(1, 2)]. Redundant."""
    P = [Step(_idx(1), "rot90")] * 4 + [
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    vocab = frozenset({"rot90", "align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is False


def test_tier5_defers_on_translate():
    """Translate breaks the source-label abstraction."""
    P = [
        Step(_idx(1), "translate_up"),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    vocab = frozenset({"translate_up", "align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True  # defers


def test_tier5_defers_on_add_border():
    P = [
        Step(_idx(1), "add_border"),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    vocab = frozenset({"add_border", "align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier5_defers_on_swap_in_program():
    """Swap in program → defer (Tier 4 handles)."""
    P = [
        Step(_idx(1), "swap_colors", target=_idx(2)),
        Step(_idx(1), "align_row", target=_idx(2)),
    ]
    vocab = frozenset({"swap_colors", "align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier5_defers_no_align_in_vocab():
    P = [Step(_idx(1), "rot90")]
    vocab = frozenset({"rot90"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True


def test_tier5_defers_single_shape():
    P = [Step(_idx(1), "align_row", target=_idx(2))]
    vocab = frozenset({"align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=1, vocab=vocab,
    ) is True


def test_tier5_defers_dynamic_selector_target():
    """topmost as target (single-select dynamic) → Tier 5 defers."""
    P = [Step(_idx(1), "align_row", target=Selector("topmost"))]
    vocab = frozenset({"align_row"})
    assert is_minimal_tier5_align(
        P, n_shapes=2, vocab=vocab,
    ) is True  # defers (non-by_index target)
