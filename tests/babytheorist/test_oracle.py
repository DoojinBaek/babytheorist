"""Oracle tests — cross-check `trajectory_distinct` against brute-force
symbolic enumeration.

**Two properties, two directions.**

*Soundness*: if the rule rejects ``P``, some strictly shorter ``Q`` must
produce the same symbolic final state. Expected to hold for *any* vocab.

*Completeness*: if no strictly shorter ``Q`` is equivalent to ``P``,
the rule must accept ``P``. ``trajectory_distinct`` alone is only
complete on "safe" vocabs (single cyclic / monotone / toggle
dimensions). Combined with Tier 1 canonical-form analysis
(``babytheorist.canonical.is_minimal_tier1``), the filter is complete on all
non-interacting vocabs (no ``translate_*``, no relational, single shape).

The remaining completeness gap is ``translate_*`` interacting with
``rot90`` / ``add_border`` via current-height dependence — Tier 2
territory (not yet implemented). ``test_known_gap_rot_split_around_translate``
documents this.

**Vocabulary note.** The alignment primitives (``align_row``,
``align_col``) exhibit overwrite dead-code (``align(A,B); align(A,C)``
— the first align is dead) that Rule 1 cannot catch by design. Rule 2
(not yet implemented) handles these. We exclude ``align_*`` from oracle
vocabs until Rule 2 lands.
"""

from __future__ import annotations

import sys; sys.path.insert(0, ".")
from itertools import product

import pytest

from babytheorist.canonical import is_minimal_tier1
from babytheorist.core import ALL, Selector, Step
from babytheorist.program import no_dead_writes, trajectory_distinct
from babytheorist.symbolic import PreconditionFail, apply_symbolic_program


# ---------------------------------------------------------------------------
# Enumeration helpers
# ---------------------------------------------------------------------------


_RELATIONAL_NAMES = frozenset({"swap_colors", "align_row", "align_col"})


def _enum_steps(
    primitives: list[str],
    multi_selectors: list[Selector],
    single_selectors: list[Selector],
) -> list[Step]:
    """All valid ``Step(selector, primitive[, target])`` combinations."""
    steps: list[Step] = []
    for p in primitives:
        if p in _RELATIONAL_NAMES:
            for src in single_selectors:
                for tgt in single_selectors:
                    steps.append(Step(src, p, target=tgt))
        else:
            for sel in multi_selectors + single_selectors:
                steps.append(Step(sel, p))
    return steps


def _enum_programs(steps: list[Step], depth: int) -> list[list[Step]]:
    if depth == 0:
        return [[]]
    return [list(combo) for combo in product(steps, repeat=depth)]


def _final_state(program: list[Step], n_shapes: int, *, grid_shapes=None):
    """Hashable final symbolic state, or ``None`` if program infeasible."""
    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return None
    return tuple(traj[-1])


def _collect_shorter_finals(programs_by_depth, n_shapes, depth, *, grid_shapes=None):
    """Set of final states of all feasible programs with length < ``depth``."""
    finals = set()
    for d in range(depth):
        for Q in programs_by_depth[d]:
            f = _final_state(Q, n_shapes, grid_shapes=grid_shapes)
            if f is not None:
                finals.add(f)
    return finals


# ---------------------------------------------------------------------------
# Oracle runners
# ---------------------------------------------------------------------------


def _check_soundness(
    primitives: list[str],
    *,
    n_shapes: int,
    max_depth: int,
    multi_selectors: list[Selector] = None,
    single_selectors: list[Selector] = None,
    grid_shapes=None,
):
    """For every enumerated ``P``, assert: rule rejects ⟹ oracle rejects.

    Oracle rejects iff ``P`` is infeasible OR some strictly shorter
    program has the same final symbolic state.
    """
    multi_selectors = multi_selectors or []
    single_selectors = single_selectors or []
    steps = _enum_steps(primitives, multi_selectors, single_selectors)
    programs_by_depth = {d: _enum_programs(steps, d) for d in range(max_depth + 1)}

    violations = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(
            programs_by_depth, n_shapes, d, grid_shapes=grid_shapes,
        )
        for P in programs_by_depth[d]:
            rule_accept = trajectory_distinct(P, n_shapes, grid_shapes=grid_shapes)
            if rule_accept:
                continue
            final = _final_state(P, n_shapes, grid_shapes=grid_shapes)
            oracle_rejects = final is None or final in shorter_finals
            if not oracle_rejects:
                violations.append((P, final))

    assert not violations, _format_violations("soundness", violations)


def _check_full_equivalence(
    primitives: list[str],
    *,
    n_shapes: int,
    max_depth: int,
    multi_selectors: list[Selector] = None,
    single_selectors: list[Selector] = None,
    grid_shapes=None,
):
    """Assert rule accept/reject matches oracle on every enumerated P.

    Use only on "safe" vocabs (see module docstring).
    """
    multi_selectors = multi_selectors or []
    single_selectors = single_selectors or []
    steps = _enum_steps(primitives, multi_selectors, single_selectors)
    programs_by_depth = {d: _enum_programs(steps, d) for d in range(max_depth + 1)}

    mismatches = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(
            programs_by_depth, n_shapes, d, grid_shapes=grid_shapes,
        )
        for P in programs_by_depth[d]:
            rule_accept = trajectory_distinct(P, n_shapes, grid_shapes=grid_shapes)
            final = _final_state(P, n_shapes, grid_shapes=grid_shapes)
            oracle_rejects = final is None or final in shorter_finals
            if rule_accept == oracle_rejects:
                mismatches.append((P, rule_accept, oracle_rejects, final))

    assert not mismatches, _format_mismatches(mismatches)


def _format_violations(kind, items):
    lines = [f"{len(items)} {kind} violations:"]
    for P, final in items[:10]:
        lines.append(f"  rule rejected but oracle accepted: {_pretty(P)}  final={final}")
    if len(items) > 10:
        lines.append(f"  ... and {len(items) - 10} more")
    return "\n".join(lines)


def _format_mismatches(items):
    lines = [f"{len(items)} rule/oracle mismatches:"]
    for P, rule_accept, oracle_rejects, _final in items[:10]:
        lines.append(
            f"  {_pretty(P)}  rule_accept={rule_accept}, oracle_rejects={oracle_rejects}"
        )
    if len(items) > 10:
        lines.append(f"  ... and {len(items) - 10} more")
    return "\n".join(lines)


def _pretty(program: list[Step]) -> str:
    return "[" + ", ".join(
        (
            f"{s.primitive}({_pretty_sel(s.selector)},{_pretty_sel(s.target)})"
            if s.target is not None
            else f"{s.primitive}({_pretty_sel(s.selector)})"
        )
        for s in program
    ) + "]"


def _pretty_sel(sel: Selector) -> str:
    return f"{sel.type}({sel.param})" if sel.param is not None else sel.type


# ---------------------------------------------------------------------------
# Safe-vocab full-equivalence tests
# ---------------------------------------------------------------------------


def test_oracle_rot90_alone_is_complete():
    """Single cyclic generator (rot has order 4)."""
    _check_full_equivalence(
        ["rot90"], n_shapes=1, max_depth=5, multi_selectors=[ALL],
    )


def test_oracle_flip_h_alone_is_complete():
    """Single involution."""
    _check_full_equivalence(
        ["flip_h"], n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_oracle_next_color_alone_is_complete():
    """Single cyclic generator (order 9)."""
    _check_full_equivalence(
        ["next_color"], n_shapes=1, max_depth=10, multi_selectors=[ALL],
    )


def test_oracle_translate_up_alone_is_complete():
    """Monotone 1D walk — no return, no shortcut."""
    _check_full_equivalence(
        ["translate_up"], n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_oracle_translate_cancel_pair_is_complete():
    """1D walk with cancel pair — every non-minimal walk revisits a row."""
    _check_full_equivalence(
        ["translate_up", "translate_down"],
        n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_oracle_hollow_fill_toggle_is_complete():
    """Two-state toggle (solid ⇄ hollow). fill_interior's precondition
    on ``solid`` forces non-minimal walks to revisit ``solid``."""
    _check_full_equivalence(
        ["hollow", "fill_interior"],
        n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_oracle_hollow_fill_with_hollow_init_is_complete():
    """Same vocab, ``grid_shapes=['hollow']`` — ``[hollow]`` is dead and
    ``[fill_interior]`` is live. Exercises the initial-state override."""
    _check_full_equivalence(
        ["hollow", "fill_interior"],
        n_shapes=1, max_depth=4,
        multi_selectors=[ALL],
        grid_shapes=["hollow"],
    )


def test_oracle_add_border_alone_is_complete():
    """Monotone counter + fixed-direction position shift."""
    _check_full_equivalence(
        ["add_border"], n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


# ---------------------------------------------------------------------------
# Soundness-only tests (broader vocabs — known completeness gaps)
# ---------------------------------------------------------------------------


def test_soundness_d4_full_vocab_single_shape():
    """All three D4 generators. Alternative-path shortcuts exist but
    rule rejections must still be justified."""
    _check_soundness(
        ["rot90", "flip_h", "flip_v"],
        n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_soundness_rot_and_next_color():
    """Cross-dim independent cycling — gap: rot cycle around a
    next_color can slip past combined-state check."""
    _check_soundness(
        ["rot90", "next_color"],
        n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_soundness_rot_and_hollow_fill():
    """Cross-dim cycling: [hollow, rot, fill] = [rot] is rule-missed."""
    _check_soundness(
        ["rot90", "hollow", "fill_interior"],
        n_shapes=1, max_depth=3, multi_selectors=[ALL],
    )


def test_soundness_rot_and_translate():
    """Split-rot shortcut: [rot, rot, trans, rot, rot] ≡ [trans] at
    even parity."""
    _check_soundness(
        ["rot90", "translate_up", "translate_down"],
        n_shapes=1, max_depth=4, multi_selectors=[ALL],
    )


def test_soundness_two_shapes_rot_translate():
    """Cross-shape commutation: rot(A); trans(B); rot(A)^3."""
    _check_soundness(
        ["rot90", "translate_up"],
        n_shapes=2, max_depth=3,
        single_selectors=[Selector("by_index", 1), Selector("by_index", 2)],
    )


def test_soundness_two_shapes_swap_rot():
    """Swap (involution, relational) + rot."""
    _check_soundness(
        ["rot90", "swap_colors"],
        n_shapes=2, max_depth=3,
        single_selectors=[Selector("by_index", 1), Selector("by_index", 2)],
    )


def test_soundness_three_shapes_swap_cycles():
    """3-cycle swap on three shapes."""
    _check_soundness(
        ["swap_colors"],
        n_shapes=3, max_depth=3,
        single_selectors=[
            Selector("by_index", 1),
            Selector("by_index", 2),
            Selector("by_index", 3),
        ],
    )


def test_soundness_hollow_add_border_rot():
    """Grid-shape toggle + orientation + border."""
    _check_soundness(
        ["rot90", "hollow", "fill_interior", "add_border"],
        n_shapes=1, max_depth=3, multi_selectors=[ALL],
    )


# ---------------------------------------------------------------------------
# Known completeness gaps — documented, asserted as "rule misses"
# ---------------------------------------------------------------------------


def test_tier2_closes_split_rot_around_translate():
    """Former Gap 2 — now caught by Tier 2 canonical analysis.
    [rot, rot, trans_up, rot, rot] reaches (e, r_A - h_A), same as
    depth-1 [trans_up]. ``trajectory_distinct`` and Tier 1 don't
    see this; Tier 2's orient+translate BFS does."""
    from babytheorist.canonical import is_minimal_tier1, is_minimal_tier2
    P = [
        Step(ALL, "rot90"), Step(ALL, "rot90"),
        Step(ALL, "translate_up"),
        Step(ALL, "rot90"), Step(ALL, "rot90"),
    ]
    Q = [Step(ALL, "translate_up")]
    vocab = frozenset({"rot90", "translate_up"})
    assert trajectory_distinct(P, n_shapes=1) is True
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is True  # defers
    assert is_minimal_tier2(P, n_shapes=1, vocab=vocab) is False  # catches
    assert _final_state(P, 1) == _final_state(Q, 1)


def test_tier1_closes_d4_alternative_path():
    """Former Gap 1 — now caught by Tier 1 canonical-form analysis.
    [rot, rot, flip_h] = r²s is reachable in 1 step via [flip_v]."""
    from babytheorist.canonical import is_minimal_tier1
    P = [Step(ALL, "rot90"), Step(ALL, "rot90"), Step(ALL, "flip_h")]
    vocab_full = frozenset({"rot90", "flip_h", "flip_v"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab_full) is False


def test_tier1_closes_cross_dim_cycle():
    """Former Gap 3 — now caught by Tier 1. [hollow, rot, fill] = [rot]."""
    from babytheorist.canonical import is_minimal_tier1
    P = [Step(ALL, "hollow"), Step(ALL, "rot90"), Step(ALL, "fill_interior")]
    vocab = frozenset({"rot90", "hollow", "fill_interior"})
    assert is_minimal_tier1(P, n_shapes=1, vocab=vocab) is False


# ---------------------------------------------------------------------------
# Combined filter oracle: Rule 2 adds align-chain detection
# ---------------------------------------------------------------------------


def _combined_accept(
    P: list[Step], n_shapes: int, *, vocab: frozenset, grid_shapes=None,
) -> bool:
    """Full Phase-1 filter: trajectory_distinct ∧ no_dead_writes ∧ Tier 1."""
    if not trajectory_distinct(P, n_shapes, grid_shapes=grid_shapes):
        return False
    if not no_dead_writes(P, n_shapes, grid_shapes=grid_shapes):
        return False
    if not is_minimal_tier1(P, n_shapes, vocab=vocab, grid_shapes=grid_shapes):
        return False
    return True


def _enum_programs_relational(steps: list[Step], depth: int) -> list[list[Step]]:
    if depth == 0:
        return [[]]
    from itertools import product as _product
    return [list(c) for c in _product(steps, repeat=depth)]


def test_combined_soundness_align_chain():
    """Vocab {align_row, align_col} on 3 shapes. Rule 2 is the primary
    catcher here. Verify soundness: every rejection has a shorter
    equivalent program."""
    sels = [Selector("by_index", 1), Selector("by_index", 2), Selector("by_index", 3)]
    steps: list[Step] = []
    for p in ("align_row", "align_col"):
        for src in sels:
            for tgt in sels:
                steps.append(Step(src, p, target=tgt))
    vocab = frozenset({"align_row", "align_col"})

    max_depth = 3
    programs_by_depth = {
        d: _enum_programs_relational(steps, d) for d in range(max_depth + 1)
    }

    violations = []
    for d in range(1, max_depth + 1):
        shorter_finals = _collect_shorter_finals(programs_by_depth, 3, d)
        for P in programs_by_depth[d]:
            accepted = _combined_accept(P, 3, vocab=vocab)
            if accepted:
                continue
            final = _final_state(P, 3)
            oracle_rejects = final is None or final in shorter_finals
            if not oracle_rejects:
                violations.append((P, final))

    assert not violations, _format_violations("combined soundness", violations)


def test_rule2_catches_align_chain_oracle():
    """Verify the combined filter catches explicit align-chain cases
    that trajectory_distinct alone misses."""
    # [align(1,2), align(1,3)]: trajectory_distinct accepts but Rule 2 rejects.
    P = [
        Step(Selector("by_index", 1), "align_row", target=Selector("by_index", 2)),
        Step(Selector("by_index", 1), "align_row", target=Selector("by_index", 3)),
    ]
    assert trajectory_distinct(P, 3) is True
    assert no_dead_writes(P, 3) is False


def test_rule2_catches_translate_align_oracle():
    """[translate_up(A), align_row(A, B)] — translate dead."""
    P = [
        Step(Selector("by_index", 1), "translate_up"),
        Step(Selector("by_index", 1), "align_row", target=Selector("by_index", 2)),
    ]
    assert trajectory_distinct(P, 2) is True
    assert no_dead_writes(P, 2) is False
