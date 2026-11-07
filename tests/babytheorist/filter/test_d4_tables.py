"""Tests for babytheorist.filter.d4_tables."""

from __future__ import annotations

import pytest

from babytheorist.symbolic import D4, D4_IDENTITY
from babytheorist.filter.d4_tables import (
    D4_COMPOSE,
    D4_ELEMENTS,
    D4_INVERSE,
    D4_PRIMITIVE_TO_ELEMENT,
    D4_PRIMITIVE_VOCAB,
    D4_SUBGROUPS,
    D4_WORD_LENGTH,
    compose,
    inverse,
    min_coset_word_length,
    stabilizers_allowed_at_depth,
    stabilizers_forbidden_at_depth,
    word_length,
)


# === Basic identity checks ================================================


def test_eight_elements_unique():
    assert len(D4_ELEMENTS) == 8
    assert len(set(D4_ELEMENTS)) == 8
    assert D4_IDENTITY in D4_ELEMENTS


def test_primitive_mapping():
    assert D4_PRIMITIVE_TO_ELEMENT["rot90"] == D4(1, False)
    assert D4_PRIMITIVE_TO_ELEMENT["flip_h"] == D4(0, True)
    assert D4_PRIMITIVE_TO_ELEMENT["flip_v"] == D4(2, True)
    assert set(D4_PRIMITIVE_VOCAB) == {"rot90", "flip_h", "flip_v"}


# === Composition ==========================================================


def test_compose_identity():
    for g in D4_ELEMENTS:
        assert compose(D4_IDENTITY, g) == g
        assert compose(g, D4_IDENTITY) == g


def test_compose_known_cases():
    r = D4(1, False)
    s = D4(0, True)
    flip_v = D4(2, True)

    # r⁴ = e
    rr = compose(r, r)
    rrr = compose(r, rr)
    rrrr = compose(r, rrr)
    assert rrrr == D4_IDENTITY
    assert rr == D4(2, False)  # r²
    assert rrr == D4(3, False)  # r³

    # s² = e
    assert compose(s, s) == D4_IDENTITY

    # flip_v = r² · s  (i.e. compose(r², s) == flip_v)
    assert compose(rr, s) == flip_v

    # srs = r⁻¹
    r_inv = compose(s, compose(r, s))
    assert r_inv == D4(3, False)


def test_compose_non_commutative():
    r = D4(1, False)
    s = D4(0, True)
    # r · s ≠ s · r for generic D4 pair
    assert compose(r, s) != compose(s, r)


def test_compose_table_matches_function():
    for g in D4_ELEMENTS:
        for h in D4_ELEMENTS:
            assert D4_COMPOSE[(g, h)] == compose(g, h)


# === Inverse ==============================================================


def test_inverse_gives_identity():
    for g in D4_ELEMENTS:
        inv = inverse(g)
        assert compose(g, inv) == D4_IDENTITY
        assert compose(inv, g) == D4_IDENTITY


def test_inverse_table_matches_function():
    for g in D4_ELEMENTS:
        assert D4_INVERSE[g] == inverse(g)


def test_flipped_elements_self_inverse():
    # Every r^a · s has order 2.
    for rot in range(4):
        g = D4(rot, True)
        assert inverse(g) == g


# === Subgroup lattice =====================================================


def test_exactly_ten_subgroups():
    assert len(D4_SUBGROUPS) == 10


def test_every_subgroup_contains_identity():
    for H in D4_SUBGROUPS:
        assert D4_IDENTITY in H


def test_subgroup_closure():
    """Each D4_SUBGROUPS entry is closed under composition and inverse."""
    for H in D4_SUBGROUPS:
        for g in H:
            assert inverse(g) in H
            for h in H:
                assert compose(g, h) in H


def test_subgroup_sizes():
    sizes = sorted(len(H) for H in D4_SUBGROUPS)
    # 1 trivial + 5 order-2 + 3 order-4 + 1 full = 10
    assert sizes == [1, 2, 2, 2, 2, 2, 4, 4, 4, 8]


def test_subgroup_membership_specific():
    trivial = frozenset({D4_IDENTITY})
    assert trivial in D4_SUBGROUPS

    # Cyclic <r>
    r = D4(1, False)
    cyclic_r = frozenset({D4_IDENTITY, r, D4(2, False), D4(3, False)})
    assert cyclic_r in D4_SUBGROUPS

    # Klein of 180-rotation + horizontal/vertical flips
    klein_hv = frozenset({D4_IDENTITY, D4(2, False), D4(0, True), D4(2, True)})
    assert klein_hv in D4_SUBGROUPS

    # Full D4
    full = frozenset(D4_ELEMENTS)
    assert full in D4_SUBGROUPS


# === Word length ==========================================================


def test_identity_word_length_zero():
    assert D4_WORD_LENGTH[D4_IDENTITY] == 0
    assert word_length(D4_IDENTITY) == 0


def test_primitives_word_length_one():
    for name, g in D4_PRIMITIVE_TO_ELEMENT.items():
        assert D4_WORD_LENGTH[g] == 1, f"{name} should have word length 1"


def test_word_length_specific_elements():
    # Expected, from BFS over {rot90, flip_h, flip_v}:
    #   e:0, r:1, s:1, r²s:1, r²:2, rs:2, r³s:2, r³:3
    expected = {
        D4(0, False): 0,
        D4(1, False): 1,
        D4(2, False): 2,
        D4(3, False): 3,
        D4(0, True): 1,
        D4(1, True): 2,
        D4(2, True): 1,
        D4(3, True): 2,
    }
    assert D4_WORD_LENGTH == expected


def test_word_length_reachable_by_generators():
    # Every element reachable in word_length steps from e.
    from collections import deque
    gens = list(D4_PRIMITIVE_TO_ELEMENT.values())
    dist: dict[D4, int] = {D4_IDENTITY: 0}
    q = deque([D4_IDENTITY])
    while q:
        g = q.popleft()
        for p in gens:
            h = compose(p, g)
            if h not in dist:
                dist[h] = dist[g] + 1
                q.append(h)
    assert dist == D4_WORD_LENGTH


# === Coset min word length + allowed stabilizers ==========================


def test_min_coset_word_length_trivial_stabilizer():
    """With trivial Stab, coset = {g} so min = word_length(g)."""
    trivial = frozenset({D4_IDENTITY})
    for g in D4_ELEMENTS:
        assert min_coset_word_length(g, trivial) == D4_WORD_LENGTH[g]


def test_min_coset_word_length_full_stabilizer():
    """With full Stab = D4, coset = D4, min = 0 (contains identity)."""
    full = frozenset(D4_ELEMENTS)
    for g in D4_ELEMENTS:
        assert min_coset_word_length(g, full) == 0


def test_allowed_stabilizers_depth_zero_admits_all():
    """depth=0 is vacuously satisfied by every stabilizer."""
    for g in D4_ELEMENTS:
        assert stabilizers_allowed_at_depth(g, 0) == frozenset(D4_SUBGROUPS)


def test_allowed_stabilizers_depth_one():
    """At depth 1 (e.g. program = [rot90]), only Stabs with g ∉ Stab pass.

    If g ∈ Stab, g · Stab = Stab, which contains e (word length 0), so
    coset min = 0 < 1 → forbidden.
    """
    for g in D4_ELEMENTS:
        if g == D4_IDENTITY:
            continue  # depth 0 program, skip
        allowed = stabilizers_allowed_at_depth(g, 1)
        forbidden = stabilizers_forbidden_at_depth(g, 1)
        for H in D4_SUBGROUPS:
            if g in H:
                assert H in forbidden, f"H={H} contains g={g}, should be forbidden"
            else:
                assert H in allowed, f"H={H} doesn't contain g={g}, should be allowed"


def test_identity_net_effect_has_no_allowed_stabilizer_at_positive_depth():
    """Net = e, depth >= 1: e ∈ e·H = H always, so coset always
    contains identity (word length 0). All stabilizers forbidden —
    the program itself is dead on any shape. Caller (program-level
    R6) should have rejected it already.
    """
    for d in range(1, 5):
        assert stabilizers_allowed_at_depth(D4_IDENTITY, d) == frozenset()
        assert stabilizers_forbidden_at_depth(D4_IDENTITY, d) == frozenset(D4_SUBGROUPS)


def test_allowed_and_forbidden_disjoint_cover():
    for g in D4_ELEMENTS:
        for d in range(4):
            allowed = stabilizers_allowed_at_depth(g, d)
            forbidden = stabilizers_forbidden_at_depth(g, d)
            all_subs = frozenset(D4_SUBGROUPS)
            assert allowed | forbidden == all_subs
            assert allowed & forbidden == frozenset()


def test_allowed_monotonic_in_depth():
    """Higher depth requirement → fewer / equal allowed stabilizers."""
    for g in D4_ELEMENTS:
        prev = stabilizers_allowed_at_depth(g, 0)
        for d in range(1, 5):
            curr = stabilizers_allowed_at_depth(g, d)
            assert curr <= prev, f"depth {d} set larger than depth {d-1} for g={g}"
            prev = curr


def test_allowed_stabilizers_known_case():
    """r² (word length 2) coset on H = {e, r²}.

    g = r², H = {e, r²}, g·H = {r², r²·r²} = {r², e}. min word length 0.
    So H is forbidden at depth ≥ 1.
    """
    rr = D4(2, False)
    H = frozenset({D4_IDENTITY, rr})
    assert min_coset_word_length(rr, H) == 0
    assert H in stabilizers_forbidden_at_depth(rr, 1)


def test_allowed_stabilizers_r3_cubed_example():
    """Net = r³, depth 3. Word length of r³ is 3. Trivial Stab must pass."""
    rrr = D4(3, False)
    trivial = frozenset({D4_IDENTITY})
    assert D4_WORD_LENGTH[rrr] == 3
    # With trivial Stab, coset min = word length 3, so allowed at depth 3.
    assert trivial in stabilizers_allowed_at_depth(rrr, 3)
    # But forbidden at depth 4 (can't go higher).
    assert trivial in stabilizers_forbidden_at_depth(rrr, 4)


def test_depth_negative_rejected():
    with pytest.raises(ValueError):
        stabilizers_allowed_at_depth(D4_IDENTITY, -1)


# === Alternative vocab ====================================================


def test_word_length_alt_vocab():
    """With only rot90, every element without flip reachable, rest not."""
    # We don't generate D4 with only rot90 — should raise.
    with pytest.raises(ValueError):
        word_length(D4(0, True), vocab=frozenset({"rot90"}))


def test_word_length_vocab_rot_flip():
    """Vocab {rot90, flip_h} alone still generates D4 (flip_v = rot90² · flip_h).

    flip_v via {rot90, flip_h} needs rot90 · rot90 · flip_h = length 3.
    """
    vocab = frozenset({"rot90", "flip_h"})
    flip_v = D4(2, True)
    assert word_length(flip_v, vocab=vocab) == 3


def test_word_length_vocab_rot_flip_v_symmetric():
    """Symmetric mirror: {rot90, flip_v} should generate D4 and give
    flip_h (= s = (0,T)) length 3 via rot² · flip_v.

    Defends against asymmetric bugs in D4_PRIMITIVE_TO_ELEMENT.
    """
    vocab = frozenset({"rot90", "flip_v"})
    flip_h = D4(0, True)
    assert word_length(flip_h, vocab=vocab) == 3
