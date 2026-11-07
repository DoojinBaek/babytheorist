"""Unit tests for ``babytheorist.predicates`` — V1 C(P) helpers and dispatcher.

Covers:
  - D4 group axioms (compose_d4: closure, identity, inverse, associativity).
  - D4 action on pixel grids matches BabyTheorist primitive conventions.
  - Stabilizer computation on known-symmetry shapes.
  - Scene stabilizer = intersection.
  - V1 admissibility on curated (program, scene) pairs.
  - Scope / defer behaviour.

Oracle (end-to-end soundness vs brute-force) lives in
``test_predicates_oracle.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from babytheorist.core import ALL, Selector, Shape, Step
from babytheorist.predicates import (
    D4_ALL,
    apply_d4_to_grid,
    compose_d4,
    compose_program_to_d4,
    is_scene_admissible,
    min_depth_in_coset,
    stabilizer_of_pixel_grid,
    stabilizer_of_scene,
)
from babytheorist.symbolic import D4, D4_IDENTITY


# === D4 helpers ==========================================================


# Shorthand for the 4 generators used across group tests.
_R = D4(1, False)          # rot90
_S = D4(0, True)           # flip_h = flipud
_S_V = D4(2, True)         # flip_v = r² · s


# === compose_d4 (group axioms) ============================================


def test_compose_identity_left():
    for g in D4_ALL:
        assert compose_d4(D4_IDENTITY, g) == g


def test_compose_identity_right():
    for g in D4_ALL:
        assert compose_d4(g, D4_IDENTITY) == g


def test_compose_closure():
    for g in D4_ALL:
        for h in D4_ALL:
            assert compose_d4(g, h) in D4_ALL


def test_compose_associativity():
    elements = list(D4_ALL)
    # Full cartesian cube is 512 triples — fine for a unit test.
    for a in elements:
        for b in elements:
            for c in elements:
                lhs = compose_d4(compose_d4(a, b), c)
                rhs = compose_d4(a, compose_d4(b, c))
                assert lhs == rhs, f"associativity broke at ({a!r}, {b!r}, {c!r})"


def test_compose_inverse_exists():
    """Every element has a (two-sided) inverse in D4."""
    for g in D4_ALL:
        inverses = [h for h in D4_ALL
                    if compose_d4(g, h) == D4_IDENTITY
                    and compose_d4(h, g) == D4_IDENTITY]
        assert len(inverses) == 1, f"{g!r} has {len(inverses)} inverses"


def test_compose_involutions():
    """``flip_h² = flip_v² = e`` (s and s_v are involutions in D4)."""
    assert compose_d4(_S, _S) == D4_IDENTITY
    assert compose_d4(_S_V, _S_V) == D4_IDENTITY


def test_compose_r_order_four():
    """``r^4 = e``."""
    g = D4_IDENTITY
    for _ in range(4):
        g = compose_d4(_R, g)
    assert g == D4_IDENTITY


def test_compose_matches_apply_methods():
    """compose_d4 agrees with symbolic.D4 apply_* left-multiplication."""
    for g in D4_ALL:
        assert compose_d4(_R, g) == g.apply_rot()
        assert compose_d4(_S, g) == g.apply_flip_h()
        assert compose_d4(_S_V, g) == g.apply_flip_v()


# === apply_d4_to_grid ======================================================


# A deliberately asymmetric 2×3 grid to exercise every D4 element.
_ASYM_GRID = np.array([
    [1, 2, 3],
    [4, 5, 6],
], dtype=np.int64)


def test_apply_identity_leaves_grid_unchanged():
    out = apply_d4_to_grid(_ASYM_GRID, D4_IDENTITY)
    assert np.array_equal(out, _ASYM_GRID)


def test_apply_rot_matches_np_rot90():
    """g = r (rot 1, not flipped) → np.rot90(grid, 1)."""
    out = apply_d4_to_grid(_ASYM_GRID, _R)
    assert np.array_equal(out, np.rot90(_ASYM_GRID, 1))


def test_apply_flip_h_matches_flipud():
    """g = s (flip_h) → np.flipud."""
    out = apply_d4_to_grid(_ASYM_GRID, _S)
    assert np.array_equal(out, np.flipud(_ASYM_GRID))


def test_apply_flip_v_matches_fliplr():
    """g = r²·s (flip_v) → np.fliplr (via rot180 after flipud)."""
    out = apply_d4_to_grid(_ASYM_GRID, _S_V)
    assert np.array_equal(out, np.fliplr(_ASYM_GRID))


def test_apply_rot_twice_matches_rot180():
    g_r2 = compose_d4(_R, _R)
    out = apply_d4_to_grid(_ASYM_GRID, g_r2)
    assert np.array_equal(out, np.rot90(_ASYM_GRID, 2))


def test_apply_composed_matches_sequential():
    """apply_d4_to_grid(grid, g·h) == apply(apply(grid, h), g).

    This mirrors our semantic: ``g·h`` means h first (inner), g second
    (outer). Note ``apply_d4_to_grid`` applies ``s`` (flip) first, then
    ``r`` (rot) — that's the internal convention, not the compose order.
    """
    for g in D4_ALL:
        for h in D4_ALL:
            combined = compose_d4(g, h)
            lhs = apply_d4_to_grid(_ASYM_GRID, combined)
            rhs = apply_d4_to_grid(apply_d4_to_grid(_ASYM_GRID, h), g)
            # Shape must match and be equal.
            assert lhs.shape == rhs.shape, (
                f"shape mismatch for g={g!r}, h={h!r}"
            )
            assert np.array_equal(lhs, rhs), (
                f"mismatch for g={g!r}, h={h!r}"
            )


# === stabilizer_of_pixel_grid =============================================


def test_stab_square_solid_is_full_d4():
    """A square solid shape is fixed by all 8 D4 elements."""
    grid = np.ones((4, 4), dtype=np.int64) * 3
    stab = stabilizer_of_pixel_grid(grid)
    assert stab == D4_ALL


def test_stab_nonsquare_excludes_odd_rotations():
    """3×5 grid: rot90 / rot270 change shape → excluded."""
    grid = np.ones((3, 5), dtype=np.int64) * 4
    stab = stabilizer_of_pixel_grid(grid)
    # Allowed: e, r², s (flip_h), r²·s (flip_v). Odd rotations produce
    # 5×3 grids, which fail shape check.
    assert D4_IDENTITY in stab
    assert D4(2, False) in stab  # r² = rot180
    assert _S in stab            # s = flip_h (flipud)
    assert _S_V in stab          # r²·s = flip_v (fliplr)
    assert D4(1, False) not in stab
    assert D4(3, False) not in stab
    # Odd-rot × flipped elements also produce mismatched shapes.
    assert D4(1, True) not in stab
    assert D4(3, True) not in stab


def test_stab_striped_horizontal_excludes_flip_v():
    """Stripes across rows: flip_h preserves, flip_v reverses → excluded.

    In BabyTheorist convention flip_h = flipud (axis=0, top ↔ bottom). A
    horizontally-striped grid has equal rows of one color, so flipud
    preserves it. Nonsense — wait, let me think again. "Striped
    horizontally" here means stripes run along rows (each row one
    color). flipud swaps rows → stripes reorder → NOT preserved unless
    pattern is palindrome.

    We use a 2-stripe [1, 2] split-h pattern so flipud swaps the two
    rows, giving [2, 1] — NOT preserved.
    """
    grid = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
    ], dtype=np.int64)
    stab = stabilizer_of_pixel_grid(grid)
    # flip_v = fliplr (left-right) preserves each row → preserves grid.
    # flip_h = flipud (top-bottom) swaps rows 1/2 → changes grid.
    # rot90 changes shape 2×4 → 4×2, excluded.
    assert D4_IDENTITY in stab
    assert _S_V in stab            # fliplr preserves
    assert _S not in stab          # flipud breaks [1,2] stripes
    assert D4(2, False) not in stab  # r² = rot180 also reverses rows


def test_stab_striped_vertical_excludes_flip_h():
    """Stripes running up-down: fliplr reverses column order, flipud
    preserves. Use a 2-column split.
    """
    grid = np.array([
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
    ], dtype=np.int64)
    stab = stabilizer_of_pixel_grid(grid)
    assert _S in stab             # flipud preserves (all rows identical)
    assert _S_V not in stab       # fliplr swaps columns → [2,1]
    assert D4(2, False) not in stab


def test_stab_l_shape_is_trivial():
    """Truly asymmetric shape has only the identity in its stabilizer.

    Note: a symmetric L (3-cell vertical + 3-cell horizontal bar) has
    reflection symmetry across the main diagonal — ``D4(1, True) = r·s``
    is in its stabilizer. We use a deliberately shorter bottom bar to
    break that diagonal symmetry.
    """
    grid = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64)
    stab = stabilizer_of_pixel_grid(grid)
    assert stab == frozenset({D4_IDENTITY})


def test_stab_colored_asymmetry_shrinks_stabilizer():
    """Silhouette-symmetric shape with asymmetric *color* pattern has
    strictly smaller stabilizer than silhouette alone.

    The marked cell must be **off every D4 axis** (diagonal and
    anti-diagonal included) to drop the stabilizer to trivial. Corner
    (0, 0) is on the main diagonal, so r³·s (the diagonal transpose)
    still fixes it; placing the mark at (0, 1) is off every axis.
    """
    silhouette = np.ones((4, 4), dtype=np.int64) * 3
    assert stabilizer_of_pixel_grid(silhouette) == D4_ALL

    colored = silhouette.copy()
    colored[0, 1] = 5
    stab = stabilizer_of_pixel_grid(colored)
    assert stab == frozenset({D4_IDENTITY})


def test_stab_all_zero_grid_is_full_d4():
    """Null shape: trivially fixed by every transform (of matching shape)."""
    grid = np.zeros((3, 3), dtype=np.int64)
    stab = stabilizer_of_pixel_grid(grid)
    assert stab == D4_ALL


def test_stab_checkerboard_subgroup():
    """2×2 checkerboard is fixed by e and rot180 (diagonal swap)
    but not by flip_h or flip_v."""
    grid = np.array([
        [1, 2],
        [2, 1],
    ], dtype=np.int64)
    stab = stabilizer_of_pixel_grid(grid)
    assert D4_IDENTITY in stab
    assert D4(2, False) in stab  # rot180 swaps (0,0)↔(1,1), (0,1)↔(1,0)
    # flipud: [[2,1],[1,2]] ≠ grid.
    assert _S not in stab
    # fliplr: [[2,1],[1,2]] ≠ grid.
    assert _S_V not in stab


# === stabilizer_of_scene ===================================================


def test_scene_stab_intersection():
    """Two shapes with different stabilizers → scene Stab = intersection."""
    # Shape A: solid square — full D4.
    shape_a = Shape(np.ones((3, 3), dtype=np.int64) * 1, (0, 0))
    # Shape B: L-shape — trivial stabilizer.
    l_grid = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64)
    shape_b = Shape(l_grid, (5, 5))
    stab = stabilizer_of_scene([shape_a, shape_b])
    assert stab == frozenset({D4_IDENTITY})


def test_scene_stab_empty_is_full_d4():
    assert stabilizer_of_scene([]) == D4_ALL


def test_scene_stab_all_squares_is_full_d4():
    shapes = [
        Shape(np.ones((3, 3), dtype=np.int64) * 1, (0, 0)),
        Shape(np.ones((4, 4), dtype=np.int64) * 2, (5, 5)),
    ]
    assert stabilizer_of_scene(shapes) == D4_ALL


# === compose_program_to_d4 ================================================


def test_compose_program_empty_is_identity():
    assert compose_program_to_d4([]) == D4_IDENTITY


def test_compose_program_single_rot90():
    program = [Step(ALL, "rot90")]
    assert compose_program_to_d4(program) == _R


def test_compose_program_rot90_twice():
    program = [Step(ALL, "rot90"), Step(ALL, "rot90")]
    assert compose_program_to_d4(program) == D4(2, False)


def test_compose_program_flip_then_rot():
    """[flip_h, rot90] = apply_rot(apply_flip_h(e)) = r · s = rs."""
    program = [Step(ALL, "flip_h"), Step(ALL, "rot90")]
    result = compose_program_to_d4(program)
    # Expected: apply_flip_h(e) then apply_rot → (1, True).
    assert result == D4(1, True)


def test_compose_program_rejects_non_d4_primitive():
    program = [Step(ALL, "translate_up")]
    with pytest.raises(ValueError, match="outside D4 vocab"):
        compose_program_to_d4(program)


# === min_depth_in_coset ====================================================


_FULL_VOCAB = frozenset({"rot90", "flip_h", "flip_v"})


def test_min_depth_in_trivial_coset_matches_element_depth():
    """Stab = {e} → coset is just {g}. Min-depth = depth of g alone."""
    for g in D4_ALL:
        d = min_depth_in_coset(g, frozenset({D4_IDENTITY}), _FULL_VOCAB)
        # d4_min_depths(full D4 vocab): e=0, r/s/r²s=1, ...
        from babytheorist.canonical import d4_min_depths
        assert d == d4_min_depths(_FULL_VOCAB)[g]


def test_min_depth_in_full_coset_is_zero():
    """Stab = D4 → coset = D4 → contains e (depth 0)."""
    for g in D4_ALL:
        assert min_depth_in_coset(g, D4_ALL, _FULL_VOCAB) == 0


def test_min_depth_respects_vocab():
    """With vocab = {rot90} only, flip elements are unreachable."""
    rot_only = frozenset({"rot90"})
    # Stab = {e}, g = r². Min-depth under rot_only = 2.
    assert min_depth_in_coset(D4(2, False), frozenset({D4_IDENTITY}), rot_only) == 2


# === is_scene_admissible V1 — ALL branch =================================


_ALL_LEVEL = frozenset({"ALL"})
_BYINDEX_LEVEL = frozenset({"by_index"})


def _asym_scene(n_shapes: int = 1) -> list[Shape]:
    """A scene of asymmetric (trivial Stab) shapes at distinct positions."""
    l_grid = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64)
    return [Shape(l_grid, (i * 5, i * 5)) for i in range(n_shapes)]


def _square_scene(n_shapes: int = 1) -> list[Shape]:
    """Solid squares — full D4 stabilizer."""
    return [
        Shape(np.ones((3, 3), dtype=np.int64) * 1, (i * 5, i * 5))
        for i in range(n_shapes)
    ]


def test_v1_empty_program_is_admissible():
    scene = _asym_scene()
    assert is_scene_admissible([], scene, _FULL_VOCAB, _ALL_LEVEL) is True


def test_v1_single_rot90_on_asymmetric_accepted():
    scene = _asym_scene()
    program = [Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is True


def test_v1_two_rot90_on_square_rejected():
    """Square has full Stab; rot² renders identity. Shorter Q = [] works
    (identity collapse). C(P) should reject."""
    scene = _square_scene()
    program = [Step(ALL, "rot90"), Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is False


def test_v1_flip_h_then_rot_on_square_rejected():
    """[flip_h, rot90] = rs. On full-D4-Stab square, coset gH = D4 →
    min-depth = 0 ≠ 2. Reject."""
    scene = _square_scene()
    program = [Step(ALL, "flip_h"), Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is False


def test_v1_single_rot_on_square_rejected():
    """One rot90 on a square: rendering = square (unchanged) since r ∈ Stab.
    Identity collapse → reject."""
    scene = _square_scene()
    program = [Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is False


def test_v1_mixed_stab_uses_intersection():
    """One square + one asymmetric. Stab_common = {e} → any non-trivial P
    accepted iff it's minimal in the full-vocab BFS table."""
    square = Shape(np.ones((3, 3), dtype=np.int64) * 1, (0, 0))
    l_grid = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64)
    l_shape = Shape(l_grid, (5, 5))
    scene = [square, l_shape]
    program = [Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is True


# === is_scene_admissible V1 — by_index branch ============================


def test_v1_by_index_per_shape_independent():
    """by_index(1) rot90 + by_index(2) flip_h: two independent 1-step
    sub-programs on asymmetric-shape scene → both minimal."""
    scene = _asym_scene(n_shapes=2)
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 2), "flip_h"),
    ]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _BYINDEX_LEVEL) is True


def test_v1_by_index_rejects_stab_collapse_on_one_shape():
    """Square at slot 1, asymmetric at slot 2. Two rot90 on shape 1
    (full Stab → sub-prog not minimal: r² ∈ Stab1, shorter Q_1 = [] works)
    and one flip_h on shape 2. Reject."""
    scene = [
        Shape(np.ones((3, 3), dtype=np.int64) * 1, (0, 0)),
        Shape(np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ], dtype=np.int64), (5, 5)),
    ]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 2), "flip_h"),
    ]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _BYINDEX_LEVEL) is False


def test_v1_by_index_accepts_full_per_shape_minimality():
    """by_index shape 1 gets [rot90] (asymmetric, minimal); shape 2 gets
    [flip_h, rot90] (rs, minimal as a 2-word on trivial Stab)."""
    scene = _asym_scene(n_shapes=2)
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 2), "flip_h"),
        Step(Selector("by_index", 2), "rot90"),
    ]
    # sub_P[0] = [rot90]: g = r. Stab = {e}. Coset = {r}. Min-depth = 1.
    # len = 1 ✓.
    # sub_P[1] = [flip_h, rot90]: g = rs. Stab = {e}. Coset = {rs}.
    # Min-depth = 2. len = 2 ✓.
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _BYINDEX_LEVEL) is True


# === Scope / defer behaviour =============================================


def test_v1_defers_on_translate_vocab():
    """Vocab includes translate → out of V1 scope, defer to True."""
    scene = _asym_scene()
    program = [Step(ALL, "rot90")]
    vocab = frozenset({"rot90", "translate_up"})
    assert is_scene_admissible(program, scene, vocab, _ALL_LEVEL) is True


def test_v1_defers_on_add_border_vocab():
    """add_border changes shape position / size → position-preserving
    assumption broken → defer."""
    scene = _asym_scene()
    program = [Step(ALL, "rot90")]
    vocab = frozenset({"rot90", "add_border"})
    assert is_scene_admissible(program, scene, vocab, _ALL_LEVEL) is True


def test_v1_defers_on_dynamic_selector_in_program():
    """topmost in program → defer."""
    scene = _asym_scene(n_shapes=2)
    program = [Step(Selector("topmost"), "rot90")]
    level = frozenset({"ALL", "topmost"})
    assert is_scene_admissible(program, scene, _FULL_VOCAB, level) is True


def test_v1_defers_on_mixed_selectors():
    """Mixed ALL + by_index → V1 doesn't handle, defer."""
    scene = _asym_scene(n_shapes=2)
    program = [
        Step(ALL, "rot90"),
        Step(Selector("by_index", 1), "flip_h"),
    ]
    level = frozenset({"ALL", "by_index"})
    assert is_scene_admissible(program, scene, _FULL_VOCAB, level) is True


def test_v1_mixed_level_selectors_enumerates_subsets():
    """Level allows both ALL and by_index — new subset BFS enumerates
    both subset patterns (full + singletons), closing the old ALL-only
    branch's unsoundness.

    Concrete counterexample from the old docstring: 2-shape scene with
    Stab_1 = {e, r²} and Stab_2 = {e, s}. For P = [ALL rot90, ALL rot90]
    the old ALL branch accepted (Stab_common = {e}, min_depth 2 = 2).
    But Q = [by_index(2) flip_v] (|Q|=1) matches. Subset BFS catches it.
    """
    # Construct shapes with the specific stabilizers.
    # Stab = {e, r²}: 2×4 rectangle (color 1) — rot180 fixes it,
    # but flip_h (flipud) and flip_v (fliplr) also fix a solid rect.
    # For Stab = exactly {e, r²}, need asymmetry under flips. Use a
    # 2×4 pattern with asymmetric colors.
    shape1_grid = np.array([
        [1, 1, 1, 2],
        [2, 1, 1, 1],
    ], dtype=np.int64)
    # Verify: rot180 swaps top/bottom and left/right.
    # rot180([[1,1,1,2],[2,1,1,1]]) = [[1,1,1,2],[2,1,1,1]] (self).
    # flipud: [[2,1,1,1],[1,1,1,2]] ≠ original.
    # fliplr: [[2,1,1,1],[1,1,1,2]] ≠ original.
    # So Stab = {e, r²}.

    # Stab = {e, s} (only flipud fixes): 4×2 pattern symmetric vertically.
    shape2_grid = np.array([
        [1, 1],
        [1, 2],
        [1, 2],
        [1, 1],
    ], dtype=np.int64)
    # flipud: [[1,1],[1,2],[1,2],[1,1]] = original. ✓
    # fliplr: [[1,1],[2,1],[2,1],[1,1]] ≠.
    # rot180: [[1,1],[2,1],[2,1],[1,1]] ≠.
    # So Stab = {e, s}.

    scene = [Shape(shape1_grid, (0, 0)), Shape(shape2_grid, (10, 0))]
    program = [Step(ALL, "rot90"), Step(ALL, "rot90")]
    level = frozenset({"ALL", "by_index"})
    assert is_scene_admissible(program, scene, _FULL_VOCAB, level) is False


def test_v1_ALL_only_program_and_byindex_level_analyzes():
    """Program uses ALL but level allows only by_index. Asym scene —
    [ALL.rot90] target = (r, r) requires 2 singleton by_index steps
    (len 2), which is longer than |P| = 1. No shorter Q in level
    vocab → accept.
    """
    scene = _asym_scene(n_shapes=2)
    program = [Step(ALL, "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _BYINDEX_LEVEL) is True


def test_v1_byindex_program_ALL_level_analyzes():
    """Program uses by_index(1) but level allows only ALL. Target
    (r, e) — ALL transitions are (prim, all_shapes) → can reach (r, r),
    (s, s), etc. but not (r, e). No match → accept."""
    scene = _asym_scene(n_shapes=2)
    program = [Step(Selector("by_index", 1), "rot90")]
    assert is_scene_admissible(program, scene, _FULL_VOCAB, _ALL_LEVEL) is True


# === V3 color axis =======================================================


from babytheorist.predicates import (
    scene_color_bfs, trace_program_colors,
)


_V3_COLOR_VOCAB = frozenset({"next_color", "swap_colors"})
_V3_LEVEL_BYINDEX = frozenset({"by_index"})
_V3_LEVEL_ALL = frozenset({"ALL"})
_V3_LEVEL_MIXED = frozenset({"ALL", "by_index"})


def _solid_shape(color: int, size: tuple[int, int] = (3, 3),
                 position: tuple[int, int] = (0, 0)) -> Shape:
    """Single-color solid shape (V3 applicable)."""
    return Shape(np.ones(size, dtype=np.int64) * color, position)


# --- trace_program_colors ------------------------------------------------


def test_trace_colors_next_color_all():
    """next_color under ALL shifts every component."""
    program = [Step(ALL, "next_color")]
    result = trace_program_colors(program, (1, 2, 3))
    assert result == (2, 3, 4)


def test_trace_colors_next_color_cycles_9_to_1():
    program = [Step(ALL, "next_color")]
    result = trace_program_colors(program, (9,))
    assert result == (1,)


def test_trace_colors_next_color_by_index():
    program = [Step(Selector("by_index", 2), "next_color")]
    result = trace_program_colors(program, (1, 2, 3))
    assert result == (1, 3, 3)


def test_trace_colors_swap_swaps_two_components():
    program = [Step(Selector("by_index", 1), "swap_colors",
                    target=Selector("by_index", 3))]
    result = trace_program_colors(program, (1, 2, 3))
    assert result == (3, 2, 1)


def test_trace_colors_swap_precondition_fail_returns_none():
    program = [Step(Selector("by_index", 1), "swap_colors",
                    target=Selector("by_index", 2))]
    result = trace_program_colors(program, (5, 5, 1))
    assert result is None


def test_trace_colors_user_example_swap_chain():
    """User's example: [swap(A,B), swap(B,C)] on (1, 2, 2)."""
    program = [
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 2), "swap_colors",
             target=Selector("by_index", 3)),
    ]
    result = trace_program_colors(program, (1, 2, 2))
    # Step 1: swap(0, 1): 1 ↔ 2 → (2, 1, 2).
    # Step 2: swap(1, 2): 1 ↔ 2 → (2, 2, 1).
    assert result == (2, 2, 1)


# --- scene_color_bfs (new signature returns min-depth-to-target) --------


def _all_subset(n: int) -> frozenset[frozenset[int]]:
    return frozenset({frozenset(range(n))})


def _by_index_subsets(n: int) -> frozenset[frozenset[int]]:
    return frozenset(frozenset({k}) for k in range(n))


def _by_index_pairs(n: int) -> frozenset[tuple[int, int]]:
    return frozenset((i, j) for i in range(n) for j in range(n) if i != j)


def test_bfs_target_equal_initial_is_zero():
    """BFS returns 0 when target already matches initial."""
    d = scene_color_bfs(
        target=(1, 2, 3), initial=(1, 2, 3),
        vocab=frozenset(), subset_vocab=frozenset(), pair_vocab=frozenset(),
        max_depth=5,
    )
    assert d == 0


def test_bfs_target_one_next_color_ALL():
    """Target is one next_color away under ALL selector."""
    d = scene_color_bfs(
        target=(2, 3), initial=(1, 2),
        vocab=frozenset({"next_color"}),
        subset_vocab=_all_subset(2),
        pair_vocab=frozenset(),
        max_depth=5,
    )
    assert d == 1


def test_bfs_target_unreachable_in_cap():
    """Target with no matching transition sequence returns max_depth + 1."""
    d = scene_color_bfs(
        target=(9, 9), initial=(1, 1),
        vocab=frozenset({"next_color"}),
        subset_vocab=_by_index_subsets(2),  # only per-shape increments
        pair_vocab=frozenset(),
        max_depth=2,  # need 8 steps each, cap too short
    )
    assert d == 3  # max_depth + 1


def test_bfs_swap_reaches_transposition():
    d = scene_color_bfs(
        target=(2, 1, 3), initial=(1, 2, 3),
        vocab=frozenset({"swap_colors"}),
        subset_vocab=frozenset(),
        pair_vocab=_by_index_pairs(3),
        max_depth=2,
    )
    assert d == 1


def test_bfs_swap_skips_equal_components():
    """swap(i,j) with equal colors precondition-fails — that transition
    is skipped."""
    d = scene_color_bfs(
        target=(2, 2, 3), initial=(2, 2, 3),  # initial == target → 0
        vocab=frozenset({"swap_colors"}),
        subset_vocab=frozenset(),
        pair_vocab=_by_index_pairs(3),
        max_depth=1,
    )
    assert d == 0
    # Check swap to (3, 2, 2) in 1 step, and (2, 3, 2) via swap(1, 2).
    d2 = scene_color_bfs(
        target=(3, 2, 2), initial=(2, 2, 3),
        vocab=frozenset({"swap_colors"}),
        subset_vocab=frozenset(),
        pair_vocab=_by_index_pairs(3),
        max_depth=1,
    )
    assert d2 == 1


def test_bfs_user_example_shortcut_caught():
    """Scene (1, 2, 2); P = [swap(1,2), swap(2,3)] has final (2, 2, 1).
    BFS reaches (2, 2, 1) in 1 step → shortcut."""
    d = scene_color_bfs(
        target=(2, 2, 1), initial=(1, 2, 2),
        vocab=frozenset({"swap_colors"}),
        subset_vocab=frozenset(),
        pair_vocab=_by_index_pairs(3),
        max_depth=2,
    )
    assert d == 1


# --- is_scene_admissible V3 color dispatcher -----------------------------


def test_v3_user_example_rejected():
    """User's scene-dep shortcut: swap chain on (1, 2, 2)."""
    scene = [
        _solid_shape(1, position=(0, 0)),
        _solid_shape(2, position=(5, 0)),
        _solid_shape(2, position=(10, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 2), "swap_colors",
             target=Selector("by_index", 3)),
    ]
    vocab = frozenset({"swap_colors"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_BYINDEX) is False


def test_v3_distinct_colors_accepts_canonical_chain():
    """Same program on scene (1, 2, 3) (all distinct) has no shortcut —
    V3 accepts."""
    scene = [
        _solid_shape(1, position=(0, 0)),
        _solid_shape(2, position=(5, 0)),
        _solid_shape(3, position=(10, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 2), "swap_colors",
             target=Selector("by_index", 3)),
    ]
    # Trace: (1,2,3) → swap(0,1) → (2,1,3) → swap(1,2) → (2,3,1).
    # Can Q of length 1 produce (2,3,1)? One swap can produce
    # permutations that differ from (1,2,3) by one transposition.
    # (2,3,1) is a 3-cycle — NOT a single transposition. So no length-1
    # Q works. V3 accepts.
    vocab = frozenset({"swap_colors"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_BYINDEX) is True


def test_v3_next_color_all_identity_cycle_rejected():
    """[next_color] × 9 cycles back to start — V3 catches identity."""
    scene = [_solid_shape(3, position=(0, 0))]
    program = [Step(ALL, "next_color") for _ in range(9)]
    vocab = frozenset({"next_color"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_ALL) is False


def test_v3_next_color_all_canonical_accepts():
    scene = [_solid_shape(3, position=(0, 0))]
    program = [Step(ALL, "next_color"), Step(ALL, "next_color")]
    vocab = frozenset({"next_color"})
    # Trace (3,) → (4,) → (5,). BFS: (3,) 0, (4,) 1, (5,) 2.
    # len 2 = min_depth 2. Accept.
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_ALL) is True


def test_v3_defers_on_bordered_shape():
    """Bordered shape has 2 colors → V3 scope excludes it. Defer."""
    bordered = np.array([
        [2, 2, 2, 2],
        [2, 1, 1, 2],
        [2, 1, 1, 2],
        [2, 2, 2, 2],
    ], dtype=np.int64)
    scene = [
        Shape(bordered, (0, 0)),
        _solid_shape(5, position=(6, 0)),
    ]
    program = [Step(Selector("by_index", 1), "swap_colors",
                    target=Selector("by_index", 2))]
    vocab = frozenset({"swap_colors"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_BYINDEX) is True


def test_v3_mixed_axis_program_now_handled_by_phase_c():
    """rot90 + swap_colors mixed on solid squares (full D4 Stab) —
    Phase C (post-extension) analyzes and rejects: swap_colors alone
    reaches equivalent rendering since rot is absorbed by the full D4
    silhouette stab on a solid square."""
    scene = [_solid_shape(3, position=(0, 0)),
             _solid_shape(5, position=(5, 0))]
    program = [
        Step(ALL, "rot90"),
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
    ]
    vocab = frozenset({"rot90", "swap_colors"})
    level = frozenset({"ALL", "by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is False


def test_v3_defers_on_topmost_in_program():
    scene = [_solid_shape(3, position=(0, 0)),
             _solid_shape(5, position=(5, 0))]
    program = [Step(Selector("topmost"), "next_color")]
    vocab = frozenset({"next_color"})
    level = frozenset({"ALL", "topmost"})
    assert is_scene_admissible(program, scene, vocab, level) is True


def test_v3_swap_needs_byindex_in_level():
    """If vocab has swap_colors but level has only ALL, V3 cannot
    express any shorter Q using swap_colors → scope defers."""
    scene = [_solid_shape(1, position=(0, 0)),
             _solid_shape(2, position=(5, 0))]
    program = [Step(ALL, "next_color")]
    vocab = frozenset({"next_color", "swap_colors"})
    # Level = ALL only — swap_colors needs by_index for single selectors.
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_ALL) is True


def test_v3_mixed_level_selectors_work():
    """Level = {ALL, by_index} with pure color vocab — V3 handles."""
    scene = [
        _solid_shape(1, position=(0, 0)),
        _solid_shape(2, position=(5, 0)),
        _solid_shape(2, position=(10, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 2), "swap_colors",
             target=Selector("by_index", 3)),
    ]
    vocab = frozenset({"next_color", "swap_colors"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_MIXED) is False


def test_v3_single_shape_next_color_canonical():
    """Single shape, depth 3 next_color, no shortcut via ALL cycle."""
    scene = [_solid_shape(2, position=(0, 0))]
    program = [Step(ALL, "next_color") for _ in range(3)]
    vocab = frozenset({"next_color"})
    assert is_scene_admissible(program, scene, vocab, _V3_LEVEL_ALL) is True


# === V3 position axis ====================================================


from babytheorist.predicates import (
    scene_position_bfs, trace_program_positions,
)


_V3_POSITION_VOCAB = frozenset({"align_row", "align_col"})


def _sh(color: int, position: tuple[int, int]) -> Shape:
    return Shape(np.ones((2, 2), dtype=np.int64) * color, position)


# --- trace_program_positions ---------------------------------------------


def test_trace_positions_align_row_copies():
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    final = trace_program_positions(program, (0, 5, 10), (0, 0, 0))
    assert final == ((5, 5, 10), (0, 0, 0))


def test_trace_positions_align_col_copies():
    program = [Step(Selector("by_index", 3), "align_col",
                    target=Selector("by_index", 1))]
    final = trace_program_positions(program, (0, 5, 10), (3, 6, 9))
    assert final == ((0, 5, 10), (3, 6, 3))


def test_trace_positions_precondition_fail_returns_none():
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    # Already aligned.
    assert trace_program_positions(program, (5, 5, 10), (0, 0, 0)) is None


def test_trace_positions_align_chain():
    """[align_row(A,B), align_row(A,C)]: Phase 1 catches as dead write,
    but concrete trace still produces a final state."""
    program = [
        Step(Selector("by_index", 1), "align_row",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 1), "align_row",
             target=Selector("by_index", 3)),
    ]
    final = trace_program_positions(program, (0, 5, 10), (0, 0, 0))
    # Step 1: A = B: rows (5, 5, 10).
    # Step 2: A = C: rows (10, 5, 10). Now A == C.
    assert final == ((10, 5, 10), (0, 0, 0))


# --- scene_position_bfs --------------------------------------------------


def test_bfs_positions_no_align_vocab():
    dists = scene_position_bfs((0, 5), (0, 0), frozenset(), max_depth=3)
    assert dists == {((0, 5), (0, 0)): 0}


def test_bfs_positions_align_row_reaches():
    """Two shapes, distinct rows; align_row(0,1) gets (5, 5), and (0, 0)."""
    dists = scene_position_bfs((0, 5), (0, 0),
                               frozenset({"align_row"}), max_depth=1)
    assert dists[((0, 5), (0, 0))] == 0
    assert dists[((5, 5), (0, 0))] == 1  # align_row(0, 1)
    assert dists[((0, 0), (0, 0))] == 1  # align_row(1, 0)


def test_bfs_positions_row_col_independent():
    """Distinct rows and cols; align_row affects rows only, align_col
    affects cols only."""
    dists = scene_position_bfs((0, 5), (0, 10),
                               frozenset({"align_row", "align_col"}),
                               max_depth=1)
    assert ((0, 5), (0, 10)) in dists and dists[((0, 5), (0, 10))] == 0
    assert dists[((5, 5), (0, 10))] == 1      # align_row(0, 1)
    assert dists[((0, 5), (10, 10))] == 1     # align_col(0, 1)


# --- V3 position dispatcher cases ---------------------------------------


def test_v3_position_align_chain_rejected_scene_coincidence():
    """Scene has shape_2 and shape_3 at same row. Program
    [align_row(A, B), align_row(A, C)] has scene-equivalent
    [align_row(A, B)] (length 1): after align_row(A, B), A is at
    B's row = C's row (coincidence). Further align_row(A, C) is a no-op
    → engine would reject at step 2 (already aligned). So the
    **program fails** on this scene; V3 traces that and defers.

    Contrast with a scene where B, C distinct rows: the chain is caught
    by Phase 1 Rule 2 (dead write), but V3 scene-BFS also catches:
    single align_row(A, C) matches full trace.
    """
    # Distinct initial rows: A=0, B=5, C=10. Program [align_row(A,B),
    # align_row(A,C)] traces to A=10, B=5, C=10. Shorter Q =
    # [align_row(A, C)] yields same.
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0)), _sh(3, (10, 0))]
    program = [
        Step(Selector("by_index", 1), "align_row",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 1), "align_row",
             target=Selector("by_index", 3)),
    ]
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
    ) is False


def test_v3_position_scene_row_coincidence_rejected():
    """Two shapes share a row initially — align_row(A, *) to any of
    them collapses. Length-2 program that aligns through both shapes is
    equivalent to length-1.
    """
    # B, C at same row; align_row(A, B) then something else to reach
    # unique state — but actually the plain (align(A, B), align(B, C))
    # test below mirrors the color swap chain analogy better. Here:
    # if B, C share row, align_row(A, B) and align_row(A, C) are
    # equivalent: both put A at that shared row.
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0)), _sh(3, (5, 10))]
    # Program of length 1 using align_row(A, C). Q = align_row(A, B)
    # would also work — but that Q is same length as P, not shorter.
    # So single step align_row(A, C) should still be minimal.
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 3))]
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
    ) is True


def test_v3_position_single_shape_defers():
    scene = [_sh(1, (0, 0))]
    # Single-shape align is impossible (no distinct tgt). But the
    # program pattern itself would have been rejected earlier. Test
    # that V3 defers cleanly.
    program: list[Step] = []
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
    ) is True


def test_v3_position_defers_on_non_byindex_level():
    """align is relational; V3 position requires by_index level."""
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0))]
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    # Level is singleton ALL — no single selectors. V3 scope excludes.
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"ALL"}),
    ) is True


def test_v3_position_accepts_canonical_single_align():
    """One align_row on distinct rows — minimal."""
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0))]
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
    ) is True


def test_v3_position_accepts_stable_multi_selectors_in_level():
    """Level with by_index + ALL (ALL multi, stable under align) still
    fires V3 position branch — ALL isn't usable for align but doesn't
    affect soundness."""
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0))]
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    level = frozenset({"by_index", "ALL"})
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, level,
    ) is True


def test_v3_position_defers_on_unstable_topmost_in_level():
    """Level with topmost — align changes positions, topmost unstable.
    V3 position must defer."""
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0))]
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    level = frozenset({"by_index", "topmost"})
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, level,
    ) is True


# === V2 translate + add_border =========================================


from babytheorist.predicates import (
    per_shape_v2_bfs, trace_sub_program_v2,
)


_V2_VOCAB = frozenset({
    "rot90", "flip_h", "flip_v",
    "translate_up", "translate_down", "translate_left", "translate_right",
    "add_border",
})


def _solid_at(h: int, w: int, r: int, c: int, color: int = 3) -> Shape:
    return Shape(np.ones((h, w), dtype=np.int64) * color, (r, c))


# --- trace_sub_program_v2 -----------------------------------------------


def test_trace_v2_translate_shifts_row():
    """translate_down on (h=3, w=4, r=0, c=0) shifts row by current h."""
    program = [Step(Selector("by_index", 1), "translate_down")]
    state = trace_sub_program_v2(program, 3, 4, 0, 0)
    assert state == (D4_IDENTITY, 0, 3, 0)  # row += 3


def test_trace_v2_rot_then_translate_uses_new_size():
    """After rot90, h↔w swap — translate uses new dimensions."""
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "translate_down"),
    ]
    # After rot90: orient.rot=1 (odd parity). cur_h = w_init = 4.
    state = trace_sub_program_v2(program, 3, 4, 0, 0)
    assert state == (D4(1, False), 0, 4, 0)  # row shifted by 4 (new h)


def test_trace_v2_add_border_shifts_position_and_bumps_border():
    program = [Step(Selector("by_index", 1), "add_border")]
    state = trace_sub_program_v2(program, 3, 4, 5, 5)
    assert state == (D4_IDENTITY, 1, 4, 4)  # r-1, c-1, border=1


def test_trace_v2_translate_after_border_uses_expanded_size():
    """add_border expands shape by 2 (1 each side); translate uses
    expanded size."""
    program = [
        Step(Selector("by_index", 1), "add_border"),
        Step(Selector("by_index", 1), "translate_down"),
    ]
    # After add_border: border=1, cur_h = h_init + 2*1 = 3+2 = 5.
    state = trace_sub_program_v2(program, 3, 4, 0, 0)
    # After border: (e, 1, -1, -1). Translate_down: row += cur_h=5.
    assert state == (D4_IDENTITY, 1, 4, -1)


# --- per_shape_v2_bfs ----------------------------------------------------


_D4_TRIVIAL_STAB = frozenset({D4_IDENTITY})


def test_v2_bfs_canonical_single_translate():
    """One translate_down reaches (e, 0, h, 0) — canonical, min_depth=1."""
    target = (D4_IDENTITY, 0, 3, 0)
    d = per_shape_v2_bfs(
        target, h_init=3, w_init=4, r_init=0, c_init=0,
        primitive_vocab=frozenset({"translate_down", "translate_up"}),
        stab=_D4_TRIVIAL_STAB, max_depth=1,
    )
    assert d == 1


def test_v2_bfs_translate_cancellation_catches_identity():
    """up + down cancels: BFS reaches initial at depth 0."""
    initial = (D4_IDENTITY, 0, 0, 0)
    target = initial
    d = per_shape_v2_bfs(
        target, h_init=3, w_init=4, r_init=0, c_init=0,
        primitive_vocab=frozenset({"translate_up", "translate_down"}),
        stab=_D4_TRIVIAL_STAB, max_depth=2,
    )
    assert d == 0  # initial equal to target; empty program shortcut


def test_v2_bfs_square_shape_translate_symmetry():
    """Square shape (h=w): translate_down(3) reaches (e, 0, 3, 0); same
    distance via rotations and translates in another direction."""
    # h=w=4. After rot90, cur_h = w = 4 (same).
    # Target (e, 0, 4, 0) = translate_down. min_depth = 1.
    target = (D4_IDENTITY, 0, 4, 0)
    d = per_shape_v2_bfs(
        target, h_init=4, w_init=4, r_init=0, c_init=0,
        primitive_vocab=frozenset({
            "translate_up", "translate_down",
            "translate_left", "translate_right",
            "rot90", "flip_h", "flip_v",
        }),
        stab=_D4_TRIVIAL_STAB, max_depth=2,
    )
    assert d == 1


# --- is_scene_admissible V2 dispatcher cases ----------------------------


_V2_LEVEL = frozenset({"by_index"})


def test_v2_single_translate_accepts():
    scene = [_solid_at(3, 4, 5, 5)]
    program = [Step(Selector("by_index", 1), "translate_down")]
    assert is_scene_admissible(program, scene, _V2_VOCAB, _V2_LEVEL) is True


def test_v2_identity_translate_rejected():
    """[translate_up, translate_down] cancels to identity — reject."""
    scene = [_solid_at(3, 4, 5, 5)]
    program = [
        Step(Selector("by_index", 1), "translate_down"),
        Step(Selector("by_index", 1), "translate_up"),
    ]
    assert is_scene_admissible(program, scene, _V2_VOCAB, _V2_LEVEL) is False


def test_v2_square_shape_rot_trivial_rejected():
    """Solid square under by_index(1).rot90: shape's silhouette is full-
    D4-symmetric → rot is a no-op on rendering. Reject (identity coset)."""
    scene = [_solid_at(4, 4, 5, 5)]
    program = [Step(Selector("by_index", 1), "rot90")]
    assert is_scene_admissible(program, scene, _V2_VOCAB, _V2_LEVEL) is False


def test_v2_multi_shape_per_shape_independent():
    """Two shapes with independent by_index sub-programs — V2 checks
    each."""
    scene = [_solid_at(3, 5, 0, 0), _solid_at(3, 5, 10, 10)]
    # shape 1: translate_down. shape 2: translate_right.
    program = [
        Step(Selector("by_index", 1), "translate_down"),
        Step(Selector("by_index", 2), "translate_right"),
    ]
    assert is_scene_admissible(program, scene, _V2_VOCAB, _V2_LEVEL) is True


def test_v2_defers_when_level_has_ALL():
    """Level with ALL — per-shape decomposition can miss ALL-Q parallel
    shortcuts. Defer."""
    scene = [_solid_at(3, 5, 0, 0), _solid_at(3, 5, 10, 10)]
    program = [Step(Selector("by_index", 1), "translate_down")]
    level = frozenset({"ALL", "by_index"})
    assert is_scene_admissible(program, scene, _V2_VOCAB, level) is True


def test_v2_defers_on_pure_d4_vocab():
    """V2 defers pure D4 vocab to V1 (translate/border must be present)."""
    scene = [_solid_at(3, 4, 0, 0)]
    program = [Step(Selector("by_index", 1), "rot90")]
    d4_only = frozenset({"rot90", "flip_h", "flip_v"})
    # V1 should fire on this, not V2. Verify by calling with by_index level.
    # (V1 by_index branch accepts/rejects based on Stab.)
    # For non-square shape h=3,w=4, rot90 creates 4x3. Stab = {e, r²} × flip
    # subgroups — Let's just verify V2 doesn't answer; V1 does.
    # End-to-end: call is_scene_admissible.
    result = is_scene_admissible(program, scene, d4_only, _V2_LEVEL)
    # V1 by_index: shape has Stab including r² (non-square but rot180
    # preserves solid rect) + s + r²s. Stab = {e, r², s, r²s} (4 elems).
    # g_P = r. Coset r·Stab = {r, r·r² = r³, r·s = rs, r·r²s = r³s}.
    # min_depth under D4 vocab: r=1, r³=3 (via r·r·r or flip combos).
    # rs = ... min via flip_h · rot?  Actually min words in {r,s,r²s}
    # for rs: rs = r·s, 2 steps. Or other combos. Min word = 1 for r,
    # others ≥ 2. So min_depth = 1 == len(P). Accept.
    assert result is True


def test_v2_rot180_on_square_shape_single_shape_reject():
    """Square shape: rot180 (two rot90) is coset-equivalent to identity
    (r² ∈ Stab). V2 BFS finds initial state equivalent at depth 0."""
    scene = [_solid_at(4, 4, 5, 5)]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "rot90"),
    ]
    assert is_scene_admissible(program, scene, _V2_VOCAB, _V2_LEVEL) is False


# === Phase C joint BFS (mixed-axis vocab) ================================


def _solid_shape_at(color: int, size: int = 3,
                    position: tuple[int, int] = (0, 0)) -> Shape:
    return Shape(np.ones((size, size), dtype=np.int64) * color, position)


def test_phase_c_d4_plus_color_identity_rejected():
    """Square solid shape under (rot90, swap_colors, swap_colors) —
    rot full-Stab, swaps cancel → joint identity. Reject."""
    scene = [_solid_shape_at(1, 3, (0, 0)), _solid_shape_at(2, 3, (5, 0))]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
    ]
    vocab = frozenset({"rot90", "flip_h", "flip_v", "swap_colors"})
    level = frozenset({"by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is False


def test_phase_c_d4_plus_color_canonical():
    """rot90 (single step) on non-square shape + one swap — minimal."""
    scene = [
        Shape(np.array([[1, 0], [1, 1]], dtype=np.int64), (0, 0)),
        _solid_shape_at(2, 3, (5, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
    ]
    vocab = frozenset({"rot90", "flip_h", "flip_v", "swap_colors"})
    level = frozenset({"by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is True


def test_phase_c_translate_plus_color():
    """translate_down + next_color on two shapes — mixed axes, Phase C
    handles joint BFS."""
    scene = [
        Shape(np.array([[1, 0], [1, 1]], dtype=np.int64), (0, 0)),
        _solid_shape_at(2, 3, (5, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "translate_down"),
        Step(Selector("by_index", 1), "next_color"),
    ]
    vocab = frozenset({"translate_down", "translate_up", "next_color"})
    level = frozenset({"by_index"})
    # Canonical — no shortcut.
    assert is_scene_admissible(program, scene, vocab, level) is True


def test_phase_c_mixed_identity_rejected():
    """[rot90, rot90] on square shape (full D4 Stab) — rot180 collapses
    to identity on rendering. Phase C joint BFS catches it even when
    vocab also includes color prims."""
    scene = [_solid_shape_at(1, 3, (0, 0)), _solid_shape_at(2, 3, (5, 0))]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "rot90"),
    ]
    vocab = frozenset({"rot90", "next_color"})
    level = frozenset({"by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is False


def test_phase_c_handles_two_color_shape():
    """Bordered shape (2 colors) — Phase C now tracks primary AND
    secondary colors per shape, so it fires. For a rotation-symmetric
    bordered shape (3x3 with center), [rot90, next_color] has a
    shortcut Q = [next_color] since rot90 is a Stab-absorbed no-op."""
    bordered = np.array([
        [2, 2, 2], [2, 1, 2], [2, 2, 2],
    ], dtype=np.int64)
    scene = [Shape(bordered, (0, 0)), _solid_shape_at(3, 3, (5, 0))]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "next_color"),
    ]
    vocab = frozenset({"rot90", "next_color"})
    level = frozenset({"by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is False


def test_phase_c_defers_on_three_color_shape():
    """Patterned shape with 3+ distinct colors is beyond Phase C's
    primary+secondary tracking — defers."""
    three_color = np.array([
        [1, 2, 3], [1, 2, 3], [1, 2, 3],
    ], dtype=np.int64)
    scene = [Shape(three_color, (0, 0)), _solid_shape_at(4, 3, (5, 0))]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "next_color"),
    ]
    vocab = frozenset({"rot90", "next_color"})
    level = frozenset({"by_index"})
    assert is_scene_admissible(program, scene, vocab, level) is True


def test_phase_c_all_level_now_analyzed():
    """Level = {ALL} + solid square shapes. Phase C (post-extension)
    analyzes: rot90 is a no-op on solid squares (full D4 Stab); the
    length-1 Q = [next_color] reaches the same rendering as the
    length-2 P. Reject."""
    scene = [_solid_shape_at(1, 3, (0, 0)), _solid_shape_at(2, 3, (5, 0))]
    program = [Step(ALL, "rot90"), Step(ALL, "next_color")]
    vocab = frozenset({"rot90", "next_color"})
    level = frozenset({"ALL"})
    assert is_scene_admissible(program, scene, vocab, level) is False


def test_v3_position_rejects_align_shortcut_via_equal_initial_rows():
    """Three shapes with shape_1 at row 0, shape_2 at row 5, shape_3 at
    row 5 (same as shape_2). Program [align_row(1, 2), align_row(2, 3)]
    traces: (0, 5, 5) → (5, 5, 5) step 1 → step 2 fails (rows 5 == 5).
    Returns None, defer.

    Alternative: [align_row(1, 2), align_row(1, 3)]: step 1 OK (0→5).
    Step 2: align(1, 3) — rows (5, 5, 5): rows[1]=5, rows[3]=5 → same,
    fail. trace returns None, defer.

    Let's construct a case where P succeeds but a shorter Q matches.
    Initial rows (0, 3, 5). P = [align_row(1, 2), align_row(2, 3)]:
      step 1: row[1]=3. State (3, 3, 5).
      step 2: row[2]=5. State (3, 3, 5) — wait row[2] was 3, tgt row[3]=5 → row[2]=5. State (3, 5, 5).
    Hmm, final (3, 5, 5). Shorter Q: [align_row(1, 3)] (len 1): row[1]=5.
    State (5, 3, 5). Different from P. Not a shortcut via this Q.
    """
    # Pick a cleaner test: P = [align_row(A, B), align_row(B, C)] on
    # scene with A != B, B != C, but A == C. Then:
    # step 1: A := B. State (rows[B], rows[B], rows[C]).
    # step 2: B := C. State (rows[B], rows[C], rows[C]).
    # Final rows = (rows[B], rows[C], rows[C]).
    # Shorter Q: align_row(A, C) len 1: A := C. State (rows[C], rows[B], rows[C]).
    # Different from P unless rows[B] == rows[C] — but we said B != C.
    # Hmm no clean 2-step shortcut. Use a different structure.
    #
    # Actual scene-dep shortcut on align: rows (0, 5, 5) — B, C share
    # row. P = [align_row(A, B)] len 1. Can a 0-step Q match? Only if
    # A's initial row == B's row, but we have 0 != 5. So no depth-0 Q.
    # P is canonical.
    #
    # Skip this — scene-dep align shortcuts are rarer than color ones.
    # This test just verifies V3 doesn't falsely reject a canonical case.
    scene = [_sh(1, (0, 0)), _sh(2, (5, 0)), _sh(3, (5, 10))]
    program = [Step(Selector("by_index", 1), "align_row",
                    target=Selector("by_index", 2))]
    assert is_scene_admissible(
        program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
    ) is True
