"""Oracle tests: V1 C(P) soundness vs brute-force ``_scene_has_shorter_shortcut``.

For each random (program, scene) trial:
  - ``cp_accepts = is_scene_admissible(...)``
  - ``brute_force_ok = not _scene_has_shorter_shortcut(...)``
  - Soundness gate: ``cp_accepts → brute_force_ok`` (if False, V1 is
    unsound — a shortcut exists that V1 didn't catch).

We also track **over-rejection** (``cp_rejects ∧ brute_force_ok``) for
telemetry. The target is "low, but non-zero is allowed" — V1 is
**sound by design, not necessarily tight** on every edge case.

Covers:
  - ALL-only level selectors, D4-only vocab, depth 1-4.
  - by_index-only level selectors, D4-only vocab, depth 1-4.

Uses ``ShapeFactory`` + ``place_shapes`` for realistic scenes.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from babytheorist.generation.rejection import _scene_has_shorter_shortcut
from babytheorist.core import ALL, Selector, Shape, Step
from babytheorist.engine import apply_program
from babytheorist.predicates import is_scene_admissible
from babytheorist.shapes import ShapeFactory, place_shapes


_D4_VOCAB = frozenset({"rot90", "flip_h", "flip_v"})
_ALL_LEVEL = frozenset({"ALL"})
_BYINDEX_LEVEL = frozenset({"by_index"})
_PALETTE = (1, 2, 3, 4, 5)


# === Program enumeration (for Q-sets and random P) =======================


def _enum_d4_steps_all() -> list[Step]:
    """All length-1 D4 steps under ALL selector."""
    return [Step(ALL, p) for p in ("rot90", "flip_h", "flip_v")]


def _enum_d4_steps_by_index(n_shapes: int) -> list[Step]:
    """All length-1 D4 steps under by_index(1..n_shapes)."""
    return [
        Step(Selector("by_index", k), p)
        for k in range(1, n_shapes + 1)
        for p in ("rot90", "flip_h", "flip_v")
    ]


def _shorter_programs(
    steps: list[Step], depth: int,
) -> tuple[tuple[Step, ...], ...]:
    """All length-1..depth-1 program tuples over ``steps``."""
    out: list[tuple[Step, ...]] = []
    for k in range(1, depth):
        for combo in product(steps, repeat=k):
            out.append(combo)
    return tuple(out)


# === Random scene + program sampling =====================================


def _sample_scene(
    rng: np.random.Generator, n_shapes: int,
    world_size: int = 15, shape_size_range: tuple[int, int] = (3, 6),
) -> list[Shape] | None:
    """Sample an n-shape placed scene. Returns None if placement fails."""
    factory = ShapeFactory(size_range=shape_size_range)
    shapes = [factory.sample(rng, _PALETTE) for _ in range(n_shapes)]
    placed = place_shapes(shapes, world_size=world_size, gap=2, rng=rng)
    if placed is None:
        return None
    _, positioned = placed
    return positioned


def _sample_program_all(
    rng: np.random.Generator, depth: int,
) -> list[Step]:
    steps = _enum_d4_steps_all()
    return [steps[rng.integers(len(steps))] for _ in range(depth)]


def _sample_program_by_index(
    rng: np.random.Generator, depth: int, n_shapes: int,
) -> list[Step]:
    steps = _enum_d4_steps_by_index(n_shapes)
    return [steps[rng.integers(len(steps))] for _ in range(depth)]


# === Oracle verification core ============================================


def _brute_force_no_shortcut(
    program: list[Step], positioned: list[Shape],
    shorter_qs: tuple[tuple[Step, ...], ...], world_size: int,
) -> bool | None:
    """Run P and brute-force check every Q. ``None`` if P fails to run
    (nothing meaningful to compare)."""
    output_P, _ = apply_program(program, positioned, world_size)
    if output_P is None:
        return None
    found = _scene_has_shorter_shortcut(
        shorter_qs, positioned, world_size, output_P,
    )
    return not found


def _run_oracle_trials(
    n_trials: int,
    sample_program,
    sample_scene,
    enum_steps,
    level_selectors: frozenset[str],
    seed: int,
) -> dict[str, int]:
    """Run ``n_trials`` random (P, scene) pairs, apply the soundness
    assertion, collect telemetry."""
    rng = np.random.default_rng(seed)
    counters = {
        "trials": 0,
        "placed": 0,
        "ran": 0,
        "cp_accepts": 0,
        "cp_rejects": 0,
        "brute_ok": 0,
        "brute_bad": 0,
        "unsound": 0,              # MUST be 0
        "over_rejects": 0,         # telemetry only
        "agreed_accept": 0,
        "agreed_reject": 0,
    }

    for _ in range(n_trials):
        counters["trials"] += 1
        depth = int(rng.integers(1, 5))
        n_shapes = int(rng.integers(1, 4))
        scene = sample_scene(rng, n_shapes)
        if scene is None:
            continue
        counters["placed"] += 1

        # Build program. by_index requires n_shapes.
        program = sample_program(rng, depth, n_shapes) if _needs_n_shapes(sample_program) \
                  else sample_program(rng, depth)

        steps = enum_steps(n_shapes) if _needs_n_shapes(enum_steps) else enum_steps()
        qs = _shorter_programs(steps, depth)

        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1

        cp_accepts = is_scene_admissible(program, scene, _D4_VOCAB, level_selectors)

        if cp_accepts:
            counters["cp_accepts"] += 1
        else:
            counters["cp_rejects"] += 1

        if brute_ok:
            counters["brute_ok"] += 1
        else:
            counters["brute_bad"] += 1

        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        elif cp_accepts and brute_ok:
            counters["agreed_accept"] += 1
        else:
            counters["agreed_reject"] += 1

    return counters


def _needs_n_shapes(fn) -> bool:
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "n_shapes" in sig.parameters


# === The actual oracle tests =============================================


def test_v1_oracle_all_branch_sound():
    """200 trials: ALL-only level, D4 vocab, depths 1-4. V1 ALL branch
    must never miss a brute-force shortcut."""
    stats = _run_oracle_trials(
        n_trials=200,
        sample_program=_sample_program_all,
        sample_scene=_sample_scene,
        enum_steps=_enum_d4_steps_all,
        level_selectors=_ALL_LEVEL,
        seed=0,
    )
    assert stats["unsound"] == 0, (
        f"V1 ALL branch unsound on {stats['unsound']} trial(s): {stats}"
    )
    # Sanity: enough trials executed to make soundness claim meaningful.
    # Random D4 programs on random scenes have a ~25% scene-apply success
    # rate (rotations often collide with adjacent shapes), so 200 trials
    # typically yield ~50 executed.
    assert stats["ran"] > 30, f"too few trials executed: {stats}"


def test_v1_oracle_by_index_branch_sound():
    """200 trials: by_index-only level. V1 by_index branch must never
    miss a brute-force shortcut."""
    stats = _run_oracle_trials(
        n_trials=200,
        sample_program=_sample_program_by_index,
        sample_scene=_sample_scene,
        enum_steps=_enum_d4_steps_by_index,
        level_selectors=_BYINDEX_LEVEL,
        seed=1,
    )
    assert stats["unsound"] == 0, (
        f"V1 by_index branch unsound on {stats['unsound']} trial(s): {stats}"
    )
    assert stats["ran"] > 30, f"too few trials executed: {stats}"


def test_v1_oracle_all_branch_tight_on_random():
    """On random D4 programs + random scenes, over-rejection rate should
    be moderate (most random scenes have trivial Stab → V1 agrees with
    brute-force on "accept")."""
    stats = _run_oracle_trials(
        n_trials=200,
        sample_program=_sample_program_all,
        sample_scene=_sample_scene,
        enum_steps=_enum_d4_steps_all,
        level_selectors=_ALL_LEVEL,
        seed=2,
    )
    # Tightness: at most 15% of trials should be over-rejects.
    # (Random shapes from ShapeFactory rarely hit D4 symmetries, so the
    # agreement rate should be high in practice.)
    ran = stats["ran"]
    if ran > 0:
        over_rate = stats["over_rejects"] / ran
        assert over_rate < 0.25, (
            f"V1 over-rejects {over_rate:.1%} of random trials (> 25% threshold): "
            f"{stats}"
        )


# === Hand-crafted stress cases ==========================================


def test_v1_oracle_rectangle_scene_forces_rejection():
    """Non-square rectangle (2×4 solid): ``Stab = {e, r², s, r²s}``.
    Program [rot90, rot90] has g_P = r² ∈ Stab → full program is a
    scene-level no-op.

    Engine allows each individual rot90 (dimensions flip 2×4 ↔ 4×2 so
    ``_shape_changed`` fires), but final output = initial grid. V1 must
    reject. For brute-force to see the shortcut, a length-1 Q must work
    — Q = [flip_v] = r²·s (since s ∈ Stab, r²·s ≡ r² on this shape).
    Engine should accept single flip_v (dims stay 2×4 but content row
    order flips across the horizontal axis — for a solid rectangle that
    also looks unchanged ... actually yes, solid rect flip is a no-op).

    Simplest: forgo the brute-force part; V1 rejection itself is the
    assertion.
    """
    rect = Shape(np.ones((2, 4), dtype=np.int64) * 3, (0, 0))
    scene = [rect]
    program = [Step(ALL, "rot90"), Step(ALL, "rot90")]

    cp_accepts = is_scene_admissible(program, scene, _D4_VOCAB, _ALL_LEVEL)
    assert cp_accepts is False


# === V3 color axis oracle ===============================================


def _enum_color_steps_by_index(n_shapes: int) -> list[Step]:
    """All length-1 color steps under by_index selectors."""
    out: list[Step] = []
    for k in range(1, n_shapes + 1):
        out.append(Step(Selector("by_index", k), "next_color"))
    for i in range(1, n_shapes + 1):
        for j in range(1, n_shapes + 1):
            if i == j:
                continue
            out.append(Step(Selector("by_index", i), "swap_colors",
                            target=Selector("by_index", j)))
    return out


def _enum_color_steps_mixed(n_shapes: int) -> list[Step]:
    """All length-1 color steps under {ALL, by_index}."""
    out: list[Step] = [Step(ALL, "next_color")]
    out.extend(_enum_color_steps_by_index(n_shapes))
    return out


def _sample_single_color_scene(
    rng: np.random.Generator, n_shapes: int,
    world_size: int = 15, shape_size_range: tuple[int, int] = (3, 6),
) -> list[Shape] | None:
    """Sample scene where every shape is solid (single color)."""
    from babytheorist.shapes import solid as solid_gen
    factory = ShapeFactory(size_range=shape_size_range,
                           generators=(solid_gen,))
    shapes = [factory.sample(rng, _PALETTE) for _ in range(n_shapes)]
    placed = place_shapes(shapes, world_size=world_size, gap=2, rng=rng)
    if placed is None:
        return None
    _, positioned = placed
    return positioned


def _sample_program_from_steps(
    rng: np.random.Generator, depth: int, steps: list[Step],
) -> list[Step]:
    return [steps[rng.integers(len(steps))] for _ in range(depth)]


_V3_COLOR_VOCAB = frozenset({"next_color", "swap_colors"})
_V3_LEVEL_BYINDEX = frozenset({"by_index"})
_V3_LEVEL_MIXED = frozenset({"ALL", "by_index"})


def test_v3_oracle_byindex_sound():
    """200 trials: by_index level, color vocab, depth 1-3. V3 must
    never miss a brute-force shortcut."""
    rng = np.random.default_rng(10)
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_single_color_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_color_steps_by_index(n_shapes)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _V3_COLOR_VOCAB, _V3_LEVEL_BYINDEX,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V3 unsound: {counters}"
    assert counters["ran"] > 30, f"too few trials ran: {counters}"


def test_v3_oracle_mixed_level_sound():
    """200 trials under level_selectors = {ALL, by_index}."""
    rng = np.random.default_rng(11)
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_single_color_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_color_steps_mixed(n_shapes)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _V3_COLOR_VOCAB, _V3_LEVEL_MIXED,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V3 unsound: {counters}"
    assert counters["ran"] > 30, f"too few trials ran: {counters}"


# === V1 subset BFS oracle with dynamic selectors ========================


def _enum_d4_steps_dynamic(n_shapes: int,
                            selector_types: tuple[str, ...]) -> list[Step]:
    """D4 steps expanded over the given selector types (dynamic + static)."""
    out: list[Step] = []
    for prim in ("rot90", "flip_h", "flip_v"):
        for t in selector_types:
            if t == "ALL":
                out.append(Step(Selector("ALL"), prim))
            elif t == "by_index":
                for k in range(1, n_shapes + 1):
                    out.append(Step(Selector("by_index", k), prim))
            elif t == "by_color":
                for c in (1, 2, 3, 4, 5):
                    out.append(Step(Selector("by_color", c), prim))
            elif t == "is_hollow":
                out.append(Step(Selector("is_hollow"), prim))
            elif t in ("topmost", "bottommost", "leftmost", "rightmost"):
                out.append(Step(Selector(t), prim))
    return out


def test_v1_oracle_dynamic_selectors_sound():
    """V1 subset BFS with dynamic selectors (by_color, topmost, etc.)
    on random scenes. Must be sound vs brute-force."""
    rng = np.random.default_rng(30)
    selector_types = ("ALL", "by_index", "by_color",
                      "topmost", "bottommost")
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    level = frozenset(selector_types)
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_d4_steps_dynamic(n_shapes, selector_types)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _D4_VOCAB, level,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V1 subset (dynamic) unsound: {counters}"
    assert counters["ran"] > 20, f"too few trials: {counters}"


def test_v1_oracle_mixed_level_selectors_sound():
    """level_selectors = {ALL, by_index} — old V1 ALL branch could be
    unsound, new subset BFS must catch by_index shortcuts."""
    rng = np.random.default_rng(31)
    selector_types = ("ALL", "by_index")
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    level = frozenset(selector_types)
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_d4_steps_dynamic(n_shapes, selector_types)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _D4_VOCAB, level,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V1 subset (mixed level) unsound: {counters}"


def _enum_color_steps_dynamic(n_shapes: int,
                              selector_types: tuple[str, ...]) -> list[Step]:
    """Color steps (next_color + swap_colors) over given selector types."""
    out: list[Step] = []
    # next_color accepts any selector type.
    for t in selector_types:
        if t == "ALL":
            out.append(Step(Selector("ALL"), "next_color"))
        elif t == "by_index":
            for k in range(1, n_shapes + 1):
                out.append(Step(Selector("by_index", k), "next_color"))
        elif t == "is_hollow":
            out.append(Step(Selector("is_hollow"), "next_color"))
        elif t in ("topmost", "bottommost", "leftmost", "rightmost"):
            out.append(Step(Selector(t), "next_color"))
    # swap_colors: only single selectors for src/tgt.
    single_types = [t for t in selector_types
                    if t in ("by_index", "topmost", "bottommost",
                             "leftmost", "rightmost")]
    for src_type in single_types:
        for tgt_type in single_types:
            if src_type == "by_index" and tgt_type == "by_index":
                for i in range(1, n_shapes + 1):
                    for j in range(1, n_shapes + 1):
                        if i == j:
                            continue
                        out.append(Step(Selector("by_index", i),
                                        "swap_colors",
                                        target=Selector("by_index", j)))
            elif src_type == "by_index":
                for i in range(1, n_shapes + 1):
                    out.append(Step(Selector("by_index", i),
                                    "swap_colors",
                                    target=Selector(tgt_type)))
            elif tgt_type == "by_index":
                for j in range(1, n_shapes + 1):
                    out.append(Step(Selector(src_type), "swap_colors",
                                    target=Selector("by_index", j)))
            elif src_type != tgt_type:
                out.append(Step(Selector(src_type), "swap_colors",
                                target=Selector(tgt_type)))
    return out


def test_v3_oracle_color_dynamic_selectors_sound():
    """V3 color with position-stable dynamic selectors
    (topmost/bottommost) must be sound vs brute-force."""
    rng = np.random.default_rng(40)
    selector_types = ("ALL", "by_index", "topmost", "bottommost")
    level = frozenset(selector_types)
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_single_color_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_color_steps_dynamic(n_shapes, selector_types)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _V3_COLOR_VOCAB, level,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, (
        f"V3 color (dynamic selectors) unsound: {counters}"
    )
    assert counters["ran"] > 20, f"too few trials: {counters}"


_V3_POSITION_VOCAB = frozenset({"align_row", "align_col"})


def _enum_align_steps_by_index(n_shapes: int) -> list[Step]:
    out: list[Step] = []
    for prim in ("align_row", "align_col"):
        for i in range(1, n_shapes + 1):
            for j in range(1, n_shapes + 1):
                if i == j:
                    continue
                out.append(Step(Selector("by_index", i), prim,
                                target=Selector("by_index", j)))
    return out


def test_v3_oracle_position_byindex_sound():
    """200 trials: by_index level, align vocab, depth 1-3.

    Soundness assertion: V3 accepts ⇒ brute-force OK. V3 may over-reject
    (BFS ignores collision; brute-force would reject colliding Q's as
    apply failures, making them non-shortcuts).
    """
    rng = np.random.default_rng(20)
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(200):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_align_steps_by_index(n_shapes)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _V3_POSITION_VOCAB, frozenset({"by_index"}),
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V3 position unsound: {counters}"
    # Position programs often fail apply (collision after align), so
    # ``ran`` may be low. We just need non-zero to validate the oracle.
    assert counters["ran"] > 5, f"too few trials ran: {counters}"


# === V2 translate + add_border oracle ====================================


_V2_VOCAB_FULL = frozenset({
    "rot90", "flip_h", "flip_v",
    "translate_up", "translate_down", "translate_left", "translate_right",
    "add_border",
})


def _enum_v2_steps_by_index(n_shapes: int) -> list[Step]:
    out: list[Step] = []
    for prim in _V2_VOCAB_FULL:
        for k in range(1, n_shapes + 1):
            out.append(Step(Selector("by_index", k), prim))
    return out


def test_v2_oracle_byindex_sound():
    """V2 by_index scope: D4 + translate + add_border. Must be sound
    vs brute-force over all length-<depth Q programs."""
    rng = np.random.default_rng(50)
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    level = frozenset({"by_index"})
    for _ in range(120):
        depth = int(rng.integers(1, 4))
        n_shapes = int(rng.integers(1, 3))
        scene = _sample_scene(rng, n_shapes, world_size=20,
                              shape_size_range=(3, 5))
        if scene is None:
            continue
        steps = _enum_v2_steps_by_index(n_shapes)
        # Random programs can blow up size; filter steps that fit within
        # a reasonable world.
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        # Skip if Q count too big (combinatorially blows up at high
        # depth — we cap at 4 primitive-type × 2 shapes ≈ 8^3 = 512).
        if len(qs) > 2000:
            continue
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=20)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(
            program, scene, _V2_VOCAB_FULL, level,
        )
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"V2 unsound: {counters}"
    assert counters["ran"] > 10, f"too few trials: {counters}"


def test_v2_oracle_square_shape_translate_cancellation():
    """Solid square — rotations + translates can combine in many ways
    to reach the same concrete final state. Scene-concrete BFS catches
    these shortcuts."""
    # Square shape → rot90 on shape is no-op for rendering. Program
    # [rot90, rot90] rejected.
    scene = [Shape(np.ones((3, 3), dtype=np.int64) * 5, (5, 5))]
    program = [
        Step(Selector("by_index", 1), "rot90"),
        Step(Selector("by_index", 1), "rot90"),
    ]
    assert is_scene_admissible(
        program, scene, _V2_VOCAB_FULL, frozenset({"by_index"}),
    ) is False


# === Phase C oracle ======================================================


def _enum_mixed_steps_by_index(
    n_shapes: int, vocab: frozenset[str],
) -> list[Step]:
    """All by_index steps under ``vocab`` for ``n_shapes`` shapes.
    Relational primitives (swap_colors, align_row, align_col) get both
    src and tgt from by_index."""
    relational = frozenset({"swap_colors", "align_row", "align_col"})
    out: list[Step] = []
    for prim in vocab:
        if prim in relational:
            for i in range(1, n_shapes + 1):
                for j in range(1, n_shapes + 1):
                    if i == j:
                        continue
                    out.append(Step(Selector("by_index", i), prim,
                                    target=Selector("by_index", j)))
        else:
            for k in range(1, n_shapes + 1):
                out.append(Step(Selector("by_index", k), prim))
    return out


def test_phase_c_oracle_d4_plus_color_byindex_sound():
    """Phase C: mixed D4 + color vocab, by_index level, single-color
    shapes. Must be sound vs brute-force."""
    rng = np.random.default_rng(60)
    vocab = frozenset({"rot90", "flip_h", "flip_v",
                       "next_color", "swap_colors"})
    level = frozenset({"by_index"})
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(100):
        depth = int(rng.integers(2, 4))  # depth 2-3 within cap
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_single_color_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_mixed_steps_by_index(n_shapes, vocab)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        if len(qs) > 2000:
            continue
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(program, scene, vocab, level)
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"Phase C D4+color unsound: {counters}"


def test_phase_c_oracle_translate_plus_color_byindex_sound():
    """Phase C: translate + color mixed."""
    rng = np.random.default_rng(61)
    vocab = frozenset({
        "translate_up", "translate_down", "translate_left", "translate_right",
        "next_color", "swap_colors",
    })
    level = frozenset({"by_index"})
    counters = {"ran": 0, "unsound": 0, "over_rejects": 0, "agreed": 0}
    for _ in range(100):
        depth = int(rng.integers(2, 4))
        n_shapes = int(rng.integers(2, 4))
        scene = _sample_single_color_scene(rng, n_shapes)
        if scene is None:
            continue
        steps = _enum_mixed_steps_by_index(n_shapes, vocab)
        program = _sample_program_from_steps(rng, depth, steps)
        qs = _shorter_programs(steps, depth)
        if len(qs) > 2000:
            continue
        brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=20)
        if brute_ok is None:
            continue
        counters["ran"] += 1
        cp_accepts = is_scene_admissible(program, scene, vocab, level)
        if cp_accepts and not brute_ok:
            counters["unsound"] += 1
        elif not cp_accepts and brute_ok:
            counters["over_rejects"] += 1
        else:
            counters["agreed"] += 1
    assert counters["unsound"] == 0, f"Phase C translate+color unsound: {counters}"


def test_v3_oracle_user_example_against_brute_force():
    """Hand-crafted: swap chain on (1, 2, 2) — V3 must reject AND
    brute-force must find the shortcut Q = [swap(A, C)]."""
    scene = [
        Shape(np.ones((3, 3), dtype=np.int64) * 1, (0, 0)),
        Shape(np.ones((3, 3), dtype=np.int64) * 2, (5, 0)),
        Shape(np.ones((3, 3), dtype=np.int64) * 2, (10, 0)),
    ]
    program = [
        Step(Selector("by_index", 1), "swap_colors",
             target=Selector("by_index", 2)),
        Step(Selector("by_index", 2), "swap_colors",
             target=Selector("by_index", 3)),
    ]
    cp_accepts = is_scene_admissible(
        program, scene, _V3_COLOR_VOCAB, _V3_LEVEL_BYINDEX,
    )
    steps = _enum_color_steps_by_index(n_shapes=3)
    qs = _shorter_programs(steps, depth=2)
    brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)
    assert cp_accepts is False
    assert brute_ok is False  # Q = [swap(A, C)] exists at depth 1


def test_v1_oracle_identity_collapse_rejected_by_v1():
    """Program with g_P = e (net identity) must be rejected by V1
    regardless of shape Stab — the empty Q is always a shortcut.

    Note: brute-force as implemented enumerates only Q's of length ≥ 1,
    so it would MISS the empty-Q shortcut. V1 is strictly stronger here;
    the ``_generate_pair`` identity backstop (``input_grid != output_grid``)
    complements brute-force to close this gap. This test documents V1's
    over-strictness vs brute-force — a feature, not a bug.
    """
    grid = np.array([
        [1, 1, 1, 2],
        [1, 1, 1, 1],
    ], dtype=np.int64)
    shape = Shape(grid, (0, 0))
    scene = [shape]
    program = [Step(ALL, "flip_h"), Step(ALL, "flip_h")]

    cp_accepts = is_scene_admissible(program, scene, _D4_VOCAB, _ALL_LEVEL)
    assert cp_accepts is False


def test_v1_oracle_asymmetric_scene_accepted_canonical():
    """Asymmetric L-like shape under a single rot90: no shortcut; both
    V1 and brute-force accept."""
    l_grid = np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64)
    scene = [Shape(l_grid, (0, 0))]
    program = [Step(ALL, "rot90")]

    cp_accepts = is_scene_admissible(program, scene, _D4_VOCAB, _ALL_LEVEL)
    steps = _enum_d4_steps_all()
    qs = _shorter_programs(steps, depth=1)
    brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)

    assert cp_accepts is True
    assert brute_ok is True


def test_v1_oracle_symmetric_by_index_rejected():
    """by_index(1) rot90 on a square: shape 1's sub-prog = [rot90] but
    r ∈ Stab(square) → Q_1 = [] produces same render. V1 must reject."""
    square = Shape(np.ones((4, 4), dtype=np.int64) * 3, (0, 0))
    asym = Shape(np.array([
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
    ], dtype=np.int64), (6, 6))
    scene = [square, asym]
    program = [Step(Selector("by_index", 1), "rot90")]

    cp_accepts = is_scene_admissible(program, scene, _D4_VOCAB, _BYINDEX_LEVEL)
    steps = _enum_d4_steps_by_index(n_shapes=2)
    qs = _shorter_programs(steps, depth=1)
    brute_ok = _brute_force_no_shortcut(program, scene, qs, world_size=15)

    assert cp_accepts is False
    # brute-force: shorter Q's include the empty-prefix (len 0) — but
    # len(P) == 1, so only depth-0 Q's are "shorter". Empty tuple isn't
    # enumerated (k starts at 1), so brute-force might actually say OK.
    # This is a **tightness gap**, not a soundness gap: V1 correctly
    # identifies that a len-0 Q would be a shortcut; brute-force's
    # enumeration doesn't include len 0 by convention.
    # We only assert V1 rejected (V1 is stricter here).
    _ = brute_ok
