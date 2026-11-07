"""Canonical-form minimality analysis for programs (Phase 1 Rule 1 boost).

**Problem**: ``trajectory_distinct`` detects only "returns to a prior
state" in one program's own trajectory. It misses three structural
completeness gaps — alternative-path shortcuts (e.g.
``rot·rot·flip_h = flip_v``), cross-dim independent cycles, and
phase-invariant cycle nesting. All three boil down to: **"P's net
effect is reachable in fewer primitives than ``len(P)``."**

**Principled fix**: for each symbolic effect, compute ``min_depth``
under the given vocab. ``P`` is minimal iff ``len(P) == min_depth(effect(P))``.

**Tier 1 scope** (this module's scope): programs where effect
dimensions don't interact — i.e. the vocab excludes ``translate_*``
(whose delta depends on orient) and mixes of ``add_border`` with
``translate`` (border changes height, which affects translate). For
such programs, the effect decomposes cleanly per dimension and
``min_depth`` is an additive sum over D4 / Z_9 / grid-toggle / border
contributions. Single-shape only; relational primitives and
multi-shape coupling are Tier 3+ work.

Programs outside Tier 1 scope → ``is_minimal_tier1`` returns ``True``
(pass through; the weaker ``trajectory_distinct`` still filters them).
"""

from __future__ import annotations

from functools import lru_cache

from .core import Selector, Step
from .symbolic import (
    ColorVal,
    D4,
    D4_IDENTITY,
    PreconditionFail,
    apply_symbolic_program,
)


__all__ = [
    "d4_min_depths",
    "min_depth_orient",
    "min_depth_color",
    "min_depth_grid",
    "min_depth_border",
    "is_minimal_tier1",
    "orient_position_bfs_table",
    "is_minimal_tier2",
    "orient_border_position_bfs_table",
    "is_minimal_tier3",
    "is_minimal_multi_shape_by_index",
    "color_joint_bfs_table",
    "is_minimal_tier4_swap",
    "row_col_source_joint_bfs_table",
    "is_minimal_tier5_align",
]


_ORIENT_PRIMITIVES = frozenset({"rot90", "flip_h", "flip_v"})
_COLOR_PRIMITIVES = frozenset({"next_color"})
_GRID_PRIMITIVES = frozenset({"hollow", "fill_interior"})
_BORDER_PRIMITIVES = frozenset({"add_border"})
_TRANSLATE_PRIMITIVES = frozenset({
    "translate_up", "translate_down", "translate_left", "translate_right",
})
_RELATIONAL_PRIMITIVES = frozenset({"swap_colors", "align_row", "align_col"})


def _can_analyze_as_single_shape(
    program: list[Step], n_shapes: int, grid_shapes: list[str] | None,
) -> bool:
    """True iff a multi-shape program can be reduced to single-shape
    canonical-form analysis on shape 0.

    Conditions:
      1. Every step uses the ``ALL`` selector (and ``ALL`` target for
         relational). Under ``ALL``, each primitive affects every shape
         uniformly, so shape 0's effect is representative of the whole
         program's effect (modulo per-shape init-var renaming).
      2. All shapes share the same initial ``grid_shape``. Without this,
         grid-shape transitions per shape could diverge and shape 0
         would no longer be representative.

    ``n_shapes == 1`` trivially satisfies — no multi-shape flattening
    needed.
    """
    if n_shapes == 1:
        return True
    for step in program:
        if step.selector.type != "ALL":
            return False
        if step.target is not None and step.target.type != "ALL":
            return False
    if grid_shapes is not None and len(set(grid_shapes)) > 1:
        return False
    return True


# === Per-dimension min-depth =============================================


def _d4_generators(orient_vocab: frozenset[str]):
    gens = []
    if "rot90" in orient_vocab:
        gens.append(D4.apply_rot)
    if "flip_h" in orient_vocab:
        gens.append(D4.apply_flip_h)
    if "flip_v" in orient_vocab:
        gens.append(D4.apply_flip_v)
    return gens


@lru_cache(maxsize=None)
def d4_min_depths(orient_vocab: frozenset[str]) -> dict[D4, int]:
    """BFS from ``e``: min primitives (from ``orient_vocab``) to reach each
    D4 element. Cached per vocab.

    Unreachable elements are absent from the returned dict.
    """
    gens = _d4_generators(orient_vocab)
    distances: dict[D4, int] = {D4_IDENTITY: 0}
    if not gens:
        return distances
    frontier = [D4_IDENTITY]
    while frontier:
        nxt = []
        for g in frontier:
            for fn in gens:
                g2 = fn(g)
                if g2 not in distances:
                    distances[g2] = distances[g] + 1
                    nxt.append(g2)
        frontier = nxt
    return distances


def min_depth_orient(target: D4, orient_vocab: frozenset[str]) -> int:
    """Min primitives (from ``orient_vocab``) composing ``target`` from identity."""
    table = d4_min_depths(orient_vocab)
    if target not in table:
        raise ValueError(
            f"D4 element {target!r} unreachable from {sorted(orient_vocab)}"
        )
    return table[target]


def min_depth_color(offset: int, color_vocab: frozenset[str]) -> int:
    """Min ``next_color`` applications to reach ``offset mod 9``."""
    offset %= 9
    if offset == 0:
        return 0
    if "next_color" not in color_vocab:
        raise ValueError(f"color offset {offset} unreachable without next_color")
    return offset


def min_depth_grid(
    initial: str, final: str, grid_vocab: frozenset[str],
) -> int:
    """Min ``hollow``/``fill_interior`` primitives to go from
    ``initial`` grid_shape to ``final`` grid_shape.

    fill_interior requires ``initial == "hollow"``.
    """
    if initial == final:
        return 0
    if initial == "solid" and final == "hollow":
        if "hollow" not in grid_vocab:
            raise ValueError("cannot reach hollow without 'hollow'")
        return 1
    if initial == "hollow" and final == "solid":
        if "fill_interior" not in grid_vocab:
            raise ValueError("cannot reach solid from hollow without 'fill_interior'")
        return 1
    raise ValueError(f"unknown grid transition {initial!r} → {final!r}")


def min_depth_border(count: int, border_vocab: frozenset[str]) -> int:
    """Min ``add_border`` applications to reach net border count ``count``."""
    if count == 0:
        return 0
    if "add_border" not in border_vocab:
        raise ValueError(f"border count {count} unreachable without add_border")
    return count


# === Tier 1 minimality check ===========================================


def is_minimal_tier1(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """True iff ``program`` is minimal under Tier 1 analysis.

    ``vocab`` is the **sampler's universe vocab** — all primitives the
    sampler could draw from. Shorter alternatives are compared against
    this vocab, not ``program``'s own primitives. (Otherwise a program
    using only a subset would always look minimal within its own
    subset, missing alternatives that use other primitives from the
    full vocab.)

    Tier 1 applies only when the vocab permits non-interacting
    analysis and the scene is single-shape:
      - vocab ⊆ {rot90, flip_h, flip_v, next_color, hollow,
                  fill_interior, add_border}
      - no ``translate_*`` (orient-dependent effect)
      - no relational primitives (swap_colors, align_row, align_col)
      - ``n_shapes == 1``

    Outside this scope, returns ``True`` (no conclusion drawn).

    Within scope, returns ``False`` iff ``len(program)`` strictly
    exceeds the sum of per-dimension minimum depths required to reach
    the program's symbolic final state under ``vocab``.

    Multi-shape scenes are supported when every step uses the ``ALL``
    selector and all shapes share the same initial ``grid_shape`` —
    in that case every shape evolves identically (modulo per-shape
    init-var renaming), and shape 0's effect is representative.
    """
    if not program:
        return True
    if not _can_analyze_as_single_shape(program, n_shapes, grid_shapes):
        return True

    if vocab & _TRANSLATE_PRIMITIVES:
        return True  # Tier 2 territory
    if vocab & _RELATIONAL_PRIMITIVES:
        return True  # Tier 3 territory

    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False  # infeasible → reject

    final = traj[-1][0]
    initial_grid = (grid_shapes or ["solid"])[0]

    if final.color.base != "0":
        return True  # unexpected (relational must've slipped in)
    color_offset = final.color.offset

    orient_vocab = vocab & _ORIENT_PRIMITIVES
    color_vocab = vocab & _COLOR_PRIMITIVES
    grid_vocab = vocab & _GRID_PRIMITIVES
    border_vocab = vocab & _BORDER_PRIMITIVES

    try:
        total = (
            min_depth_orient(final.orientation, orient_vocab)
            + min_depth_color(color_offset, color_vocab)
            + min_depth_grid(initial_grid, final.grid_shape, grid_vocab)
            + min_depth_border(final.border_count, border_vocab)
        )
    except ValueError:
        # Reachability error under this vocab — defensively accept.
        return True

    return len(program) == total


# === Tier 2 — orient + translate joint min-depth via BFS =================


# Tier 2 vocab shape: orient primitives + translate primitives
# (+ color/grid as orthogonal extensions).  No add_border (its row/col
# shift plus size-on-translate dependency creates schedule-dependent
# constants that break the additive decomposition).  No relational.
_TIER2_OP_PRIMITIVES = _ORIENT_PRIMITIVES | _TRANSLATE_PRIMITIVES
_TIER2_COMPATIBLE = _TIER2_OP_PRIMITIVES | _COLOR_PRIMITIVES | _GRID_PRIMITIVES

# BFS depth cap for the orient+translate state-space precompute.  The
# state count grows as O(d^4) so depth 20 (comfortably covers curriculum
# levels 1–15) costs under ~60K states.
_TIER2_BFS_DEPTH = 20


def _op_state_step(name: str, state: tuple) -> tuple:
    """Apply one orient/translate primitive to an
    ``(orient, rh, rw, ch, cw)`` state. ``rh``/``rw``/``ch``/``cw`` are
    net integer coefficients on ``h_A``/``w_A`` for row/col deltas.
    Translate effect depends on ``orient.rot % 2``.
    """
    orient, rh, rw, ch, cw = state
    if name == "rot90":
        return (orient.apply_rot(), rh, rw, ch, cw)
    if name == "flip_h":
        return (orient.apply_flip_h(), rh, rw, ch, cw)
    if name == "flip_v":
        return (orient.apply_flip_v(), rh, rw, ch, cw)
    parity = orient.rot % 2
    if name == "translate_up":
        return (orient, rh - 1, rw, ch, cw) if parity == 0 else (orient, rh, rw - 1, ch, cw)
    if name == "translate_down":
        return (orient, rh + 1, rw, ch, cw) if parity == 0 else (orient, rh, rw + 1, ch, cw)
    if name == "translate_left":
        return (orient, rh, rw, ch, cw - 1) if parity == 0 else (orient, rh, rw, ch - 1, cw)
    if name == "translate_right":
        return (orient, rh, rw, ch, cw + 1) if parity == 0 else (orient, rh, rw, ch + 1, cw)
    raise ValueError(f"{name!r} not a Tier 2 op primitive")


@lru_cache(maxsize=None)
def orient_position_bfs_table(op_vocab: frozenset[str]) -> dict:
    """BFS distances from ``(identity, 0, 0, 0, 0)`` to every reachable
    state under ``op_vocab`` ⊆ ``_TIER2_OP_PRIMITIVES``, up to depth
    ``_TIER2_BFS_DEPTH``.  Cached per vocab — first call per vocab
    runs the BFS, subsequent calls are free.
    """
    initial = (D4_IDENTITY, 0, 0, 0, 0)
    distances: dict = {initial: 0}
    frontier = [initial]
    while frontier:
        next_frontier = []
        for state in frontier:
            d = distances[state]
            if d >= _TIER2_BFS_DEPTH:
                continue
            for name in op_vocab:
                new_state = _op_state_step(name, state)
                if new_state not in distances:
                    distances[new_state] = d + 1
                    next_frontier.append(new_state)
        frontier = next_frontier
    return distances


def is_minimal_tier2(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """True iff ``program`` is minimal under Tier 2 analysis.

    Tier 2 handles vocabs combining **orient + translate** primitives
    (with color/grid as orthogonal extras) on a single shape.  Closes
    the translate-phase-navigation gap Tier 1 defers on.  Outside
    scope (``add_border`` / relational / ``n_shapes > 1``): returns
    ``True`` (no conclusion). Pure orient-without-translate vocabs are
    better handled by Tier 1 — those also return ``True`` so callers
    AND both results without double-counting.

    Extracts the program's final orient + row/col coefficients, looks
    up joint min-depth in the cached BFS table, and adds the
    independent color and grid min-depths. Returns ``False`` iff
    ``len(program)`` exceeds the sum.

    Multi-shape via uniform ``ALL`` selectors is analysed on shape 0
    (see ``_can_analyze_as_single_shape``).
    """
    if not program:
        return True
    if not _can_analyze_as_single_shape(program, n_shapes, grid_shapes):
        return True
    if vocab - _TIER2_COMPATIBLE:
        return True  # add_border or relational present → defer
    if not (vocab & _TRANSLATE_PRIMITIVES):
        return True  # no translate → Tier 1 is tighter; defer here

    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False

    final = traj[-1][0]
    initial_grid = (grid_shapes or ["solid"])[0]

    if final.color.base != "0":
        return True
    color_offset = final.color.offset

    # Extract position coefs; require const == 0 (no add_border in scope)
    # and initial r_0/c_0 preserved (align not in scope).
    row_dict = final.row.as_dict()
    col_dict = final.col.as_dict()
    if final.row.const != 0 or final.col.const != 0:
        return True
    if row_dict.get("r_0", 0) != 1 or col_dict.get("c_0", 0) != 1:
        return True
    rh = row_dict.get("h_0", 0)
    rw = row_dict.get("w_0", 0)
    ch = col_dict.get("h_0", 0)
    cw = col_dict.get("w_0", 0)

    op_vocab = vocab & _TIER2_OP_PRIMITIVES
    color_vocab = vocab & _COLOR_PRIMITIVES
    grid_vocab = vocab & _GRID_PRIMITIVES

    op_table = orient_position_bfs_table(op_vocab)
    op_depth = op_table.get((final.orientation, rh, rw, ch, cw))
    if op_depth is None:
        # State not reached within BFS depth cap → its min-depth ≥ cap.
        # len(program) ≤ cap by construction of curriculum, so treat as
        # minimal (can't prove redundancy).
        return True

    try:
        color_depth = min_depth_color(color_offset, color_vocab)
        grid_depth = min_depth_grid(initial_grid, final.grid_shape, grid_vocab)
    except ValueError:
        return True  # sub-vocab too thin; safe accept

    return len(program) == op_depth + color_depth + grid_depth


# === Multi-shape by-index slicing ========================================


def _single_shape_canonical_minimal(
    program: list[Step], vocab: frozenset[str], grid_shapes: list[str] | None,
) -> bool:
    """Run all single-shape canonical rules (Tier 1/2/3). Returns True
    iff every applicable rule says minimal. Internal helper."""
    if not is_minimal_tier1(program, 1, vocab=vocab, grid_shapes=grid_shapes):
        return False
    if not is_minimal_tier2(program, 1, vocab=vocab, grid_shapes=grid_shapes):
        return False
    if not is_minimal_tier3(program, 1, vocab=vocab, grid_shapes=grid_shapes):
        return False
    return True


def _by_index_slice(
    program: list[Step], shape_idx: int,
) -> list[Step] | None:
    """Extract the sub-program that targets shape ``shape_idx`` (0-based),
    rewritten so the single remaining shape is indexed as ``by_index(1)``.

    Returns ``None`` if the program contains any selector outside
    ``by_index`` (e.g. ``ALL``, relational, dynamic) — those cases can't
    be cleanly sliced per shape.
    """
    target_param = shape_idx + 1  # by_index is 1-based
    result: list[Step] = []
    for step in program:
        if step.selector.type != "by_index":
            return None
        if step.target is not None:
            # Relational crosses shapes; per-shape slicing doesn't
            # cleanly capture the two-shape interaction. Defer.
            return None
        if step.selector.param == target_param:
            result.append(Step(Selector("by_index", 1), step.primitive))
    return result


def is_minimal_multi_shape_by_index(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """Multi-shape analysis for programs using only ``by_index`` selectors
    with no relational primitives in the program.

    Each by_index step targets exactly one shape.  The program
    decomposes into per-shape sub-programs that are independent, so
    joint minimality is equivalent to every sub-program being minimal
    under single-shape canonical form.

    Returns ``True`` (defer) whenever slicing is not applicable
    (non-by_index selectors present in the program, relational
    primitives in the program, or ``n_shapes == 1``).

    If the ``vocab`` (not the program) contains relational primitives,
    analysis proceeds with relational stripped from the vocab so T1/T2/T3
    can still check against shorter non-relational alternatives.  A
    shorter program that uses relational would not be found this way —
    sound but incomplete.
    """
    if not program:
        return True
    if n_shapes == 1:
        return True
    # Slicing itself filters: any relational step in program → return None.
    # So if slice succeeds, program is relational-free, and we only need
    # non-relational shorter-equivalent checks.
    analysis_vocab = vocab - _RELATIONAL_PRIMITIVES
    for i in range(n_shapes):
        slice_prog = _by_index_slice(program, i)
        if slice_prog is None:
            return True  # program not fully by-index-sliceable
        single_grid = None if grid_shapes is None else [grid_shapes[i]]
        if not _single_shape_canonical_minimal(slice_prog, analysis_vocab, single_grid):
            return False
    return True


# === Tier 3 — orient + translate + add_border joint BFS ==================


# Tier 3 extends Tier 2 by tracking ``border_count`` and the constant
# terms of ``row``/``col`` deltas. ``add_border`` is the new primitive:
# it shifts row/col by -1 (direct) and makes subsequent translates use
# a size of ``initial + 2 * border_count``, which contributes a further
# ``-2k`` to the constant per translate.  Scheduling of borders vs
# translates is effect-observable (not just count-dependent), so the
# BFS state must include the constants.
_TIER3_OP_PRIMITIVES = _TIER2_OP_PRIMITIVES | _BORDER_PRIMITIVES
_TIER3_COMPATIBLE = _TIER3_OP_PRIMITIVES | _COLOR_PRIMITIVES | _GRID_PRIMITIVES

# BFS depth cap. State space grows faster than Tier 2 because of the
# extra border + 2 const dimensions, so a smaller cap is prudent.
# Covers curriculum levels up to ~10.
_TIER3_BFS_DEPTH = 10


def _op_border_state_step(name: str, state: tuple) -> tuple:
    """Apply one Tier-3 op primitive (orient / translate / add_border)
    to ``(orient, border, rh, rw, ch, cw, row_const, col_const)``."""
    orient, border, rh, rw, ch, cw, rc, cc = state
    if name == "rot90":
        return (orient.apply_rot(), border, rh, rw, ch, cw, rc, cc)
    if name == "flip_h":
        return (orient.apply_flip_h(), border, rh, rw, ch, cw, rc, cc)
    if name == "flip_v":
        return (orient.apply_flip_v(), border, rh, rw, ch, cw, rc, cc)
    if name == "add_border":
        return (orient, border + 1, rh, rw, ch, cw, rc - 1, cc - 1)
    parity = orient.rot % 2
    if name == "translate_up":
        delta_const = -2 * border
        if parity == 0:
            return (orient, border, rh - 1, rw, ch, cw, rc + delta_const, cc)
        return (orient, border, rh, rw - 1, ch, cw, rc + delta_const, cc)
    if name == "translate_down":
        delta_const = 2 * border
        if parity == 0:
            return (orient, border, rh + 1, rw, ch, cw, rc + delta_const, cc)
        return (orient, border, rh, rw + 1, ch, cw, rc + delta_const, cc)
    if name == "translate_left":
        delta_const = -2 * border
        if parity == 0:
            return (orient, border, rh, rw, ch, cw - 1, rc, cc + delta_const)
        return (orient, border, rh, rw, ch - 1, cw, rc, cc + delta_const)
    if name == "translate_right":
        delta_const = 2 * border
        if parity == 0:
            return (orient, border, rh, rw, ch, cw + 1, rc, cc + delta_const)
        return (orient, border, rh, rw, ch + 1, cw, rc, cc + delta_const)
    raise ValueError(f"{name!r} not a Tier 3 op primitive")


@lru_cache(maxsize=None)
def orient_border_position_bfs_table(op_border_vocab: frozenset[str]) -> dict:
    """BFS distances from the initial state ``(e, 0, 0, 0, 0, 0, 0, 0)``
    over the Tier-3 op vocab.  Cached per vocab; depth capped at
    ``_TIER3_BFS_DEPTH`` to keep memory bounded.
    """
    initial = (D4_IDENTITY, 0, 0, 0, 0, 0, 0, 0)
    distances: dict = {initial: 0}
    frontier = [initial]
    while frontier:
        next_frontier = []
        for state in frontier:
            d = distances[state]
            if d >= _TIER3_BFS_DEPTH:
                continue
            for name in op_border_vocab:
                new_state = _op_border_state_step(name, state)
                if new_state not in distances:
                    distances[new_state] = d + 1
                    next_frontier.append(new_state)
        frontier = next_frontier
    return distances


def is_minimal_tier3(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """True iff ``program`` is minimal under Tier 3 analysis.

    Tier 3 extends Tier 2 with ``add_border`` support by expanding the
    BFS state with ``border_count`` and the row/col constant shifts
    from border-direct (-1) + translate-at-border (-2k) contributions.
    Scope: vocab ⊆ {orient, translate, add_border, color, grid}, single
    shape, no relational.

    Only activates when the program length is ≤ ``_TIER3_BFS_DEPTH``;
    deeper programs return ``True`` (BFS table can't prove redundancy).
    Also defers when vocab has no ``add_border`` (Tier 2 is tighter)
    or no translate (Tier 1).

    Multi-shape via uniform ``ALL`` selectors is analysed on shape 0
    (see ``_can_analyze_as_single_shape``).
    """
    if not program:
        return True
    if not _can_analyze_as_single_shape(program, n_shapes, grid_shapes):
        return True
    if vocab - _TIER3_COMPATIBLE:
        return True  # relational or anything else → defer
    if not (vocab & _BORDER_PRIMITIVES):
        return True  # no add_border → Tier 2 is the right check
    if not (vocab & _TRANSLATE_PRIMITIVES):
        return True  # no translate → Tier 1 is the right check
    if len(program) > _TIER3_BFS_DEPTH:
        return True  # beyond BFS cap — cannot conclude

    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False

    final = traj[-1][0]
    initial_grid = (grid_shapes or ["solid"])[0]

    if final.color.base != "0":
        return True
    color_offset = final.color.offset

    row_dict = final.row.as_dict()
    col_dict = final.col.as_dict()
    if row_dict.get("r_0", 0) != 1 or col_dict.get("c_0", 0) != 1:
        return True
    target_state = (
        final.orientation,
        final.border_count,
        row_dict.get("h_0", 0),
        row_dict.get("w_0", 0),
        col_dict.get("h_0", 0),
        col_dict.get("w_0", 0),
        final.row.const,
        final.col.const,
    )

    op_border_vocab = vocab & _TIER3_OP_PRIMITIVES
    color_vocab = vocab & _COLOR_PRIMITIVES
    grid_vocab = vocab & _GRID_PRIMITIVES

    table = orient_border_position_bfs_table(op_border_vocab)
    op_depth = table.get(target_state)
    if op_depth is None:
        # Target state not reached within BFS depth cap.  Since
        # len(program) ≤ cap and P demonstrably reaches the state in
        # len(program) steps, this shouldn't happen in correct code —
        # but defensively return True (can't prove shorter exists).
        return True

    try:
        color_depth = min_depth_color(color_offset, color_vocab)
        grid_depth = min_depth_grid(initial_grid, final.grid_shape, grid_vocab)
    except ValueError:
        return True

    return len(program) == op_depth + color_depth + grid_depth


# === Tier 4 — multi-shape relational (swap_colors joint BFS) ============


# Color dim joint BFS for multi-shape programs containing swap_colors.
# Decomposes per-dimension: color dim is analysed jointly (swap couples
# shapes), other dims are analysed per-shape via single-shape canonical
# rules.  Total min-depth is the sum.
#
# Scope:
#  - No ``align_row``/``align_col`` in the program (align couples row/col
#    dim — would need a parallel joint BFS, not yet implemented).
#  - ``by_index`` selectors only (for per-shape slicing of non-color
#    primitives).
#  - ``n_shapes`` ≥ 2 (single-shape case handled elsewhere).
#
# State: tuple of ``ColorVal`` per shape. BFS capped at depth 20.

_TIER4_BFS_DEPTH = 20


@lru_cache(maxsize=None)
def color_joint_bfs_table(
    color_vocab: frozenset[str], n_shapes: int,
) -> dict:
    """BFS distances from ``(ColorVal(i, 0) for i in shapes)`` to every
    reachable joint color state under the color vocab ⊆
    ``{next_color, swap_colors}``.  Cached per ``(vocab, n_shapes)``.

    Enumerates transitions via all valid selector/target combinations
    (``next_color`` with ``ALL`` or ``by_index(k)``; ``swap_colors``
    with ``by_index(i), by_index(j)`` for ``i != j``).
    """
    initial = tuple(ColorVal(str(i), 0) for i in range(n_shapes))
    distances: dict = {initial: 0}
    frontier = [initial]
    while frontier:
        next_frontier = []
        for state in frontier:
            d = distances[state]
            if d >= _TIER4_BFS_DEPTH:
                continue
            candidates = []
            if "next_color" in color_vocab:
                # ALL selector: every shape's offset += 1.
                candidates.append(
                    tuple(ColorVal(c.base, (c.offset + 1) % 9) for c in state)
                )
                # by_index(k) for each shape.
                for k in range(n_shapes):
                    ns = list(state)
                    ns[k] = ColorVal(state[k].base, (state[k].offset + 1) % 9)
                    candidates.append(tuple(ns))
            if "swap_colors" in color_vocab:
                for i in range(n_shapes):
                    for j in range(n_shapes):
                        if i == j:
                            continue
                        if state[i] == state[j]:
                            continue  # precondition
                        ns = list(state)
                        ns[i], ns[j] = ns[j], ns[i]
                        candidates.append(tuple(ns))
            for new_state in candidates:
                if new_state not in distances:
                    distances[new_state] = d + 1
                    next_frontier.append(new_state)
        frontier = next_frontier
    return distances


_COLOR_PRIMITIVE_NAMES = frozenset({"next_color", "swap_colors"})
_ALIGN_PRIMITIVE_NAMES = frozenset({"align_row", "align_col"})


def _split_by_color_dim(
    program: list[Step],
) -> tuple[list[Step], list[Step]]:
    """Split program into (color_steps, non_color_steps)."""
    color_steps: list[Step] = []
    non_color: list[Step] = []
    for step in program:
        if step.primitive in _COLOR_PRIMITIVE_NAMES:
            color_steps.append(step)
        else:
            non_color.append(step)
    return color_steps, non_color


def _non_color_per_shape_slice(
    non_color_steps: list[Step], shape_idx: int,
) -> list[Step] | None:
    """Per-shape slice of non-color by_index-only steps."""
    target_param = shape_idx + 1
    result: list[Step] = []
    for step in non_color_steps:
        if step.selector.type != "by_index":
            return None
        if step.target is not None:
            return None  # relational non-color (align) — not handled
        if step.selector.param == target_param:
            result.append(Step(Selector("by_index", 1), step.primitive))
    return result


def is_minimal_tier4_swap(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """Multi-shape analysis for programs with ``swap_colors``.

    Decomposes by dimension: color dim via joint BFS (swap couples
    shapes), other dims via per-shape by_index slicing + single-shape
    canonical forms. Total min-depth is the sum.

    **Scope** (returns True and defers outside):
      - ``n_shapes < 2``,
      - no ``swap_colors`` in vocab,
      - ``align_row``/``align_col`` in vocab (couples row/col — parallel
        joint BFS not yet implemented),
      - any non-``by_index`` selector in the program.
    """
    if not program:
        return True
    if n_shapes < 2:
        return True
    if "swap_colors" not in vocab:
        return True
    # Check the PROGRAM (not vocab) for align — align can't appear in a
    # single-shape slice's shorter equivalent (precondition requires
    # distinct rows, single-shape slicing uses by_index(1) for both
    # src and tgt), so align in vocab but not in program is safe.
    for step in program:
        if step.primitive in _ALIGN_PRIMITIVE_NAMES:
            return True  # align in program couples row dim — defer
        if step.selector.type != "by_index":
            return True
        if step.target is not None and step.target.type != "by_index":
            return True

    _, non_color_steps = _split_by_color_dim(program)

    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False
    target_colors = tuple(s.color for s in traj[-1])

    color_vocab = vocab & _COLOR_PRIMITIVE_NAMES
    color_table = color_joint_bfs_table(color_vocab, n_shapes)
    color_depth = color_table.get(target_colors)
    if color_depth is None:
        return True  # target outside BFS cap

    non_color_vocab = vocab - _COLOR_PRIMITIVE_NAMES
    non_color_total = 0
    for i in range(n_shapes):
        slice_i = _non_color_per_shape_slice(non_color_steps, i)
        if slice_i is None:
            return True
        single_grid = None if grid_shapes is None else [grid_shapes[i]]
        if not _single_shape_canonical_minimal(slice_i, non_color_vocab, single_grid):
            return False
        non_color_total += len(slice_i)

    return len(program) == color_depth + non_color_total


# === Tier 5 — multi-shape align_row/align_col joint BFS ===================


# Align couples row (for align_row) or col (for align_col) across shapes:
# ``align_row(A, B)`` replaces A's row expression with B's current row.
# Under the narrow scope "no translate, no add_border" the row of shape i
# at any point is exactly ``r_{k}`` for some ``k`` — a single-variable
# "row source" label. Joint state is the tuple of row sources and col
# sources per shape. This keeps the BFS state space at ``n_shapes^(2n)``
# — 729 for n=3 — and BFS is essentially free.

_ALIGN_PRIMITIVE_NAMES_T5 = frozenset({"align_row", "align_col"})
_TIER5_BFS_DEPTH = 20


@lru_cache(maxsize=None)
def row_col_source_joint_bfs_table(
    align_vocab: frozenset[str], n_shapes: int,
) -> dict:
    """BFS over joint ``(row_sources, col_sources)`` state under the
    align sub-vocab ⊆ ``{align_row, align_col}``.  Each source is the
    index of the shape whose **initial** row/col the current shape
    has adopted (starts at its own index).

    Enumerates every valid ``align_row(i, j)``/``align_col(i, j)``
    (``i ≠ j`` and sources are currently distinct — precondition).
    Depth cap keeps memory bounded; at n=3 the reachable space is 27
    states per axis, 729 joint, so the cap rarely matters.
    """
    initial_rows = tuple(range(n_shapes))
    initial_cols = tuple(range(n_shapes))
    initial = (initial_rows, initial_cols)
    distances: dict = {initial: 0}
    frontier = [initial]
    while frontier:
        nxt = []
        for state in frontier:
            d = distances[state]
            if d >= _TIER5_BFS_DEPTH:
                continue
            rows, cols = state
            if "align_row" in align_vocab:
                for i in range(n_shapes):
                    for j in range(n_shapes):
                        if i == j:
                            continue
                        if rows[i] == rows[j]:
                            continue
                        new_rows = list(rows)
                        new_rows[i] = rows[j]
                        new_state = (tuple(new_rows), cols)
                        if new_state not in distances:
                            distances[new_state] = d + 1
                            nxt.append(new_state)
            if "align_col" in align_vocab:
                for i in range(n_shapes):
                    for j in range(n_shapes):
                        if i == j:
                            continue
                        if cols[i] == cols[j]:
                            continue
                        new_cols = list(cols)
                        new_cols[i] = cols[j]
                        new_state = (rows, tuple(new_cols))
                        if new_state not in distances:
                            distances[new_state] = d + 1
                            nxt.append(new_state)
        frontier = nxt
    return distances


def _extract_source(expr, var_prefix: str) -> int | None:
    """If ``expr`` is exactly ``1 * <var_prefix>_k`` with zero const,
    return ``k``. Else None."""
    if expr.const != 0:
        return None
    d = expr.as_dict()
    if len(d) != 1:
        return None
    (var, coef), = d.items()
    if coef != 1:
        return None
    if not var.startswith(var_prefix + "_"):
        return None
    try:
        return int(var[len(var_prefix) + 1:])
    except ValueError:
        return None


def _non_align_per_shape_slice(
    program: list[Step], shape_idx: int,
) -> list[Step] | None:
    """Per-shape non-align slice for Tier 5. Filters out align steps
    entirely (handled in the joint BFS) and keeps only by_index steps
    that target ``shape_idx`` (rewritten to ``by_index(1)``).

    Returns ``None`` if the program has any non-align relational or
    non-by_index selector — these require deferral."""
    target_param = shape_idx + 1
    result: list[Step] = []
    for step in program:
        if step.primitive in _ALIGN_PRIMITIVE_NAMES_T5:
            continue
        if step.selector.type != "by_index":
            return None
        if step.target is not None:
            # Other relational (e.g. swap_colors) not handled here.
            return None
        if step.selector.param == target_param:
            result.append(Step(Selector("by_index", 1), step.primitive))
    return result


def is_minimal_tier5_align(
    program: list[Step],
    n_shapes: int,
    *,
    vocab: frozenset[str],
    grid_shapes: list[str] | None = None,
) -> bool:
    """Multi-shape analysis for programs with ``align_row``/``align_col``.

    Decomposes by dimension:
      - **Row/col source dim** (align_row/align_col): joint BFS over
        per-shape row-source + col-source tuples. Small state space.
      - **Other dims** (orient, color, grid): per-shape slice via
        by_index → single-shape Tier 1/2/3 canonical rules.

    Total min-depth = align_joint_depth + sum(non-align per-shape depths).

    **Scope** (returns True and defers outside):
      - ``n_shapes < 2``,
      - no align in vocab,
      - any ``translate_*`` or ``add_border`` in **program** (they shift
        row/col and break the source-label abstraction),
      - ``swap_colors`` in program (handled by Tier 4 instead),
      - any non-by_index selector in the program.
    """
    if not program:
        return True
    if n_shapes < 2:
        return True
    if not (vocab & _ALIGN_PRIMITIVE_NAMES_T5):
        return True
    # Check program scope.
    for step in program:
        if step.selector.type != "by_index":
            return True
        if step.target is not None and step.target.type != "by_index":
            return True
        if step.primitive in _TRANSLATE_PRIMITIVES:
            return True
        if step.primitive in _BORDER_PRIMITIVES:
            return True
        if step.primitive == "swap_colors":
            return True

    try:
        traj = apply_symbolic_program(program, n_shapes, grid_shapes=grid_shapes)
    except PreconditionFail:
        return False

    # Extract per-shape (row_source, col_source) from final state.
    target_rows: list[int] = []
    target_cols: list[int] = []
    for i in range(n_shapes):
        state = traj[-1][i]
        r = _extract_source(state.row, "r")
        c = _extract_source(state.col, "c")
        if r is None or c is None:
            return True  # row/col no longer a clean source — defer
        target_rows.append(r)
        target_cols.append(c)

    align_vocab = vocab & _ALIGN_PRIMITIVE_NAMES_T5
    align_table = row_col_source_joint_bfs_table(align_vocab, n_shapes)
    target = (tuple(target_rows), tuple(target_cols))
    align_depth = align_table.get(target)
    if align_depth is None:
        return True  # out of BFS cap (shouldn't happen for n ≤ 3)

    # Non-align per-shape slices: analyse each via single-shape canonical.
    non_align_vocab = (
        vocab - _ALIGN_PRIMITIVE_NAMES_T5 - _RELATIONAL_PRIMITIVES
    )
    non_align_total = 0
    for i in range(n_shapes):
        slice_i = _non_align_per_shape_slice(program, i)
        if slice_i is None:
            return True
        single_grid = None if grid_shapes is None else [grid_shapes[i]]
        if not _single_shape_canonical_minimal(slice_i, non_align_vocab, single_grid):
            return False
        non_align_total += len(slice_i)

    return len(program) == align_depth + non_align_total
