"""Program-level rule engine.

For the D4-only MVP (Phase 6) we implement:

- :func:`check_count_limits` (R1): per-shape per-primitive application
  counts can't exceed each primitive's cyclic / involution / idempotent
  limit.
- :func:`check_state_revisit` (R3): a shape's symbolic state at step
  ``k`` must differ from its state at every earlier step.
- :func:`check_minimal_word` (R6): each shape's per-primitive count
  equals the word length of the net D4 effect on that shape.

R2 (relational pair count), R4 (dead write), R5 (dependency-aware
translate cancellation) are no-ops under the D4-only vocab used in
Phase 6 and will be added with translate / relational expansion.

Every rule check returns either ``None`` (pass) or a short reason
string (fail). Callers dispatch on truthiness.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional

from babytheorist.core import Program, RELATIONAL_PRIMITIVE_NAMES
from babytheorist.primitives import cyclic_orders, involutions, idempotents

from .d4_tables import (
    D4_ELEMENTS,
    D4_IDENTITY,
    D4_PRIMITIVE_TO_ELEMENT,
    D4_PRIMITIVE_VOCAB,
    compose,
    stabilizers_allowed_at_depth,
    word_length,
)
from .state import FilterState


__all__ = [
    "RuleViolation",
    "check_count_limits",
    "check_state_revisit",
    "check_minimal_word",
    "check_joint_stabilizer_shortcut",
    "check_d4_stabilizer_shortcut",
    "check_translate_cancellation",
    "check_translate_joint_orchestration",
    "check_grid_shape_cancellation",
    "check_add_border_d4_split_shortcut",
    "check_dead_writes",
    "check_trace_identity",
    "program_is_valid",
]


class RuleViolation:
    """Sentinel wrapper so callers can pattern-match rule + reason."""

    __slots__ = ("rule", "reason")

    def __init__(self, rule: str, reason: str):
        self.rule = rule
        self.reason = reason

    def __bool__(self) -> bool:  # for `if violation: ...`
        return True

    def __repr__(self) -> str:
        return f"RuleViolation({self.rule!r}, {self.reason!r})"


# === Per-shape sub-program extraction =====================================


def _per_shape_primitive_counts(
    program: Program, fs: FilterState,
) -> dict[int, Counter]:
    """For each shape index, return Counter({primitive_name: count})
    restricted to steps that affected that shape.

    Uses :data:`fs.history` so the resolver sees each step's
    pre-execution state (dynamic selectors resolved correctly).
    """
    counts: dict[int, Counter] = defaultdict(Counter)
    for step_idx, step in enumerate(program):
        pre_state = fs.history[step_idx]
        self_override = (
            list(fs.self_resolutions[step_idx])
            if step_idx < len(fs.self_resolutions) else None
        )
        indices = _affected_shapes(step, pre_state, self_override=self_override)
        for i in indices:
            counts[i][step.primitive] += 1
    return counts


def _affected_shapes(step, states, *, self_override=None) -> list[int]:
    """Shapes whose state changes due to ``step``.

    Uses :func:`babytheorist.symbolic.step_writes` as the source of truth —
    handles transforms (writes to selector-resolved shapes), movements
    (same), and relationals (swap writes both src + target; align
    writes only src). The result is the sorted set of shape indices
    whose state is modified.

    ``self_override`` substitutes pre-committed indices for the
    step's ``selector`` role (used when the selector reads scene-time
    data and can't be resolved symbolically).
    """
    from babytheorist.symbolic import step_writes
    writes = step_writes(step, list(states), self_override=self_override)
    return sorted({i for (i, _) in writes})


# === R1: count limits =====================================================


def _count_limits() -> dict[str, int]:
    """Per-primitive max count per shape, derived from algebraic metadata.

    rot90 has cyclic_order 4 → max 3 per shape (4 would revisit e).
    flip_h / flip_v involutions → max 1. hollow / fill_interior idempotent
    → max 1. next_color cyclic_order 9 → max 8. Others default to no cap.
    """
    limits: dict[str, int] = {}
    for name, order in cyclic_orders().items():
        # Cyclic of order n: 0..n-1 distinct powers; n applications loop.
        limits[name] = order - 1
    for name in involutions():
        limits[name] = 1
    for name in idempotents():
        limits[name] = 1
    return limits


_COUNT_LIMITS = _count_limits()


def check_count_limits(
    program: Program, fs: FilterState,
) -> Optional[RuleViolation]:
    """R1: per-shape count of each primitive ≤ its algebraic limit."""
    per_shape = _per_shape_primitive_counts(program, fs)
    for shape_idx, counter in per_shape.items():
        for prim, count in counter.items():
            limit = _COUNT_LIMITS.get(prim)
            if limit is None:
                continue
            if count > limit:
                return RuleViolation(
                    "R1",
                    f"shape {shape_idx}: {prim} count {count} exceeds limit {limit}",
                )
    return None


# === R3: state revisit ====================================================


def check_state_revisit(
    program: Program, fs: FilterState,
) -> Optional[RuleViolation]:
    """R3: no shape's per-step state truly repeats within its trajectory.

    Looks at ``fs.history`` — which has len = n_steps + 1 (initial +
    after-each-step). Per-shape trajectory length is the same.

    Consecutive duplicate states (an idle step on an untouched shape)
    are collapsed first, so a shape that is simply not written by a
    step does not count as a revisit. A revisit fires only when a
    shape's state changes and later returns to an earlier value —
    the genuine trajectory-cycle case this rule is meant to catch.

    The joint-stabilizer aliasing that strict-R3 used to mask — where
    a subset of shapes' D4 sub-programs could be re-orchestrated via
    shared stabilizers into a shorter Q — is now covered by I6
    (:func:`check_joint_stabilizer_shortcut`).
    """
    for shape_idx in range(fs.n_shapes):
        traj = fs.shape_trajectory(shape_idx)
        dedup: list = []
        for st in traj:
            if not dedup or dedup[-1] != st:
                dedup.append(st)
        seen: set = set()
        for idx, st in enumerate(dedup):
            if st in seen:
                return RuleViolation(
                    "R3",
                    f"shape {shape_idx}: state at dedupe position {idx} "
                    f"revisits an earlier state",
                )
            seen.add(st)
    return None


# === R6: per-shape minimal-word check (D4 axis) ===========================


def check_minimal_word(
    program: Program,
    fs: FilterState,
    d4_vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
) -> Optional[RuleViolation]:
    """R6: for each shape the D4 sub-program is minimal-length.

    Per-shape D4 count ``d_i`` must equal ``word_length(net_i)`` where
    ``net_i`` is the composition of D4 primitives on shape i.

    For the D4-only vocab all primitives are D4; later vocabs mix in
    non-D4 axes so we restrict to the D4 subset via ``d4_vocab``.
    """
    counts = _per_shape_primitive_counts(program, fs)
    for shape_idx in range(fs.n_shapes):
        d4_count = sum(
            c for name, c in counts[shape_idx].items() if name in d4_vocab
        )
        if d4_count == 0:
            continue
        net = _net_d4_effect(program, fs, shape_idx, d4_vocab)
        wl = word_length(net, d4_vocab)
        if wl != d4_count:
            return RuleViolation(
                "R6",
                f"shape {shape_idx}: {d4_count} D4 primitives reach net "
                f"{net} with word length {wl}",
            )
    return None


def _net_d4_effect(
    program: Program, fs: FilterState, shape_idx: int,
    d4_vocab: frozenset[str],
):
    """Composition of D4 primitives applied to ``shape_idx``.

    Iterates program in order; each affected-by-this-shape D4 step
    left-multiplies the running net.
    """
    net = D4_IDENTITY
    for step_idx, step in enumerate(program):
        if step.primitive not in d4_vocab:
            continue
        pre_state = fs.history[step_idx]
        self_override = (
            list(fs.self_resolutions[step_idx])
            if step_idx < len(fs.self_resolutions) else None
        )
        if shape_idx not in _affected_shapes(
            step, pre_state, self_override=self_override,
        ):
            continue
        g = D4_PRIMITIVE_TO_ELEMENT[step.primitive]
        net = compose(g, net)
    return net


# === I6: joint orchestration shortcut (trace-alignment BFS) ===============


_STATIC_SUBSET_SELECTORS = frozenset({"ALL", "by_index"})


def _i6_selector_subsets(
    selector_types: Iterable[str], n_shapes: int,
) -> list:
    """Shape-index subsets a Q-step could address under
    ``selector_types``. Dynamic selectors (``is_hollow``, positional,
    ``by_color``) can resolve to any non-empty subset under some
    scene — we conservatively include all non-empty subsets whenever
    any dynamic selector is enabled. With only ALL + by_index, the
    subset universe is ``{ALL, {0}, …, {n-1}}`` — ``n+1`` entries.

    Over-approximating the subset universe is sound (admits more
    candidate Q's → more rejections) but never under-approximates
    (which would miss shortcuts).
    """
    selector_types = frozenset(selector_types)
    subsets: list = []
    if "ALL" in selector_types:
        subsets.append(frozenset(range(n_shapes)))
    if "by_index" in selector_types:
        for k in range(n_shapes):
            s = frozenset({k})
            if s not in subsets:
                subsets.append(s)
    dynamic = selector_types - _STATIC_SUBSET_SELECTORS
    if dynamic:
        for mask in range(1, 1 << n_shapes):
            s = frozenset(i for i in range(n_shapes) if mask & (1 << i))
            if s not in subsets:
                subsets.append(s)
    return subsets


def check_joint_stabilizer_shortcut(
    program: Program, fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
) -> Optional[RuleViolation]:
    """I6: reject the program if some Q with length strictly less than
    ``len(program)`` reproduces every shape's ordered write trace
    (primitive + role) via some feasible orchestration.

    **Formulation.** The per-shape write trace is the ordered list of
    ``(primitive, role)`` entries at every step that wrote to the
    shape. Two programs P and Q produce identical per-shape visual
    effects iff they have identical per-shape write traces. The
    minimum-length Q that matches P's traces is a combinatorial
    orchestration: at each Q-step we pick a primitive and a
    selector-expressible subset of shapes; all shapes in the subset
    advance one position along their trace, with the primitive
    matching each advanced shape's next trace entry.

    **Algorithm.** BFS over ``(c_0, c_1, …, c_{n-1})`` state where
    ``c_i`` is the next-unread position in shape ``i``'s trace.
    Start = ``(0, …, 0)``; goal = ``(|trace_0|, …, |trace_{n-1}|)``.
    Edges: for each primitive ``p`` and each expressible subset
    ``S`` such that every ``i ∈ S`` has ``trace_i[c_i] = (p, role)``
    with matching roles consistent across ``S``, transition to
    ``c'_i = c_i + 1`` for ``i ∈ S`` else ``c_i``. Relational
    primitives additionally require ``S`` to contain exactly one
    "self" shape and one "target" shape (the roles).

    If BFS reaches goal in ``< len(program)`` steps → shortcut → reject.

    **State space.** ``∏ (|trace_i| + 1) ≤ (d + 1)^n``. For
    ``n ≤ 4`` and ``d ≤ 6``, this is ≤ 2401 states — trivial.

    **Soundness.** The subset universe is over-approximated under
    dynamic selectors (conservative: admits Q's that may not be
    scene-realizable, increasing rejection rate but never missing
    real shortcuts under a specific scene).

    **Replaces** strict R3's implicit coverage: R3 strict rejected
    every program with any idle shape at any step; this rule
    precisely catches the subset that actually admits a shorter Q.
    """
    from .trace_identity import per_shape_write_trace

    n = fs.n_shapes
    if n == 0:
        return None
    d_program = len(program)
    if d_program <= 1:
        return None

    traces: list[list] = per_shape_write_trace(program, fs)
    if all(len(t) == 0 for t in traces):
        return None

    # Build "advance options" per state. We enumerate moves lazily
    # inside BFS since the move set depends on the current state.
    subset_universe = _i6_selector_subsets(selector_types, n)

    # Convert traces to tuples of (primitive, role) for indexing.
    traces_t: list[tuple] = [tuple(t) for t in traces]
    goal = tuple(len(t) for t in traces_t)
    start = tuple(0 for _ in range(n))
    if start == goal:
        return None

    # BFS with depth cap = d_program - 1 (strictly shorter Q).
    seen = {start}
    frontier: list = [start]
    for depth in range(1, d_program):
        next_frontier: list = []
        for state in frontier:
            # For each primitive the state could advance on:
            # collect shapes eligible for this primitive with their roles.
            prim_to_roles: dict[str, dict[int, str]] = {}
            for i in range(n):
                if state[i] >= len(traces_t[i]):
                    continue
                p_i, role_i = traces_t[i][state[i]]
                prim_to_roles.setdefault(p_i, {})[i] = role_i
            for p, role_map in prim_to_roles.items():
                is_relational = p in RELATIONAL_PRIMITIVE_NAMES
                if is_relational and any(r == "target" for r in role_map.values()):
                    # Relational move with target writes (swap_colors):
                    # pick (src, tgt) where roles match. align_row/align_col
                    # write only src, so they fall through to the subset move.
                    selfs = [i for i, r in role_map.items() if r == "self"]
                    tgts = [i for i, r in role_map.items() if r == "target"]
                    for src in selfs:
                        for tgt in tgts:
                            if src == tgt:
                                continue
                            new_state = list(state)
                            new_state[src] += 1
                            new_state[tgt] += 1
                            new_state = tuple(new_state)
                            if new_state == goal:
                                return RuleViolation(
                                    "I6",
                                    f"orchestration shortcut: program of length "
                                    f"{d_program} has a Q of length {depth} "
                                    f"reproducing all per-shape write traces",
                                )
                            if new_state not in seen:
                                seen.add(new_state)
                                next_frontier.append(new_state)
                else:
                    # Non-relational: every eligible shape has role "self".
                    eligible = frozenset(
                        i for i, r in role_map.items() if r == "self"
                    )
                    if not eligible:
                        continue
                    for subset in subset_universe:
                        S = subset & eligible
                        if not S:
                            continue
                        if S != subset:
                            # The selector's subset would also address
                            # shapes whose next trace entry differs from p
                            # (or already finished) — not a valid move.
                            continue
                        new_state = tuple(
                            state[i] + 1 if i in S else state[i]
                            for i in range(n)
                        )
                        if new_state == goal:
                            return RuleViolation(
                                "I6",
                                f"orchestration shortcut: program of length "
                                f"{d_program} has a Q of length {depth} "
                                f"reproducing all per-shape write traces",
                            )
                        if new_state not in seen:
                            seen.add(new_state)
                            next_frontier.append(new_state)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


# === I6b: joint D4 stabilizer shortcut (coset BFS) ========================


def check_d4_stabilizer_shortcut(
    program: Program, fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    d4_vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
) -> Optional[RuleViolation]:
    """I6b: reject if the program's D4 sub-program admits a shorter
    Q via joint stabilizer cosets.

    Complements :func:`check_joint_stabilizer_shortcut` (I6a). I6a
    aligns per-shape *traces*: it catches orchestration shortcuts
    where the same primitives re-occur and can be merged across
    shapes. I6b aligns per-shape D4 *nets* modulo allowed
    stabilizers: it catches shortcuts where the D4 primitives
    *differ* between P and Q but land in equivalent cosets on a
    stabilizer-constrained scene (e.g. ``[flip_h, rot90, flip_v]``
    vs ``[flip_h, rot90]`` — shape 1's ``rot90·flip_v`` and
    ``flip_h·rot90`` both equal the D4 element ``D4(1, True)``).

    **Algorithm.** Compute per-shape net D4 effect ``g_i`` and D4
    step count ``d_i``. Let allowed
    stabilizers at (g_i, d_i) be ``S_i``. Goal set ``G_i = ∪_{H ∈ S_i}
    g_i · H`` is the set of D4 elements that render indistinguishably
    from ``g_i`` on some shape whose symmetry is in ``S_i``.

    BFS over ``D4^n`` state, start ``(e, …, e)``. At each step pick
    a D4 primitive ``p`` and a selector-expressible subset ``S``;
    transition advances ``compose(p_elt, state_i)`` for ``i ∈ S``.
    Goal: first state where ``state_i ∈ G_i`` for all ``i``. If
    reached in ``< d_D4`` steps → shortcut → reject.

    **Scope.** Runs on pure-D4 and D4 + color-axis programs
    (``next_color`` / ``swap_colors``), because color operations
    commute with orientation. Other mixed-vocab programs are skipped:
    translate / shape-axis steps can depend on intermediate orientation.

    **State space.** ``|D4|^n = 8^n``. For ``n ≤ 4``, ≤ 4096 states.
    Branching factor ``|D4_prims| × |subsets|`` ≤ 3 × (n+1) ≤ 15.
    Depth cap ``d_D4 - 1 ≤ 5``. Trivial — no BFS cost concern.
    """
    n = fs.n_shapes
    if n == 0:
        return None

    color_axis = frozenset({"next_color", "swap_colors"})
    d4_steps = [step for step in program if step.primitive in d4_vocab]
    # Mixed-vocab guard: only color-axis steps commute safely with D4.
    for step in program:
        if step.primitive not in d4_vocab and step.primitive not in color_axis:
            return None

    d_D4 = len(d4_steps)
    if d_D4 < 2:
        return None

    # Per-shape D4 depth + net effect.
    d_i = [0] * n
    for step_idx, step in enumerate(program):
        if step.primitive not in d4_vocab:
            continue
        pre_state = fs.history[step_idx]
        self_override = (
            list(fs.self_resolutions[step_idx])
            if step_idx < len(fs.self_resolutions) else None
        )
        affected = _affected_shapes(
            step, pre_state, self_override=self_override,
        )
        for i in affected:
            d_i[i] += 1

    nets = [_net_d4_effect(program, fs, i, d4_vocab) for i in range(n)]

    # Per-shape goal set G_i = ∪_{H ∈ allowed} g_i · H.
    goal_sets: list[frozenset] = []
    any_active = False
    for i in range(n):
        if d_i[i] == 0:
            # Shape i is untouched under P. Any Q-move on i is
            # equivalent under stabilizer H = D4 (allowed since
            # depth-0 admits every subgroup). Goal set = whole D4.
            goal_sets.append(frozenset(D4_ELEMENTS))
            continue
        any_active = True
        allowed = stabilizers_allowed_at_depth(nets[i], d_i[i], d4_vocab)
        G: set = set()
        for H in allowed:
            for h in H:
                G.add(compose(nets[i], h))
        goal_sets.append(frozenset(G))
    if not any_active:
        return None

    subset_universe = _i6_selector_subsets(selector_types, n)
    d4_prims = [
        (name, D4_PRIMITIVE_TO_ELEMENT[name]) for name in sorted(d4_vocab)
    ]

    start = tuple(D4_IDENTITY for _ in range(n))
    # Degenerate: Q of length 0 already in goal (shouldn't happen when
    # any shape has d_i > 0, but guard anyway).
    if all(start[i] in goal_sets[i] for i in range(n)):
        return RuleViolation(
            "I6b",
            f"D4 stabilizer shortcut: program length {d_D4} collapses to "
            f"identity under allowed stabilizers",
        )

    seen = {start}
    frontier: list = [start]
    for depth in range(1, d_D4):
        next_frontier: list = []
        for state in frontier:
            for _p_name, p_elt in d4_prims:
                for subset in subset_universe:
                    new_state = tuple(
                        compose(p_elt, state[i]) if i in subset else state[i]
                        for i in range(n)
                    )
                    if new_state in seen:
                        continue
                    seen.add(new_state)
                    if all(new_state[i] in goal_sets[i] for i in range(n)):
                        return RuleViolation(
                            "I6b",
                            f"D4 stabilizer shortcut: D4 subword of length "
                            f"{d_D4} has a Q of length {depth} reaching "
                            f"equivalent cosets (nets={nets}, d={d_i})",
                        )
                    next_frontier.append(new_state)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


# === R5: dependency-aware translate cancellation ==========================


_TRANSLATE_COL = {"translate_right": +1, "translate_left": -1}
_TRANSLATE_ROW = {"translate_down": +1, "translate_up": -1}


def check_translate_cancellation(
    program: Program, fs: FilterState,
) -> Optional[RuleViolation]:
    """R5: per shape, per axis (row / col), the number of translate
    primitives applied must equal the minimum count needed to reach the
    observed net LinExpr offset.

    Catches cancellation across transparent intermediates that
    ``babytheorist.program.simplify`` can't see because the translates aren't
    adjacent. Example — ``[trans_right, trans_up, trans_left]``: trans_up
    doesn't touch ``col``, so the right/left pair effectively cancels
    and the program is 2 steps too long.

    Minimum count for net ``α·w_i + β·h_i`` is ``|α| + |β|``: each
    translate contributes exactly one ``±w`` or ``±h`` term depending
    on the shape's orientation parity at that step.
    """
    for shape_idx in range(fs.n_shapes):
        for axis_name, translate_set in (
            ("row", _TRANSLATE_ROW),
            ("col", _TRANSLATE_COL),
        ):
            writes = 0
            for step_idx, step in enumerate(program):
                if step.primitive not in translate_set:
                    continue
                pre = fs.history[step_idx]
                self_override = (
                    list(fs.self_resolutions[step_idx])
                    if step_idx < len(fs.self_resolutions) else None
                )
                if shape_idx not in _affected_shapes(
                    step, pre, self_override=self_override,
                ):
                    continue
                writes += 1
            if writes == 0:
                continue

            initial = fs.history[0][shape_idx]
            final = fs.current[shape_idx]
            delta = getattr(final, axis_name) - getattr(initial, axis_name)
            coeffs = dict(delta.coeffs)
            coeff_h = coeffs.get(initial.initial_h_var, 0)
            coeff_w = coeffs.get(initial.initial_w_var, 0)
            min_word = abs(coeff_h) + abs(coeff_w)

            if writes > min_word:
                return RuleViolation(
                    "R5",
                    f"shape {shape_idx} axis {axis_name}: "
                    f"{writes} translates reach net coeffs "
                    f"(h={coeff_h}, w={coeff_w}) with min-word {min_word}",
                )
    return None


# === R5j: joint axis orchestration (translates + next_color) =============


#: Per-primitive axis metadata: ``(axis_name, sign per step,
#: modulus)``. ``modulus = 0`` means unbounded integer (translates);
#: ``modulus = k`` means target lives in ``Z/kZ`` (involutions:
#: ``k = 2``; ``next_color``: ``k = 9``; ``rot90``: ``k = 4``). The
#: axis label groups primitives that write the same state axis
#: (e.g. ``translate_up`` and ``translate_down`` share the ``row``
#: axis with opposite signs).
#:
#: Only primitives for which a per-shape count fully characterizes
#: the effect are eligible; ``align_*`` / ``swap_colors`` (relational,
#: read cross-shape) and selector-dependent composition are excluded.
_AXIS_INDEPENDENT_PRIMS: dict[str, tuple[str, int, int]] = {
    # Translates — bidirectional signed row / col count.
    "translate_up":    ("row",    -1, 0),
    "translate_down":  ("row",    +1, 0),
    "translate_left":  ("col",    -1, 0),
    "translate_right": ("col",    +1, 0),
    # Color — unidirectional mod 9.
    "next_color":      ("color",  +1, 9),
    # Shape-axis idempotents — unidirectional mod 2 (R1 caps at 1).
    "add_border":      ("border", +1, 2),
    "hollow":          ("hollow", +1, 2),
    "fill_interior":   ("fill",   +1, 2),
    # D4 involutions — unidirectional mod 2. (rot90 is excluded: it
    # is non-involutive and also swaps h / w, breaking commutativity
    # with translate / add_border.)
    "flip_h":          ("flip_h", +1, 2),
    "flip_v":          ("flip_v", +1, 2),
}

#: Pairs of axes that do NOT commute per-shape — if both appear in
#: a program, the axis decomposition is unsound and the rule must
#: skip. Rationale:
#:
#: - ``row`` / ``col`` (translates) read current ``h`` / ``w`` to
#:   compute delta. ``add_border`` writes ``h`` / ``w`` (+2 each), so
#:   border order matters relative to any translate.
#: - ``hollow`` and ``fill_interior`` both write ``grid_shape`` on
#:   the same shape (``hollow`` requires solid→hollow, ``fill``
#:   requires hollow→solid). R5b already catches both-on-same-shape,
#:   but their relative order matters across shapes — keep them
#:   mutually exclusive here.
#: - ``flip_h`` / ``flip_v`` do not commute pairwise (``flip_h ·
#:   flip_v = rot180``), so a program containing both isn't axis-
#:   decomposable.
_AXIS_INCOMPATIBLE_PAIRS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"row", "col"}), frozenset({"add_border"})),
    (frozenset({"hollow"}), frozenset({"fill"})),
    (frozenset({"flip_h"}), frozenset({"flip_v"})),
)


def _min_axis_cost_static(targets: tuple[int, ...]) -> int:
    """Min ``|δ_ALL| + Σ|a_i - δ_ALL|`` over integer ``δ_ALL``.

    Closed-form for the static-selector case (``ALL`` + per-shape
    singletons). Each shape's LP decouples as
    ``δ_ALL + δ_i = a_i`` → ``δ_i = a_i - δ_ALL``; the objective is
    a sum of absolute values in ``δ_ALL``, minimized at a weighted
    median of ``{0, a_0, a_1, …}``. Enumerating those candidates
    gives the exact min.
    """
    if not targets or all(a == 0 for a in targets):
        return 0
    candidates = {0, *targets}
    best: Optional[int] = None
    for d in candidates:
        c = abs(d) + sum(abs(a - d) for a in targets)
        if best is None or c < best:
            best = c
    return best if best is not None else 0


def _min_unidir_axis_cost_static(targets: tuple[int, ...]) -> int:
    """Unidirectional axis (e.g. next_color): ``δ_S ≥ 0`` only.

    With ``ALL`` + singletons available and targets ``a_i ≥ 0``:
    the LP collapses to ``δ_ALL ∈ [0, min a_i]`` maximized, giving
    ``Σ a_i − (n−1)·min(a_i)``. Targets must be non-negative;
    callers are expected to reduce mod cyclic order first if
    needed (R1 caps ``next_color`` at 8 per shape, so this is
    implicit for the curriculum vocab).
    """
    if not targets or all(a == 0 for a in targets):
        return 0
    if any(a < 0 for a in targets):
        # Shouldn't happen — caller bug. Fall back to sum|a_i|.
        return sum(abs(a) for a in targets)
    n = len(targets)
    return sum(targets) - (n - 1) * min(targets)


def check_translate_joint_orchestration(
    program: Program, fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
) -> Optional[RuleViolation]:
    """Joint across-shape orchestration for axis-independent primitives.

    R5 checks per-shape per-axis minimality. This rule catches the
    joint case: the **program-wide** minimum can be lower than the
    sum of per-shape mins when ``by_index`` writes on the same axis
    can be merged into ``ALL`` writes. Examples:

        P = [by_idx(2).tr_right, ALL.tr_up, by_idx(1).tr_right]
        Q = [ALL.tr_right, ALL.tr_up]                     # 2 < 3

        P = [by_idx(1).next_color, ALL.tr_down, by_idx(2).next_color]
        Q = [ALL.next_color, ALL.tr_down]                 # 2 < 3

        P = [by_idx(2).next_color, ALL.add_border, by_idx(1).next_color]
        Q = [ALL.add_border, ALL.next_color]              # 2 < 3

        P = [by_idx(2).tr_right, ALL.flip_v, by_idx(1).tr_right]
        Q = [ALL.flip_v, ALL.tr_right]                    # 2 < 3

    All eligible primitives in :data:`_AXIS_INDEPENDENT_PRIMS`
    decompose per-shape into independent axis counts — so long as
    only a commutative subset appears in any given program (see
    :data:`_AXIS_INCOMPATIBLE_PAIRS`). A Q matching per-shape per-
    axis net produces an identical final scene on each shape.

    **Scope.** Every step must be in :data:`_AXIS_INDEPENDENT_PRIMS`
    (translates, ``next_color``, ``add_border``, ``hollow`` /
    ``fill_interior``, ``flip_h`` / ``flip_v``); rule skips on any
    ``rot90`` or relational step. Further, the *set* of axes
    exercised must avoid every incompatibility in
    :data:`_AXIS_INCOMPATIBLE_PAIRS` — e.g. a program mixing
    translates with ``add_border`` breaks axis decomposition because
    ``add_border`` writes ``h`` / ``w`` that translates read, so
    swapping their order changes translate deltas.

    **Algorithm.** Per axis ``ax``, compute per-shape target
    ``a_{ax, i}`` (reduced mod the axis modulus when unidirectional).
    Solve

        min_{δ_S} Σ_S |δ_S|   s.t.   Σ_{S ∋ i} δ_S = a_{ax, i} ∀i

    over selector-expressible subsets ``S``. With ``ALL`` +
    ``by_index`` the LP has a closed form — bidirectional axes via
    :func:`_min_axis_cost_static`, unidirectional via
    :func:`_min_unidir_axis_cost_static`. Joint
    ``min_Q = Σ_ax min_ax``; rule fires when ``min_Q < len(program)``.

    **Soundness under dynamic selectors.** When ``selector_types``
    includes ``is_hollow`` / ``by_color`` / positional members the
    subset universe is larger and the true LP optimum may be
    strictly smaller, but the static closed form remains an
    *achievable* Q-length — so any rejection is still sound.
    """
    n = fs.n_shapes
    d = len(program)
    if n == 0 or d < 2:
        return None

    # Axis-independent primitive guard.
    for step in program:
        if step.primitive not in _AXIS_INDEPENDENT_PRIMS:
            return None

    # Without ALL, per-axis min under static selectors reduces to
    # Σ|a_i|, which cannot be less than the program's step count —
    # rule never fires.
    if "ALL" not in selector_types:
        return None

    # Per-axis per-shape target accumulation.
    axes_meta: dict[str, int] = {}  # axis → modulus
    per_axis_targets: dict[str, list[int]] = {}
    for step_idx, step in enumerate(program):
        axis, sign, modulus = _AXIS_INDEPENDENT_PRIMS[step.primitive]
        if axis not in per_axis_targets:
            per_axis_targets[axis] = [0] * n
            axes_meta[axis] = modulus
        pre = fs.history[step_idx]
        self_override = (
            list(fs.self_resolutions[step_idx])
            if step_idx < len(fs.self_resolutions) else None
        )
        for i in _affected_shapes(
            step, pre, self_override=self_override,
        ):
            per_axis_targets[axis][i] += sign

    # Commutativity guard: axes present must be pairwise compatible.
    axes_present = frozenset(per_axis_targets)
    for left, right in _AXIS_INCOMPATIBLE_PAIRS:
        if axes_present & left and axes_present & right:
            return None

    joint_min = 0
    reduced_targets: dict[str, tuple[int, ...]] = {}
    for axis, modulus in axes_meta.items():
        targets = per_axis_targets[axis]
        if modulus == 0:
            # Bidirectional unbounded (translates).
            t = tuple(targets)
            reduced_targets[axis] = t
            joint_min += _min_axis_cost_static(t)
        else:
            # Unidirectional mod k: reduce into [0, k).
            t = tuple(a % modulus for a in targets)
            reduced_targets[axis] = t
            joint_min += _min_unidir_axis_cost_static(t)

    if joint_min < d:
        return RuleViolation(
            "R5j",
            f"joint axis orchestration: program length {d} exceeds "
            f"joint min {joint_min} (per-axis targets={reduced_targets})",
        )
    return None


# === R5b: grid_shape cancellation =========================================


def check_grid_shape_cancellation(
    program: Program,
    fs: FilterState,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
) -> Optional[RuleViolation]:
    """Companion to R5 for the grid_shape axis.

    ``[hollow, X, fill_interior]`` returns a shape's grid_shape to
    its initial ``solid`` value — R3 misses this when ``X`` changes
    other axes (border_count / orientation / …), because the full
    state doesn't revisit. With R1 capping each primitive at 1,
    having both ``hollow`` and ``fill_interior`` applied to the same
    shape always means a zero-net round-trip on the grid_shape axis.
    """
    for shape_idx in range(fs.n_shapes):
        has_hollow = False
        has_fill = False
        for step_idx, step in enumerate(program):
            pre = fs.history[step_idx]
            self_override = (
                list(fs.self_resolutions[step_idx])
                if step_idx < len(fs.self_resolutions) else None
            )
            if shape_idx not in _affected_shapes(
                step, pre, self_override=self_override,
            ):
                continue
            if step.primitive == "hollow":
                has_hollow = True
            elif step.primitive == "fill_interior":
                has_fill = True
        if has_hollow and has_fill:
            return RuleViolation(
                "R5b",
                f"shape {shape_idx}: hollow + fill_interior applied — "
                f"grid_shape round-trip is a no-op on that axis",
            )

    if "ALL" in selector_types:
        all_shapes = set(range(fs.n_shapes))
        segment_fill_count = 0
        segment_filled: set[int] = set()

        def flush_fill_segment() -> Optional[RuleViolation]:
            if segment_fill_count >= 2 and segment_filled == all_shapes:
                return RuleViolation(
                    "R5b",
                    "split fill_interior steps cover all shapes and can be "
                    "merged into ALL.fill_interior across D4-only steps",
                )
            return None

        for step_idx, step in enumerate(program):
            if step.primitive == "fill_interior":
                pre = fs.history[step_idx]
                self_override = (
                    list(fs.self_resolutions[step_idx])
                    if step_idx < len(fs.self_resolutions) else None
                )
                affected = set(_affected_shapes(
                    step, pre, self_override=self_override,
                ))
                if affected:
                    segment_fill_count += 1
                    segment_filled |= affected
                continue

            if (
                step.primitive in D4_PRIMITIVE_VOCAB
                and step.selector.type != "is_hollow"
            ):
                continue

            v = flush_fill_segment()
            if v is not None:
                return v
            segment_fill_count = 0
            segment_filled = set()

        v = flush_fill_segment()
        if v is not None:
            return v
    return None


# === R8: add_border / D4 commutation ======================================


def check_add_border_d4_split_shortcut(
    program: Program,
    fs: FilterState,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
) -> Optional[RuleViolation]:
    """Reject split D4/add_border programs that collapse to two ALL steps."""
    if "ALL" not in selector_types or len(program) != 3 or fs.n_shapes < 1:
        return None
    first, middle, last = program
    if middle.primitive != "add_border" or middle.selector.type != "ALL":
        return None
    if first.primitive != last.primitive:
        return None
    if first.primitive not in D4_PRIMITIVE_VOCAB:
        return None
    if first.target is not None or middle.target is not None or last.target is not None:
        return None
    if len(fs.self_resolutions) < 3:
        return None

    all_shapes = set(range(fs.n_shapes))
    first_shapes = set(fs.self_resolutions[0])
    middle_shapes = set(fs.self_resolutions[1])
    last_shapes = set(fs.self_resolutions[2])
    if middle_shapes != all_shapes:
        return None
    if not first_shapes or not last_shapes:
        return None
    if first_shapes.isdisjoint(last_shapes) and first_shapes | last_shapes == all_shapes:
        return RuleViolation(
            "R8",
            "split D4/add_border commutation collapses to ALL D4 + ALL add_border",
        )
    return None


# === R4: dead writes =======================================================


def check_dead_writes(
    program: Program, fs: FilterState,
) -> Optional[RuleViolation]:
    """R4: a step is redundant if every one of its writes is dead — i.e.
    overwritten by a later step before any later step reads the same
    ``(shape, property)``.

    Port of :func:`babytheorist.program.no_dead_writes` re-expressed against
    :class:`FilterState.history` so it slots into ``program_is_valid``
    without recomputing the symbolic trajectory.

    Catches relational dead-code patterns like
    ``align_row(A, B); align_row(A, C)`` (first align's row write
    overwritten without a read), ``translate_up(A); align_row(A, X)``
    (translate discarded by align). Non-relational dead writes
    (``hollow; hollow`` etc.) are already caught by R1; R4 fills the
    gap opened up by Phase 10' relationals.
    """
    from babytheorist.symbolic import step_reads, step_writes

    traj = list(fs.history)  # len = len(program) + 1

    def _so(k: int) -> list[int] | None:
        if k < len(fs.self_resolutions):
            return list(fs.self_resolutions[k])
        return None

    reads_per = [
        step_reads(step, list(traj[k]), self_override=_so(k))
        for k, step in enumerate(program)
    ]
    writes_per = [
        step_writes(step, list(traj[k]), self_override=_so(k))
        for k, step in enumerate(program)
    ]

    for k in range(len(program)):
        writes_k = writes_per[k]
        if not writes_k:
            continue
        all_dead = True
        for target in writes_k:
            is_dead = False
            for m in range(k + 1, len(program)):
                if target in reads_per[m]:
                    break  # live — consumed by a later step
                if target in writes_per[m]:
                    is_dead = True
                    break  # dead — overwritten without intervening read
            if not is_dead:
                all_dead = False
                break
        if all_dead:
            return RuleViolation(
                "R4",
                f"step {k} ({program[k].primitive}): all writes {sorted(writes_k)} "
                f"dead (overwritten before read)",
            )
    return None


# === R7': trace-identity (selector-union aliasing) ========================


def check_trace_identity(
    program: Program,
    fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> Optional[RuleViolation]:
    """R7': delegate to :func:`babytheorist.filter.trace_identity.check_trace_identity`.

    See that module's docstring for the rule definition. The indirection
    lives here so ``program_is_valid`` has a single entry point for all
    rule dispatches.
    """
    from .trace_identity import check_trace_identity as _impl
    return _impl(
        program, fs, selector_types=selector_types,
        constraint=constraint, palette=palette,
    )


# === Aggregate ============================================================


def program_is_valid(
    program: Program, fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> Optional[RuleViolation]:
    """Run all program-level rules in cheap-first order.

    Returns the first :class:`RuleViolation` encountered, or ``None``
    if all rules pass.

    ``constraint`` / ``palette`` reach R7' so it can reason about
    by_color subset expressibility (which depends on accumulated
    per-shape primary_color requirements). When omitted, R7' skips
    the by_color branch and only checks ALL / is_hollow expressibility.
    """
    v = check_count_limits(program, fs)
    if v is not None:
        return v
    v = check_state_revisit(program, fs)
    if v is not None:
        return v
    v = check_minimal_word(program, fs)
    if v is not None:
        return v
    v = check_joint_stabilizer_shortcut(
        program, fs, selector_types=selector_types,
    )
    if v is not None:
        return v
    v = check_d4_stabilizer_shortcut(
        program, fs, selector_types=selector_types,
    )
    if v is not None:
        return v
    v = check_translate_cancellation(program, fs)
    if v is not None:
        return v
    v = check_translate_joint_orchestration(
        program, fs, selector_types=selector_types,
    )
    if v is not None:
        return v
    v = check_grid_shape_cancellation(
        program, fs, selector_types=selector_types,
    )
    if v is not None:
        return v
    v = check_add_border_d4_split_shortcut(
        program, fs, selector_types=selector_types,
    )
    if v is not None:
        return v
    v = check_dead_writes(program, fs)
    if v is not None:
        return v
    v = check_trace_identity(
        program, fs, selector_types=selector_types,
        constraint=constraint, palette=palette,
    )
    if v is not None:
        return v
    return None
