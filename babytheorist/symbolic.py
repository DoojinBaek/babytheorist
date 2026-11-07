"""Symbolic shape-state tracking for program-level filtering (Phase 1).

Each shape carries a per-step *symbolic state* — a tuple chosen so that
two states compare equal iff the shape has returned to a previously-seen
configuration under the program. The trajectory-distinct check in
`babytheorist.program` walks these states and rejects a program if any shape
revisits an earlier state (which would mean the program had an
identity-producing subsequence on that shape).

Fields tracked per shape:

    - orientation     D4 element (rot90 + flip composition)
    - row, col        linear expression over input variables
    - border_count    number of add_border applications
    - color           (base shape id, offset mod 9) for primary_color
    - grid_shape      "solid" or "hollow"

Initial size (h_i, w_i), initial position (r_i, c_i), and initial color
(base=i, offset=0) are encoded as input variables so translate/align
effects compose as linear expressions.

V1 scope: static selectors only (ALL, by_index). Dynamic selectors
(topmost/bottommost/leftmost/rightmost/by_color/is_hollow) raise
NotImplementedError — symbolic resolution for them depends on mid-program
state and is deferred to V2.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

from .core import Selector, Step


__all__ = [
    "LinExpr", "D4", "D4_IDENTITY", "ColorVal", "ShapeState",
    "PreconditionFail",
    "initial_state", "initial_states",
    "current_height", "current_width",
    "resolve_symbolic", "write_targets",
    "apply_symbolic_step", "apply_symbolic_program",
    "PRIMITIVE_IO", "step_reads", "step_writes",
]


# === LinExpr ===============================================================


@dataclass(frozen=True)
class LinExpr:
    """Linear expression ``sum(coeffs[v] * v) + const`` over input vars.

    Stored in canonical form: ``coeffs`` is a tuple of ``(var, coeff)``
    sorted by ``var`` with zero-coeff entries dropped. Equality is
    syntactic, which matches the "generic equality" semantics we want for
    trajectory distinct: two states are equal iff the expressions agree
    for *every* valid assignment to the input variables.
    """

    coeffs: tuple[tuple[str, int], ...]
    const: int = 0

    @classmethod
    def from_dict(cls, d: Mapping[str, int], const: int = 0) -> "LinExpr":
        items = tuple(sorted((k, v) for k, v in d.items() if v != 0))
        return cls(items, const)

    @classmethod
    def var(cls, name: str) -> "LinExpr":
        return cls(((name, 1),), 0)

    @classmethod
    def constant(cls, c: int) -> "LinExpr":
        return cls((), c)

    def as_dict(self) -> dict[str, int]:
        return dict(self.coeffs)

    def __add__(self, other: "LinExpr") -> "LinExpr":
        d = self.as_dict()
        for k, v in other.coeffs:
            d[k] = d.get(k, 0) + v
        return LinExpr.from_dict(d, self.const + other.const)

    def __sub__(self, other: "LinExpr") -> "LinExpr":
        return self + (-other)

    def __neg__(self) -> "LinExpr":
        return LinExpr(tuple((k, -v) for k, v in self.coeffs), -self.const)


# === D4 ====================================================================


@dataclass(frozen=True)
class D4:
    """Element of the dihedral group D4 as ``r^rot · s^flipped`` where
    ``r`` is rot90 (CCW quarter turn) and ``s`` is flip_h.

    D4 has the relations ``r^4 = s^2 = e`` and ``s r s = r^{-1}``, which
    give the three transitions below. Each ``apply_*`` returns the
    composition *p ∘ g* — i.e. the new net orientation after applying
    primitive ``p`` on top of the current orientation ``g``.
    """

    rot: int
    flipped: bool

    def apply_rot(self) -> "D4":
        return D4((self.rot + 1) % 4, self.flipped)

    def apply_flip_h(self) -> "D4":
        # s · (r^a · s^f) = r^{-a} · s^{1-f}
        return D4((-self.rot) % 4, not self.flipped)

    def apply_flip_v(self) -> "D4":
        # flip_v = r^2 · s, so flip_v · (r^a · s^f) = r^{2-a} · s^{1-f}
        return D4((2 - self.rot) % 4, not self.flipped)


D4_IDENTITY = D4(0, False)


# === Color =================================================================


@dataclass(frozen=True)
class ColorVal:
    """Symbolic primary_color: ``(base, offset mod 9)``.

    ``base`` identifies the shape whose initial color this color
    descends from. ``offset`` counts next_color cycles (mod 9). Two
    shapes have equal colors iff the tuples match.
    """

    base: str
    offset: int


# === ShapeState ============================================================


@dataclass(frozen=True)
class ShapeState:
    """Per-shape symbolic state. Equality is field-wise."""

    initial_h_var: str
    initial_w_var: str
    orientation: D4
    border_count: int
    row: LinExpr
    col: LinExpr
    color: ColorVal
    grid_shape: str  # "solid" | "hollow"


def _shape_id(i: int) -> str:
    return str(i)


def initial_state(i: int, *, grid_shape: str = "solid") -> ShapeState:
    """Initial symbolic state for the ``i``-th shape in the scene."""
    sid = _shape_id(i)
    return ShapeState(
        initial_h_var=f"h_{sid}",
        initial_w_var=f"w_{sid}",
        orientation=D4_IDENTITY,
        border_count=0,
        row=LinExpr.var(f"r_{sid}"),
        col=LinExpr.var(f"c_{sid}"),
        color=ColorVal(sid, 0),
        grid_shape=grid_shape,
    )


def initial_states(n: int, *, grid_shapes: list[str] | None = None) -> list[ShapeState]:
    """Initial states for an ``n``-shape scene.

    ``grid_shapes`` lets the caller declare per-shape initial grid_shape
    (default all ``"solid"``). This lets Phase 1 reject programs that
    become identity under a known mix of initial shape types. For fully
    unknown initial mix, leave as default and rely on Phase 2's
    grid_sequence effectiveness check as a backstop.
    """
    if grid_shapes is None:
        grid_shapes = ["solid"] * n
    if len(grid_shapes) != n:
        raise ValueError(f"grid_shapes length {len(grid_shapes)} != n {n}")
    return [initial_state(i, grid_shape=grid_shapes[i]) for i in range(n)]


# === Size helpers ==========================================================


def current_height(s: ShapeState) -> LinExpr:
    """Current height under rotation + border_count."""
    base = s.initial_h_var if s.orientation.rot % 2 == 0 else s.initial_w_var
    return LinExpr.from_dict({base: 1}, 2 * s.border_count)


def current_width(s: ShapeState) -> LinExpr:
    """Current width under rotation + border_count."""
    base = s.initial_w_var if s.orientation.rot % 2 == 0 else s.initial_h_var
    return LinExpr.from_dict({base: 1}, 2 * s.border_count)


# === Precondition + per-primitive transitions ==============================


class PreconditionFail(Exception):
    """Raised when a step's precondition is violated symbolically.

    Signals that the program cannot execute past this step under the
    current symbolic state — equivalent to the engine returning ``None``.
    """


_TRANSFORMS = frozenset({
    "rot90", "flip_h", "flip_v", "next_color",
    "fill_interior", "hollow", "add_border",
})
_MOVEMENTS = frozenset({
    "translate_up", "translate_down", "translate_left", "translate_right",
})
_RELATIONALS = frozenset({"swap_colors", "align_row", "align_col"})


def _apply_transform(name: str, s: ShapeState) -> ShapeState:
    if name == "rot90":
        return replace(s, orientation=s.orientation.apply_rot())
    if name == "flip_h":
        return replace(s, orientation=s.orientation.apply_flip_h())
    if name == "flip_v":
        return replace(s, orientation=s.orientation.apply_flip_v())
    if name == "next_color":
        return replace(s, color=ColorVal(s.color.base, (s.color.offset + 1) % 9))
    if name == "hollow":
        return replace(s, grid_shape="hollow")
    if name == "fill_interior":
        if s.grid_shape != "hollow":
            raise PreconditionFail(
                f"fill_interior requires hollow grid_shape, got {s.grid_shape!r}"
            )
        return replace(s, grid_shape="solid")
    if name == "add_border":
        return replace(
            s,
            border_count=s.border_count + 1,
            row=s.row - LinExpr.constant(1),
            col=s.col - LinExpr.constant(1),
        )
    raise ValueError(f"unknown transform primitive: {name!r}")


def _apply_movement(name: str, s: ShapeState) -> ShapeState:
    if name == "translate_up":
        return replace(s, row=s.row - current_height(s))
    if name == "translate_down":
        return replace(s, row=s.row + current_height(s))
    if name == "translate_left":
        return replace(s, col=s.col - current_width(s))
    if name == "translate_right":
        return replace(s, col=s.col + current_width(s))
    raise ValueError(f"unknown movement primitive: {name!r}")


def _apply_relational(
    name: str, src: ShapeState, tgt: ShapeState
) -> tuple[ShapeState, ShapeState]:
    if name == "swap_colors":
        if src.color == tgt.color:
            raise PreconditionFail(
                f"swap_colors requires distinct colors; both {src.color}"
            )
        return replace(src, color=tgt.color), replace(tgt, color=src.color)
    if name == "align_row":
        if src.row == tgt.row:
            raise PreconditionFail(
                f"align_row requires distinct rows; both {src.row}"
            )
        return replace(src, row=tgt.row), tgt
    if name == "align_col":
        if src.col == tgt.col:
            raise PreconditionFail(
                f"align_col requires distinct cols; both {src.col}"
            )
        return replace(src, col=tgt.col), tgt
    raise ValueError(f"unknown relational primitive: {name!r}")


# === Selector resolution (static only) =====================================


_STATIC_SELECTORS = frozenset({"ALL", "by_index"})
# Dynamic selectors that depend on grid_shape only — supported via
# symbolic resolution since grid_shape is a concrete {"solid", "hollow"}
# attribute tracked in ShapeState. Other dynamic selectors
# (topmost/bottommost/leftmost/rightmost/by_color) depend on
# symbolic row/col/color expressions whose ordering or concrete value
# isn't decidable without scene data, so they remain unsupported.
_DYNAMIC_SELECTORS_SUPPORTED = frozenset({"is_hollow"})
_SYMBOLIC_SELECTORS = _STATIC_SELECTORS | _DYNAMIC_SELECTORS_SUPPORTED


def resolve_symbolic(selector: Selector, states: list[ShapeState]) -> list[int]:
    """Resolve a selector on symbolic states.

    Supports the static selectors ``ALL`` and ``by_index``, plus the
    dynamic selector ``is_hollow`` (which reads ``grid_shape``, a
    concrete attribute we track symbolically). Other dynamic selectors
    raise ``NotImplementedError``.
    """
    if selector.type not in _SYMBOLIC_SELECTORS:
        raise NotImplementedError(
            f"symbolic resolution for {selector.type!r} not supported "
            f"(supported: {sorted(_SYMBOLIC_SELECTORS)})"
        )
    if selector.type == "ALL":
        return list(range(len(states)))
    if selector.type == "by_index":
        k = selector.param
        return [k - 1] if 1 <= k <= len(states) else []
    if selector.type == "is_hollow":
        return [i for i, s in enumerate(states) if s.grid_shape == "hollow"]
    raise AssertionError("unreachable")


def write_targets(step: Step, states: list[ShapeState]) -> set[int]:
    """Indices of shapes whose state this step will modify.

    Uses per-primitive write-set knowledge: transforms and movements
    write to every shape matched by ``step.selector``; ``swap_colors``
    writes to both src and target; ``align_row`` / ``align_col`` write
    only to src (the target is read-only).
    """
    name = step.primitive
    if name in _RELATIONALS:
        src = set(resolve_symbolic(step.selector, states))
        if name == "swap_colors":
            tgt = set(resolve_symbolic(step.target, states))
            return src | tgt
        return src
    return set(resolve_symbolic(step.selector, states))


# === Dataflow metadata (Rule 2) ============================================


# Per-primitive read/write sets. Each entry: ``("self" | "target", property)``.
# "self" = every shape matched by ``step.selector``. "target" = every shape
# matched by ``step.target`` (only valid for relational primitives).
#
# **"reads" are computational reads only** — properties whose values the
# primitive's output actually depends on. Precondition-only checks (e.g.
# ``align_row``'s ``src.row != tgt.row`` inequality guard, ``swap_colors``'s
# color inequality guard) are *not* included: they don't affect output and
# shouldn't keep a prior write alive. See ``no_dead_writes`` in
# ``babytheorist.program`` for how this is consumed.
PRIMITIVE_IO: dict[str, dict[str, tuple[tuple[str, str], ...]]] = {
    # Transforms that compose (read + write same property)
    "rot90":     {"reads": (("self", "orientation"),), "writes": (("self", "orientation"),)},
    "flip_h":    {"reads": (("self", "orientation"),), "writes": (("self", "orientation"),)},
    "flip_v":    {"reads": (("self", "orientation"),), "writes": (("self", "orientation"),)},
    "next_color": {"reads": (("self", "color"),),       "writes": (("self", "color"),)},

    # Grid_shape: hollow is a pure overwrite (output doesn't depend on prior
    # grid_shape); fill_interior reads grid_shape to locate the interior.
    "hollow":        {"reads": (),                             "writes": (("self", "grid_shape"),)},
    "fill_interior": {"reads": (("self", "grid_shape"),),      "writes": (("self", "grid_shape"),)},

    # add_border shifts row/col by -1 and bumps border_count.
    "add_border": {
        "reads":  (("self", "border_count"), ("self", "row"), ("self", "col")),
        "writes": (("self", "border_count"), ("self", "row"), ("self", "col")),
    },

    # Translations: delta depends on current_height/width, which depend on
    # orientation parity and border_count.
    "translate_up": {
        "reads":  (("self", "row"), ("self", "orientation"), ("self", "border_count")),
        "writes": (("self", "row"),),
    },
    "translate_down": {
        "reads":  (("self", "row"), ("self", "orientation"), ("self", "border_count")),
        "writes": (("self", "row"),),
    },
    "translate_left": {
        "reads":  (("self", "col"), ("self", "orientation"), ("self", "border_count")),
        "writes": (("self", "col"),),
    },
    "translate_right": {
        "reads":  (("self", "col"), ("self", "orientation"), ("self", "border_count")),
        "writes": (("self", "col"),),
    },

    # Relational
    "swap_colors": {
        "reads":  (("self", "color"), ("target", "color")),
        "writes": (("self", "color"), ("target", "color")),
    },
    # align_row/col: pure overwrite of src's row/col from target's. src's
    # own row/col is read only for the inequality precondition — omitted
    # from computational reads so align chains are correctly dead-coded.
    "align_row": {
        "reads":  (("target", "row"),),
        "writes": (("self", "row"),),
    },
    "align_col": {
        "reads":  (("target", "col"),),
        "writes": (("self", "col"),),
    },
}


def _resolve_io_pairs(
    step: Step,
    states: list[ShapeState],
    pairs: tuple[tuple[str, str], ...],
    *,
    self_override: list[int] | None = None,
    target_override: list[int] | None = None,
) -> set[tuple[int, str]]:
    """Expand ``("self"|"target", property)`` pairs to concrete
    ``(shape_index, property)`` tuples via selector resolution.

    ``self_override`` / ``target_override`` let the filter layer
    substitute pre-committed indices for selectors that can't be
    resolved symbolically (``by_color``, positional).
    """
    result: set[tuple[int, str]] = set()
    for role, prop in pairs:
        if role == "self":
            if self_override is not None:
                indices = self_override
            else:
                indices = resolve_symbolic(step.selector, states)
            for i in indices:
                result.add((i, prop))
        elif role == "target":
            if step.target is None:
                raise ValueError(
                    f"primitive {step.primitive!r} IO references 'target' "
                    f"but step has no target selector"
                )
            if target_override is not None:
                indices = target_override
            else:
                indices = resolve_symbolic(step.target, states)
            for i in indices:
                result.add((i, prop))
        else:
            raise ValueError(f"unknown IO role: {role!r}")
    return result


def step_reads(
    step: Step,
    states: list[ShapeState],
    *,
    self_override: list[int] | None = None,
    target_override: list[int] | None = None,
) -> set[tuple[int, str]]:
    """Set of ``(shape_idx, property)`` computationally read by ``step``.

    Used by Rule 2 (``no_dead_writes``). Precondition-only reads are
    deliberately excluded from ``PRIMITIVE_IO`` so the analysis catches
    overwrite-chains like ``align(A,B); align(A,C)``.
    """
    return _resolve_io_pairs(
        step, states, PRIMITIVE_IO[step.primitive]["reads"],
        self_override=self_override, target_override=target_override,
    )


def step_writes(
    step: Step,
    states: list[ShapeState],
    *,
    self_override: list[int] | None = None,
    target_override: list[int] | None = None,
) -> set[tuple[int, str]]:
    """Set of ``(shape_idx, property)`` written by ``step``."""
    return _resolve_io_pairs(
        step, states, PRIMITIVE_IO[step.primitive]["writes"],
        self_override=self_override, target_override=target_override,
    )


# === Step application ======================================================


def apply_symbolic_step(
    step: Step,
    states: list[ShapeState],
    *,
    override_indices: list[int] | None = None,
) -> list[ShapeState]:
    """Apply one step to symbolic states.

    Returns a new list of states (input not mutated). Raises
    ``PreconditionFail`` if a symbolic precondition is violated (e.g.
    fill_interior on a known-solid shape, align_row between equal
    symbolic rows).

    ``override_indices`` (filter-internal) bypasses the normal
    ``resolve_symbolic`` path for non-relational primitives. Used by
    :class:`babytheorist.filter.FilterState` when a selector (e.g.
    ``by_color``, positional) isn't symbolically resolvable and the
    sampler has externally committed to a specific shape subset.
    Ignored for relational primitives.
    """
    name = step.primitive

    if name in _RELATIONALS:
        src_idx = resolve_symbolic(step.selector, states)
        tgt_idx = resolve_symbolic(step.target, states)
        if len(src_idx) != 1 or len(tgt_idx) != 1:
            raise PreconditionFail(
                f"relational {name!r} requires single-index src and tgt; "
                f"got src={src_idx}, tgt={tgt_idx}"
            )
        i, j = src_idx[0], tgt_idx[0]
        if i == j:
            raise PreconditionFail(
                f"relational {name!r} requires distinct objects; both index {i}"
            )
        out = list(states)
        new_src, new_tgt = _apply_relational(name, out[i], out[j])
        out[i] = new_src
        out[j] = new_tgt
        return out

    if override_indices is not None:
        indices = list(override_indices)
    else:
        indices = resolve_symbolic(step.selector, states)
    if not indices:
        raise PreconditionFail(f"step {name!r}: selector resolved to no shapes")

    out = list(states)
    if name in _TRANSFORMS:
        for i in indices:
            out[i] = _apply_transform(name, out[i])
    elif name in _MOVEMENTS:
        for i in indices:
            out[i] = _apply_movement(name, out[i])
    else:
        raise ValueError(f"unknown primitive: {name!r}")
    return out


def apply_symbolic_program(
    program: list[Step],
    n_shapes: int,
    *,
    grid_shapes: list[str] | None = None,
) -> list[list[ShapeState]]:
    """Return the full trajectory ``[state_0, state_1, ..., state_k]``.

    ``state_0`` is the initial state (pre-program); ``state_i`` is the
    state after the first ``i`` steps. Raises ``PreconditionFail`` if
    any step is symbolically infeasible.
    """
    states = initial_states(n_shapes, grid_shapes=grid_shapes)
    trajectory = [states]
    for step in program:
        states = apply_symbolic_step(step, states)
        trajectory.append(states)
    return trajectory
