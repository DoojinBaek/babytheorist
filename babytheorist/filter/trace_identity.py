"""Trace-identity rule (R7').

Replaces the Phase 6 joint BFS with a symbolic per-shape trace
comparison. The rule catches selector-union aliasing (G1): when
multiple shapes share the exact same ordered write-trace, the
program can be shortened by replacing their per-shape selectors
with a single wider selector.

Trace entry granularity is ``(primitive, role)`` where
``role ∈ {"self", "target"}`` per :mod:`babytheorist.symbolic` PRIMITIVE_IO.
Selector identity is deliberately omitted — it's exactly what the
aliasing rule is supposed to erase.

Expressibility check
--------------------

A group ``G`` of shapes with identical trace is "expressible" at a
program step ``k`` if some selector in ``selector_types`` resolves
exactly to ``G`` at the pre-state ``fs.history[k]``. Currently
supported expressibility selectors:

- ``ALL``: expressible iff ``G == {0, …, n-1}``.
- ``is_hollow``: expressible iff
  ``{i : fs.history[k][i].grid_shape == "hollow"} == G``.

``by_index`` is never a group-expressing selector (it's a singleton
selector, not a subset selector).

When expanding to dynamic selectors that read scene-time data
(``by_color``, ``topmost``, …), extend ``_group_expressible_at``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from babytheorist.core import Program
from babytheorist.symbolic import PRIMITIVE_IO, ShapeState, resolve_symbolic

from .state import FilterState


__all__ = [
    "TraceEntry",
    "per_shape_write_trace",
    "check_trace_identity",
]


#: ``(primitive_name, role)`` where role is ``"self"`` or ``"target"``.
TraceEntry = tuple[str, str]


def _step_writes_by_role(
    step, states, *, self_override: list[int] | None = None,
) -> dict[int, set[str]]:
    """For a single step, return ``{shape_idx: {roles}}``.

    Unlike :func:`babytheorist.symbolic.step_writes` which flattens to
    ``(shape_idx, property)`` pairs, this preserves the role so the
    trace can distinguish e.g. ``swap_colors`` writing to src
    (``"self"``) vs target (``"target"``).

    ``self_override`` substitutes pre-committed indices for the
    selector's "self" role — needed for selectors that aren't
    symbolically resolvable (``by_color``, positional).
    """
    result: dict[int, set[str]] = defaultdict(set)
    for role, _prop in PRIMITIVE_IO[step.primitive]["writes"]:
        if role == "self":
            if self_override is not None:
                indices = self_override
            else:
                indices = resolve_symbolic(step.selector, states)
            for i in indices:
                result[i].add("self")
        elif role == "target":
            if step.target is None:
                continue
            for i in resolve_symbolic(step.target, states):
                result[i].add("target")
        else:
            raise ValueError(f"unknown IO role: {role!r}")
    return dict(result)


def _self_override_for(fs: FilterState, k: int) -> list[int] | None:
    if k < len(fs.self_resolutions):
        return list(fs.self_resolutions[k])
    return None


def per_shape_write_trace(
    program: Program, fs: FilterState,
) -> list[list[TraceEntry]]:
    """Ordered ``(primitive, role)`` entries per shape, covering every
    step that wrote to the shape (in some role)."""
    n = fs.n_shapes
    traces: list[list[TraceEntry]] = [[] for _ in range(n)]
    for k, step in enumerate(program):
        pre = list(fs.history[k])
        by_role = _step_writes_by_role(
            step, pre, self_override=_self_override_for(fs, k),
        )
        for i, roles in by_role.items():
            for role in sorted(roles):
                traces[i].append((step.primitive, role))
    return traces


def _steps_writing_to_group(
    program: Program, fs: FilterState, group: frozenset[int],
) -> list[int]:
    """Indices of program steps that write to at least one member of
    ``group``. The trace-grouping invariant guarantees that each such
    step writes to ALL members of ``group`` (see module docstring).
    """
    step_indices: list[int] = []
    for k, step in enumerate(program):
        pre = list(fs.history[k])
        by_role = _step_writes_by_role(
            step, pre, self_override=_self_override_for(fs, k),
        )
        written = set(by_role.keys())
        if written & group:
            step_indices.append(k)
    return step_indices


def _by_color_expressible_at(
    group: frozenset[int],
    pre: list[ShapeState],
    constraint,
    palette: tuple[int, ...],
) -> bool:
    """Return True iff some ``by_color(c)`` at pre-state ``pre``
    deterministically resolves to ``group`` under ``constraint``.

    "Deterministically" means the constraint's per-shape
    ``required_primary_color`` + ``forbidden_primary_colors`` force
    the (c, group) distribution — no valid scene can contradict it.
    When the constraint leaves some shape's primary color
    unconstrained, we skip the by_color branch (conservative: this
    may miss a collapsible case whose shortcut manifests only for
    certain scenes, but the oracle catches those).
    """
    for c in palette:
        ok = True
        for i, state in enumerate(pre):
            offset = state.color.offset
            target = (c - offset) % 9
            required = constraint.shapes[i].required_primary_color
            forbidden = constraint.shapes[i].forbidden_primary_colors
            if i in group:
                # Must deterministically equal `target`.
                if required != target:
                    ok = False
                    break
            else:
                # Must deterministically differ from `target`.
                if target not in palette:
                    continue  # auto-satisfied: scene can't draw target
                if required is not None:
                    if required == target:
                        ok = False
                        break
                    continue  # required != target
                if target in forbidden:
                    continue
                # required is None and target is not forbidden — scene
                # could still draw target. Not deterministic.
                ok = False
                break
        if ok:
            return True
    return False


def _group_expressible_at(
    group: frozenset[int],
    pre: list[ShapeState],
    selector_types: frozenset[str],
    *,
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> bool:
    """Return True if some selector in ``selector_types`` resolves
    exactly to ``group`` at symbolic pre-state ``pre``.

    Supports ``ALL``, ``is_hollow``, and ``by_color`` (the latter
    only when ``constraint`` and ``palette`` are provided — the
    check depends on accumulated per-shape primary_color
    requirements). Extend here when positional selectors land.
    """
    n = len(pre)
    if "ALL" in selector_types and group == frozenset(range(n)):
        return True
    if "is_hollow" in selector_types:
        hollow = frozenset(
            i for i, s in enumerate(pre) if s.grid_shape == "hollow"
        )
        if hollow == group:
            return True
    if "by_color" in selector_types and constraint is not None \
            and palette is not None:
        if _by_color_expressible_at(group, pre, constraint, palette):
            return True
    return False


def _group_expressible_throughout(
    program: Program,
    fs: FilterState,
    group: frozenset[int],
    selector_types: frozenset[str],
    *,
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> bool:
    """Check that at every step writing to ``group``, the group is
    expressible via a single selector in ``selector_types``.

    A subset is "collapsible" only when the filter could have picked
    one wider selector at each relevant step — otherwise no valid
    shorter Q exists.
    """
    write_steps = _steps_writing_to_group(program, fs, group)
    for k in write_steps:
        pre = list(fs.history[k])
        if not _group_expressible_at(
            group, pre, selector_types,
            constraint=constraint, palette=palette,
        ):
            return False
    return True


def check_trace_identity(
    program: Program,
    fs: FilterState,
    *,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    constraint=None,
    palette: tuple[int, ...] | None = None,
) -> Optional[object]:
    """R7': flag programs where some group of ≥2 shapes share an
    identical write-trace AND that group is expressible as a single
    subset-capable selector at every step that wrote to it.

    Returns ``None`` on pass or a :class:`RuleViolation` on fail.

    Length gate. Let ``k = |trace(G)|`` be the number of distinct
    program steps writing to each member of ``G`` (equal across
    members by the grouping invariant). The collapsed version of
    ``P`` using one wider selector per trace entry writes to ``G``
    in exactly ``k`` steps, leaving the other shapes' sub-program
    untouched. If ``P`` used per-shape ``by_index`` steps within
    ``G`` — i.e. more than ``k`` distinct steps wrote to some member
    of ``G`` — there are savings, so the program is not shortest.
    Formally: flag when the number of by_index steps in ``P`` whose
    resolved shape lies in ``G`` is strictly greater than ``0`` and
    at least one such step's trace entry is shared by another
    ``G``-member (already guaranteed by identical-trace grouping).
    """
    n = fs.n_shapes
    if n < 2:
        return None

    traces = per_shape_write_trace(program, fs)

    from .rules_program import RuleViolation

    for trace_view, reason in (
        (traces, "share trace"),
        (
            [_next_color_normalized_trace(t) for t in traces],
            "share next_color-normalized trace",
        ),
    ):
        groups: dict[tuple[TraceEntry, ...], list[int]] = defaultdict(list)
        for i, t in enumerate(trace_view):
            groups[tuple(t)].append(i)

        for trace_key, members in groups.items():
            if len(members) < 2:
                continue
            if not trace_key:
                continue  # no writes → nothing to collapse

            group = frozenset(members)
            if not _group_expressible_throughout(
                program, fs, group, selector_types,
                constraint=constraint, palette=palette,
            ):
                continue

            if _has_by_index_write_on_group(program, fs, group):
                return RuleViolation(
                    "R7'",
                    f"shapes {sorted(members)} {reason} {list(trace_key)} "
                    f"and are expressible as a single selector; program uses "
                    f"by_index writes on the group — collapsible to a wider "
                    f"selector",
                )

    return None


def _next_color_normalized_trace(trace: list[TraceEntry]) -> list[TraceEntry]:
    """Canonicalize the position of next_color writes.

    ``next_color`` commutes with geometry and movement primitives in the current
    DSL, so traces that differ only by where color-cycling appears can still be
    selector-union aliases. Keep role information and multiplicity intact.
    """
    non_color = [entry for entry in trace if entry[0] != "next_color"]
    color = [entry for entry in trace if entry[0] == "next_color"]
    return non_color + color


def _has_by_index_write_on_group(
    program: Program,
    fs: FilterState,
    group: frozenset[int],
) -> bool:
    """Return True if some step in ``program`` uses a ``by_index``
    selector (in its self or target role) to write to a shape in
    ``group``.

    Existence of such a step proves the program is longer than its
    collapsed equivalent: a single wider selector (ALL / is_hollow /
    …) would have written to the whole group in one step.
    """
    for k, step in enumerate(program):
        pre = list(fs.history[k])
        by_role = _step_writes_by_role(
            step, pre, self_override=_self_override_for(fs, k),
        )
        for shape_idx, roles in by_role.items():
            if shape_idx not in group:
                continue
            if "self" in roles and step.selector.type == "by_index":
                return True
            if "target" in roles and step.target is not None \
                    and step.target.type == "by_index":
                return True
    return False
