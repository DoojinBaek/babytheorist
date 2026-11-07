"""InputConstraint — accumulated scene constraints for a sampled program.

Built up step-by-step by the input rule engine (I1-I5). Passed to the
scene rejection sampler, which rejects any candidate scene that
violates the conjunction of these constraints.

Design principle: every field is monotonic under rule accumulation.
``add_*`` methods intersect / shrink / forbid — never loosen. Once
``is_infeasible()`` returns True, the constraint is permanently
infeasible and the caller should resample the program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from babytheorist.symbolic import D4, LinExpr

from .d4_tables import D4_SUBGROUPS


__all__ = [
    "ShapeConstraint",
    "InputConstraint",
    "InfeasibleConstraint",
    "LinearCoincidenceForbid",
    "BoxBounds",
    "RasterOrdering",
]


class InfeasibleConstraint(Exception):
    """Raised when constraint accumulation leaves no satisfying scene."""


# === Per-shape constraints ================================================


@dataclass
class ShapeConstraint:
    """Attribute constraints on a single shape (indexed by position in
    :class:`InputConstraint.shapes`).

    ``allowed_stabilizers`` is the set of D4 subgroups that the shape's
    ``Stab`` may equal. It starts as all 10 subgroups and shrinks via
    :meth:`restrict_stabilizers`. Empty → infeasible.

    ``require_non_square`` demands the shape's bounding box satisfies
    ``n_rows != n_cols``. Set by :func:`apply_vocab_requirements`
    whenever the level vocab mixes rot90 with any translate — square
    bounding boxes would admit stabilizers containing diagonal
    reflections, a blind spot for the per-shape I1 check.

    - ``required_primary_color``: shape's scene-time primary color must
      equal this palette value (``None`` = unconstrained). Set by
      ``by_color(c)`` commits that include this shape.
    - ``forbidden_primary_colors``: primary colors the shape must NOT
      have. Set by ``by_color(c)`` commits that exclude this shape.
      If ``required_primary_color`` lands in the forbidden set, the
      constraint is infeasible.
    - ``required_grid_shape``: ``"solid"`` / ``"hollow"`` / ``None``.
    """

    allowed_stabilizers: frozenset[frozenset[D4]] = field(
        default_factory=lambda: frozenset(D4_SUBGROUPS)
    )
    required_primary_color: int | None = None
    forbidden_primary_colors: frozenset[int] = frozenset()
    required_grid_shape: str | None = None  # "solid" | "hollow" | None
    require_non_square: bool = False
    min_bbox_dim: int = 1

    def restrict_stabilizers(
        self, allowed: Iterable[frozenset[D4]],
    ) -> "ShapeConstraint":
        new_allowed = self.allowed_stabilizers & frozenset(allowed)
        return ShapeConstraint(
            allowed_stabilizers=new_allowed,
            required_primary_color=self.required_primary_color,
            forbidden_primary_colors=self.forbidden_primary_colors,
            required_grid_shape=self.required_grid_shape,
            require_non_square=self.require_non_square,
            min_bbox_dim=self.min_bbox_dim,
        )

    def require_primary(self, color: int) -> "ShapeConstraint":
        """Pin ``primary_color`` to ``color``. Infeasible if it's
        already pinned to a different value or listed as forbidden."""
        if self.required_primary_color is not None \
                and self.required_primary_color != color:
            raise InfeasibleConstraint(
                f"primary_color already required as "
                f"{self.required_primary_color!r}, cannot also require "
                f"{color!r}"
            )
        if color in self.forbidden_primary_colors:
            raise InfeasibleConstraint(
                f"primary_color {color!r} requested but already forbidden"
            )
        return ShapeConstraint(
            allowed_stabilizers=self.allowed_stabilizers,
            required_primary_color=color,
            forbidden_primary_colors=self.forbidden_primary_colors,
            required_grid_shape=self.required_grid_shape,
            require_non_square=self.require_non_square,
            min_bbox_dim=self.min_bbox_dim,
        )

    def forbid_primary(self, color: int) -> "ShapeConstraint":
        if self.required_primary_color == color:
            raise InfeasibleConstraint(
                f"primary_color {color!r} forbidden but already required"
            )
        return ShapeConstraint(
            allowed_stabilizers=self.allowed_stabilizers,
            required_primary_color=self.required_primary_color,
            forbidden_primary_colors=self.forbidden_primary_colors | {color},
            required_grid_shape=self.required_grid_shape,
            require_non_square=self.require_non_square,
            min_bbox_dim=self.min_bbox_dim,
        )

    def require_grid_shape(self, gs: str) -> "ShapeConstraint":
        if self.required_grid_shape is not None and self.required_grid_shape != gs:
            raise InfeasibleConstraint(
                f"grid_shape already required as {self.required_grid_shape!r}, "
                f"cannot also require {gs!r}"
            )
        return ShapeConstraint(
            allowed_stabilizers=self.allowed_stabilizers,
            required_primary_color=self.required_primary_color,
            forbidden_primary_colors=self.forbidden_primary_colors,
            required_grid_shape=gs,
            require_non_square=self.require_non_square,
            min_bbox_dim=self.min_bbox_dim,
        )

    def with_require_non_square(self) -> "ShapeConstraint":
        return ShapeConstraint(
            allowed_stabilizers=self.allowed_stabilizers,
            required_primary_color=self.required_primary_color,
            forbidden_primary_colors=self.forbidden_primary_colors,
            required_grid_shape=self.required_grid_shape,
            require_non_square=True,
            min_bbox_dim=self.min_bbox_dim,
        )

    def with_min_bbox_dim(self, d: int) -> "ShapeConstraint":
        return ShapeConstraint(
            allowed_stabilizers=self.allowed_stabilizers,
            required_primary_color=self.required_primary_color,
            forbidden_primary_colors=self.forbidden_primary_colors,
            required_grid_shape=self.required_grid_shape,
            require_non_square=self.require_non_square,
            min_bbox_dim=max(self.min_bbox_dim, d),
        )

    def is_infeasible(self) -> bool:
        return len(self.allowed_stabilizers) == 0


# === Cross-shape constraints ==============================================


@dataclass(frozen=True)
class RowCoincidenceForbid:
    """Forbid ``shape_i.row + offset == shape_j.row`` at scene time."""
    shape_i: int
    shape_j: int
    offset: int


@dataclass(frozen=True)
class ColCoincidenceForbid:
    """Forbid ``shape_i.col + offset == shape_j.col`` at scene time."""
    shape_i: int
    shape_j: int
    offset: int


@dataclass(frozen=True)
class PrimaryColorCoincidenceForbid:
    """Forbid ``(shape_i.primary + offset) mod 9 == shape_j.primary``."""
    shape_i: int
    shape_j: int
    offset: int  # mod 9


@dataclass(frozen=True)
class LinearCoincidenceForbid:
    """Forbid equality between two symbolic linear expressions at scene time."""
    left: LinExpr
    right: LinExpr


@dataclass(frozen=True)
class BoxBounds:
    """Require one symbolic shape box to stay inside the world grid."""
    shape_idx: int
    row: LinExpr
    col: LinExpr
    height: LinExpr
    width: LinExpr


@dataclass(frozen=True)
class PositionalOrdering:
    """Strict ordering ``scene_pos[lt][axis] < scene_pos[gt][axis]`` at
    scene time. Always stored in ``lt < gt`` normal form.

    Emitted by positional-selector commits (``topmost`` / ``bottommost``
    / ``leftmost`` / ``rightmost``) on programs where the shapes'
    axis-coordinates are still symbolically undisturbed — i.e. no
    prior translate or add_border has touched the axis. The filter
    rejects positional commits on post-mutation states so the
    ordering is purely an initial-placement constraint.

    Tiebreaker is dropped deliberately: the engine's positional
    semantics tiebreak on the orthogonal axis, but strict inequality
    already eliminates the tie case. Scenes on the tie boundary are
    over-conservatively rejected (bounded yield loss, no violations).
    """
    lt: int             # shape whose axis-value is strictly smaller
    gt: int             # shape whose axis-value is strictly larger
    axis: str           # "row" | "col"


@dataclass(frozen=True)
class RasterOrdering:
    """Strict lexicographic ordering of top-left positions.

    ``before`` must appear earlier than ``after`` under input-scene raster
    scan: ``(row_before, col_before) < (row_after, col_after)``.
    """
    before: int
    after: int


# === Full input constraint ================================================


@dataclass
class InputConstraint:
    """Full conjunction of scene-level constraints produced by the
    input rule engine.

    ``n_shapes`` is fixed at construction (the sampler decides arity
    before any rules run). Per-shape and cross-shape fields are
    accumulated via ``restrict_*`` / ``add_*`` methods.

    All mutators return a NEW ``InputConstraint`` (immutable-style).
    Equality is structural.
    """

    n_shapes: int
    shapes: tuple[ShapeConstraint, ...]
    row_forbids: frozenset[RowCoincidenceForbid] = frozenset()
    col_forbids: frozenset[ColCoincidenceForbid] = frozenset()
    primary_forbids: frozenset[PrimaryColorCoincidenceForbid] = frozenset()
    linear_forbids: frozenset[LinearCoincidenceForbid] = frozenset()
    box_bounds: frozenset[BoxBounds] = frozenset()
    positional_orderings: frozenset[PositionalOrdering] = frozenset()
    raster_orderings: frozenset[RasterOrdering] = frozenset()

    @classmethod
    def empty(cls, n_shapes: int) -> "InputConstraint":
        return cls(
            n_shapes=n_shapes,
            shapes=tuple(ShapeConstraint() for _ in range(n_shapes)),
        )

    # --- per-shape mutators ------------------------------------------------

    def restrict_stabilizers(
        self, shape_idx: int, allowed: Iterable[frozenset[D4]],
    ) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].restrict_stabilizers(allowed)
        return self._replace_shapes(new_shapes)

    def forbid_primary(self, shape_idx: int, color: int) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].forbid_primary(color)
        return self._replace_shapes(new_shapes)

    def require_primary(self, shape_idx: int, color: int) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].require_primary(color)
        return self._replace_shapes(new_shapes)

    def require_grid_shape(self, shape_idx: int, gs: str) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].require_grid_shape(gs)
        return self._replace_shapes(new_shapes)

    def require_non_square(self, shape_idx: int) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].with_require_non_square()
        return self._replace_shapes(new_shapes)

    def require_min_bbox_dim(self, shape_idx: int, d: int) -> "InputConstraint":
        self._check_idx(shape_idx)
        new_shapes = list(self.shapes)
        new_shapes[shape_idx] = new_shapes[shape_idx].with_min_bbox_dim(d)
        return self._replace_shapes(new_shapes)

    # --- cross-shape mutators ---------------------------------------------

    def add_row_forbid(self, i: int, j: int, offset: int) -> "InputConstraint":
        self._check_idx(i); self._check_idx(j)
        if i == j:
            raise ValueError(f"row coincidence forbid needs distinct shapes, got {i}")
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids | {RowCoincidenceForbid(i, j, offset)},
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )

    def add_col_forbid(self, i: int, j: int, offset: int) -> "InputConstraint":
        self._check_idx(i); self._check_idx(j)
        if i == j:
            raise ValueError(f"col coincidence forbid needs distinct shapes, got {i}")
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids | {ColCoincidenceForbid(i, j, offset)},
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )

    def add_primary_forbid(self, i: int, j: int, offset: int) -> "InputConstraint":
        self._check_idx(i); self._check_idx(j)
        if i == j:
            raise ValueError(f"primary coincidence forbid needs distinct shapes, got {i}")
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids | {
                PrimaryColorCoincidenceForbid(i, j, offset % 9)
            },
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )

    def add_linear_forbid(
        self, left: LinExpr, right: LinExpr,
    ) -> "InputConstraint":
        """Forbid ``left == right`` under concrete scene variable values."""
        if left == right:
            raise InfeasibleConstraint(
                f"linear forbid is unsatisfiable because both sides are {left!r}"
            )
        forbid = LinearCoincidenceForbid(left, right)
        if forbid in self.linear_forbids:
            return self
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids | {forbid},
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )

    def add_box_bounds(
        self,
        shape_idx: int,
        *,
        row: LinExpr,
        col: LinExpr,
        height: LinExpr,
        width: LinExpr,
    ) -> "InputConstraint":
        """Require a symbolic bbox to remain in bounds at scene time."""
        self._check_idx(shape_idx)
        bound = BoxBounds(
            shape_idx=shape_idx, row=row, col=col, height=height, width=width,
        )
        if bound in self.box_bounds:
            return self
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds | {bound},
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )

    def add_positional_ordering(
        self, lt: int, gt: int, axis: str,
    ) -> "InputConstraint":
        """Add ``pos[lt][axis] < pos[gt][axis]`` to the constraint.

        Performs transitive-closure cycle detection: if adding this
        edge would create a directed cycle in the per-axis ordering
        graph (e.g. ``a < b``, ``b < c``, then adding ``c < a``),
        raises :class:`InfeasibleConstraint`.
        """
        self._check_idx(lt); self._check_idx(gt)
        if lt == gt:
            raise ValueError(
                f"positional ordering needs distinct shapes, got {lt}"
            )
        if axis not in ("row", "col"):
            raise ValueError(f"axis must be 'row' or 'col', got {axis!r}")
        new_ord = PositionalOrdering(lt=lt, gt=gt, axis=axis)
        if new_ord in self.positional_orderings:
            return self  # already present, no change
        # Cycle check: reachable(gt, lt) on the per-axis graph?
        adj: dict[int, set[int]] = {}
        for o in self.positional_orderings:
            if o.axis == axis:
                adj.setdefault(o.lt, set()).add(o.gt)
        # Simulate adding the new edge.
        adj.setdefault(lt, set()).add(gt)
        # Detect cycle from lt → gt → … → lt.
        stack = [gt]
        seen = {gt}
        while stack:
            node = stack.pop()
            if node == lt:
                raise InfeasibleConstraint(
                    f"positional ordering {new_ord!r} creates cycle"
                )
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings | {new_ord},
            raster_orderings=self.raster_orderings,
        )

    def add_raster_ordering(self, before: int, after: int) -> "InputConstraint":
        """Add ``pos[before] <lex pos[after]`` to the constraint."""
        self._check_idx(before); self._check_idx(after)
        if before == after:
            raise ValueError(
                f"raster ordering needs distinct shapes, got {before}"
            )
        new_ord = RasterOrdering(before=before, after=after)
        if new_ord in self.raster_orderings:
            return self
        adj: dict[int, set[int]] = {}
        for o in self.raster_orderings:
            adj.setdefault(o.before, set()).add(o.after)
        adj.setdefault(before, set()).add(after)
        stack = [after]
        seen = {after}
        while stack:
            node = stack.pop()
            if node == before:
                raise InfeasibleConstraint(
                    f"raster ordering {new_ord!r} creates cycle"
                )
            for nxt in adj.get(node, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=self.shapes,
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings | {new_ord},
        )

    # --- feasibility ------------------------------------------------------

    def is_infeasible(self) -> bool:
        """Trivial infeasibility — a per-shape set emptied out.

        Cross-shape forbids are always satisfiable on a large enough
        world (just pick different positions / colors), so they don't
        flag infeasibility here. The scene sampler will fail loudly
        if a particular world_size makes them jointly unsolvable.
        """
        return any(s.is_infeasible() for s in self.shapes)

    # --- internals --------------------------------------------------------

    def _check_idx(self, i: int) -> None:
        if not 0 <= i < self.n_shapes:
            raise IndexError(f"shape index {i} out of range [0, {self.n_shapes})")

    def _replace_shapes(self, shapes: list[ShapeConstraint]) -> "InputConstraint":
        return InputConstraint(
            n_shapes=self.n_shapes,
            shapes=tuple(shapes),
            row_forbids=self.row_forbids,
            col_forbids=self.col_forbids,
            primary_forbids=self.primary_forbids,
            linear_forbids=self.linear_forbids,
            box_bounds=self.box_bounds,
            positional_orderings=self.positional_orderings,
            raster_orderings=self.raster_orderings,
        )
