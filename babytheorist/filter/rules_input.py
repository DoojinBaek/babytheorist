"""Input-level rule engine.

Produces an :class:`InputConstraint` from a symbolically-validated
program, encoding what an input scene must look like for the program
to reach its declared depth on every shape.

Rules implemented:

- **I1**: stabilizer forbidden set. For each shape touched by D4
  primitives with net effect ``g_i`` at per-shape depth ``d_i``, the
  shape's stabilizer ``H`` must satisfy
  ``min_{h ∈ g_i·H} word_length(h) >= d_i``.
- **Vocab-wide**: when the level vocab mixes rot90 with any translate,
  every shape is required to have a non-square bounding box (square
  shapes would admit diagonal-reflection stabilizers the per-shape
  I1 table can't see).

Additional coincidence helpers:

- **I2a** (align / translate coincidence forbids): low-cost scene-time
  row/col coincidences that let align collapse to translate are forbidden
  in the default constraint chain.
- **I2b** (translate / align coincidence forbids): broader scene-time
  coincidences that let translate collapse to align are exposed for strict
  audits, but are not enabled in the default generation chain because they
  are materially more expensive and can over-constrain generation.

Later phases (not yet implemented):

- **I4** (selector arity) joins with dynamic selectors / relational.
- **I5** (primitive preconditions) comes online with fill_interior /
  align.
- **I6** (joint stabilizer tuple) is deferred — the per-shape I1 +
  trace-identity combination is sufficient for current vocabs and a
  joint-stabilizer input precondition can land alongside Phase 10' if
  the oracle surfaces residual violations.
"""

from __future__ import annotations

from babytheorist.core import Program

from babytheorist.symbolic import current_height, current_width, resolve_symbolic

from .constraint import InfeasibleConstraint, InputConstraint
from .d4_tables import (
    D4_IDENTITY,
    D4_PRIMITIVE_TO_ELEMENT,
    D4_PRIMITIVE_VOCAB,
    compose,
    stabilizers_allowed_at_depth,
)
from .rules_program import _affected_shapes
from .state import FilterState


__all__ = [
    "apply_rule_i1",
    "apply_vocab_requirements",
    "apply_align_coincidence_forbids",
    "apply_translate_align_coincidence_forbids",
    "apply_by_color_commitments",
    "apply_swap_next_color_forbids",
    "apply_positional_commitments",
    "apply_execution_safety_constraints",
    "accumulate_constraint",
]


_ROW_POSITIONAL = frozenset({"topmost", "bottommost"})
_COL_POSITIONAL = frozenset({"leftmost", "rightmost"})
_POSITIONAL_SELECTORS = _ROW_POSITIONAL | _COL_POSITIONAL


#: Primitives that swap the shape's h/w axes when composed (odd-parity
#: D4 element). flip_h and flip_v keep axes; rot90 is the only parity-
#: changer in the current vocab.
_PARITY_CHANGING_ROTATIONS = frozenset({"rot90"})

_TRANSLATE_PRIMS = frozenset({
    "translate_up", "translate_down", "translate_left", "translate_right",
})

_BBOX_SAFETY_PRIMITIVES = frozenset({"rot90", "add_border"})


def _per_shape_d4_summary(
    program: Program, fs: FilterState, d4_vocab: frozenset[str],
) -> dict[int, tuple[object, int]]:
    """For each shape touched by D4 primitives, return
    ``(net_effect, d4_depth)``. Untouched shapes are omitted."""
    summary: dict[int, list] = {}
    for step_idx, step in enumerate(program):
        if step.primitive not in d4_vocab:
            continue
        pre_state = fs.history[step_idx]
        self_override = (
            list(fs.self_resolutions[step_idx])
            if step_idx < len(fs.self_resolutions) else None
        )
        for i in _affected_shapes(step, pre_state, self_override=self_override):
            entry = summary.setdefault(i, [D4_IDENTITY, 0])
            g = D4_PRIMITIVE_TO_ELEMENT[step.primitive]
            entry[0] = compose(g, entry[0])
            entry[1] += 1
    return {i: (net, d) for i, (net, d) in summary.items()}


def apply_rule_i1(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
    d4_vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
) -> InputConstraint:
    """Restrict each touched shape's stabilizer set to those compatible
    with its ``(net_effect, d4_depth)`` pair. Untouched shapes
    receive no restriction."""
    summary = _per_shape_d4_summary(program, fs, d4_vocab)
    for shape_idx, (net, depth) in summary.items():
        allowed = stabilizers_allowed_at_depth(net, depth, d4_vocab)
        constraint = constraint.restrict_stabilizers(shape_idx, allowed)
    return constraint


def apply_vocab_requirements(
    vocab: frozenset[str], constraint: InputConstraint,
) -> InputConstraint:
    """Constraints that follow from the vocab itself, independently of
    the sampled program.

    - ``rot90 + translate``: every shape must be non-square. Square
      shapes admit diagonal-reflection stabilizers and the ``h_i = w_i``
      position coincidence — both blind spots for I1 / R5 / R7'.
    - ``add_border + translate``: every shape must have
      ``min(h, w) ≥ 3``. The pathological coincidence
      ``2h + 1 = h + 3`` (``h = 2``) makes
      ``[trans_up × 2, add_border]`` equivalent to
      ``[add_border, trans_up]`` at the pixel level — symbolic
      LinExpr can't catch this because it needs integer equation
      solving, not syntactic equality.
    """
    has_rot = bool(vocab & _PARITY_CHANGING_ROTATIONS)
    has_translate = bool(vocab & _TRANSLATE_PRIMS)
    has_add_border = "add_border" in vocab
    if has_rot and has_translate:
        for i in range(constraint.n_shapes):
            constraint = constraint.require_non_square(i)
    if has_add_border and has_translate:
        for i in range(constraint.n_shapes):
            constraint = constraint.require_min_bbox_dim(i, 3)
    return constraint


def apply_align_coincidence_forbids(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
    *,
    vocab: frozenset[str],
) -> InputConstraint:
    """Forbid concrete row/col coincidences that make align shortcutable.

    For ``align_row(src, tgt)``, the engine copies ``tgt.row`` into
    ``src.row``. If the concrete scene also satisfies
    ``src.row ± height(src) == tgt.row``, a shorter program can replace
    the align with a single translate on ``src``. The symbolic state can
    express both sides as :class:`LinExpr`; the scene sampler evaluates
    these forbids after shape sizes and placements are known.
    """
    for k, step in enumerate(program):
        if step.primitive not in {"align_row", "align_col"}:
            continue
        if step.target is None:
            continue
        pre = list(fs.history[k])
        src_indices = resolve_symbolic(step.selector, pre)
        tgt_indices = resolve_symbolic(step.target, pre)
        if len(src_indices) != 1 or len(tgt_indices) != 1:
            continue
        src = src_indices[0]
        tgt = tgt_indices[0]
        if step.primitive == "align_row":
            src_expr = pre[src].row
            tgt_expr = pre[tgt].row
            constraint = constraint.add_linear_forbid(src_expr, tgt_expr)
            h = current_height(pre[src])
            if "translate_up" in vocab:
                constraint = constraint.add_linear_forbid(src_expr - h, tgt_expr)
            if "translate_down" in vocab:
                constraint = constraint.add_linear_forbid(src_expr + h, tgt_expr)
            if "translate_up" in vocab and "translate_down" in vocab:
                constraint = constraint.add_linear_forbid(src_expr - h, tgt_expr + h)
                constraint = constraint.add_linear_forbid(src_expr + h, tgt_expr - h)
        else:
            src_expr = pre[src].col
            tgt_expr = pre[tgt].col
            constraint = constraint.add_linear_forbid(src_expr, tgt_expr)
            w = current_width(pre[src])
            if "translate_left" in vocab:
                constraint = constraint.add_linear_forbid(src_expr - w, tgt_expr)
            if "translate_right" in vocab:
                constraint = constraint.add_linear_forbid(src_expr + w, tgt_expr)
            if "translate_left" in vocab and "translate_right" in vocab:
                constraint = constraint.add_linear_forbid(src_expr - w, tgt_expr + w)
                constraint = constraint.add_linear_forbid(src_expr + w, tgt_expr - w)
    return constraint


def apply_translate_align_coincidence_forbids(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
    *,
    vocab: frozenset[str],
) -> InputConstraint:
    """Forbid concrete coincidences that make translate shortcutable by align.

    A step like ``translate_left(src)`` can be replaced by
    ``align_col(src, tgt)`` when ``src.col - width(src) == tgt.col`` on the
    concrete scene.  The symmetric row/col and direction cases are encoded as
    linear forbids so the scene sampler never realizes those coincidences.
    """
    has_align_row = "align_row" in vocab
    has_align_col = "align_col" in vocab
    if not has_align_row and not has_align_col:
        return constraint

    for k, step in enumerate(program):
        if step.primitive not in _TRANSLATE_PRIMS:
            continue
        pre = list(fs.history[k])
        if k < len(fs.self_resolutions):
            src_indices = list(fs.self_resolutions[k])
        else:
            src_indices = resolve_symbolic(step.selector, pre)
        if not src_indices:
            continue

        if step.primitive in {"translate_up", "translate_down"}:
            if not has_align_row:
                continue
            for src in src_indices:
                src_expr = pre[src].row
                h = current_height(pre[src])
                moved_expr = (
                    src_expr - h
                    if step.primitive == "translate_up"
                    else src_expr + h
                )
                for tgt in range(len(pre)):
                    if tgt == src:
                        continue
                    constraint = constraint.add_linear_forbid(
                        moved_expr, pre[tgt].row,
                    )
        else:
            if not has_align_col:
                continue
            for src in src_indices:
                src_expr = pre[src].col
                w = current_width(pre[src])
                moved_expr = (
                    src_expr - w
                    if step.primitive == "translate_left"
                    else src_expr + w
                )
                for tgt in range(len(pre)):
                    if tgt == src:
                        continue
                    constraint = constraint.add_linear_forbid(
                        moved_expr, pre[tgt].col,
                    )
    return constraint


def apply_by_color_commitments(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
    *,
    palette: tuple[int, ...] | None,
) -> InputConstraint:
    """Translate each ``by_color(c)`` step's committed subset into
    per-shape ``primary_color`` require / forbid entries.

    For step ``k`` with selector ``by_color(c)`` and committed
    indices ``S = fs.self_resolutions[k]``:

    - For each ``i ∈ S``: scene-time ``primary_color[i]`` must equal
      ``(c - offset_k_i) mod 9`` (where ``offset_k_i`` is the per-
      shape next_color offset at pre-state ``fs.history[k]``).
    - For each ``i ∉ S`` whose target palette value is ``(c -
      offset_k_i) mod 9`` and lies inside ``palette``: that value is
      forbidden. (If the target falls outside ``palette`` the
      exclusion is automatic.)

    Raises :class:`InfeasibleConstraint` if a commit's required
    palette value is outside ``palette`` or conflicts with an
    existing require / forbid on the same shape.
    """
    if palette is None:
        palette = ()
    palette_set = frozenset(palette)
    for k, step in enumerate(program):
        if step.selector.type != "by_color":
            continue
        c = int(step.selector.param)
        if k >= len(fs.self_resolutions):
            continue
        committed = frozenset(fs.self_resolutions[k])
        pre = fs.history[k]
        for i in range(fs.n_shapes):
            offset = pre[i].color.offset
            target = (c - offset) % 9
            if i in committed:
                if target not in palette_set:
                    raise InfeasibleConstraint(
                        f"step {k} by_color(c={c}) commits shape {i} but "
                        f"target primary {target} is outside palette "
                        f"{sorted(palette_set)}"
                    )
                constraint = constraint.require_primary(i, target)
            else:
                if target in palette_set:
                    constraint = constraint.forbid_primary(i, target)
    return constraint


def apply_swap_next_color_forbids(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
) -> InputConstraint:
    """Forbid color coincidences that collapse ``swap_colors`` + ``next_color``.

    For a swap between shapes ``a`` and ``b``, followed later by
    ``next_color`` on ``b`` only, the two-step color effect can equal the
    one-step program ``next_color(a)`` when
    ``primary[a] + 1 == primary[b]``. The symmetric case holds when
    ``next_color`` later touches ``a``.

    The reverse order has its own coincidence: ``next_color`` on exactly one
    participant before a later swap can collapse to ``next_color`` on the
    other participant when their initial primary colors are equal. Encode
    those scene-specific coincidences directly instead of relying on
    secondary color markers to break the shortcut accidentally.
    """
    for k, step in enumerate(program):
        if step.primitive != "swap_colors" or step.target is None:
            continue
        pre = list(fs.history[k])
        try:
            src_indices = resolve_symbolic(step.selector, pre)
            tgt_indices = resolve_symbolic(step.target, pre)
        except NotImplementedError:
            continue
        if len(src_indices) != 1 or len(tgt_indices) != 1:
            continue
        src, tgt = src_indices[0], tgt_indices[0]
        if src == tgt:
            continue

        for earlier_idx in range(0, k):
            earlier = program[earlier_idx]
            if earlier.primitive != "next_color":
                continue
            earlier_pre = list(fs.history[earlier_idx])
            self_override = (
                list(fs.self_resolutions[earlier_idx])
                if earlier_idx < len(fs.self_resolutions) else None
            )
            try:
                affected = set(_affected_shapes(
                    earlier, earlier_pre, self_override=self_override,
                ))
            except NotImplementedError:
                continue

            if (src in affected) != (tgt in affected):
                constraint = constraint.add_primary_forbid(src, tgt, 0)

        for later_idx in range(k + 1, len(program)):
            later = program[later_idx]
            if later.primitive != "next_color":
                continue
            later_pre = list(fs.history[later_idx])
            self_override = (
                list(fs.self_resolutions[later_idx])
                if later_idx < len(fs.self_resolutions) else None
            )
            try:
                affected = set(_affected_shapes(
                    later, later_pre, self_override=self_override,
                ))
            except NotImplementedError:
                continue

            if src in affected and tgt not in affected:
                constraint = constraint.add_primary_forbid(tgt, src, 1)
            if tgt in affected and src not in affected:
                constraint = constraint.add_primary_forbid(src, tgt, 1)
    return constraint


def apply_positional_commitments(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
) -> InputConstraint:
    """Translate each positional-selector step's committed winner into
    strict ``PositionalOrdering`` entries on ``constraint``.

    At step ``k`` with selector ``topmost`` / ``bottommost`` /
    ``leftmost`` / ``rightmost`` and committed winner
    ``w = fs.self_resolutions[k][0]``:

    - ``topmost``  : ``pos[w][row] < pos[j][row]`` ∀ j ≠ w.
    - ``bottommost``: ``pos[j][row] < pos[w][row]`` ∀ j ≠ w.
    - ``leftmost`` : ``pos[w][col] < pos[j][col]`` ∀ j ≠ w.
    - ``rightmost``: ``pos[j][col] < pos[w][col]`` ∀ j ≠ w.

    Raises :class:`InfeasibleConstraint` via
    :meth:`InputConstraint.add_positional_ordering` if the new edges
    would close a cycle.
    """
    for k, step in enumerate(program):
        sel_type = step.selector.type
        if sel_type not in _POSITIONAL_SELECTORS:
            continue
        if k >= len(fs.self_resolutions):
            continue
        resolution = fs.self_resolutions[k]
        if len(resolution) != 1:
            # Positional commits are single-shape by construction.
            continue
        winner = resolution[0]
        axis = "row" if sel_type in _ROW_POSITIONAL else "col"
        winner_is_lt = sel_type in ("topmost", "leftmost")
        for j in range(fs.n_shapes):
            if j == winner:
                continue
            if winner_is_lt:
                constraint = constraint.add_positional_ordering(
                    lt=winner, gt=j, axis=axis,
                )
            else:
                constraint = constraint.add_positional_ordering(
                    lt=j, gt=winner, axis=axis,
                )
    return constraint


def apply_execution_safety_constraints(
    program: Program,
    fs: FilterState,
    constraint: InputConstraint,
) -> InputConstraint:
    """Constrain every symbolic prefix scene to stay executable.

    This is a conservative input precondition for runtime safety.  The engine
    checks actual pixels after each step; here we only precondition
    bbox-changing primitives whose in-bounds requirements are cheap and local.
    Movement and pixel collisions still go through the engine check because
    encoding them as scene constraints makes placement unnecessarily slow on
    deep programs.
    """
    if not program:
        return constraint
    for step, states, indices in zip(
        program, fs.history[1:], fs.self_resolutions,
    ):
        if step.primitive not in _BBOX_SAFETY_PRIMITIVES:
            continue
        for i in indices:
            state = states[i]
            constraint = constraint.add_box_bounds(
                i,
                row=state.row,
                col=state.col,
                height=current_height(state),
                width=current_width(state),
            )
    return constraint


def accumulate_constraint(
    program: Program,
    fs: FilterState,
    *,
    vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    d4_vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    initial_grid_shapes: list[str] | None = None,
    palette: tuple[int, ...] | None = None,
) -> InputConstraint:
    """Build the full :class:`InputConstraint` for ``program``.

    Chains I1, vocab-wide requirements, each shape's required
    initial grid_shape (when supplied), and per-shape primary-color
    requirements from any ``by_color`` commitments in ``program``.
    """
    constraint = InputConstraint.empty(fs.n_shapes)
    if initial_grid_shapes is not None:
        if len(initial_grid_shapes) != fs.n_shapes:
            raise ValueError(
                f"initial_grid_shapes length {len(initial_grid_shapes)} "
                f"!= fs.n_shapes {fs.n_shapes}"
            )
        for i, gs in enumerate(initial_grid_shapes):
            constraint = constraint.require_grid_shape(i, gs)
    constraint = apply_rule_i1(program, fs, constraint, d4_vocab=d4_vocab)
    constraint = apply_vocab_requirements(vocab, constraint)
    constraint = apply_align_coincidence_forbids(
        program, fs, constraint, vocab=vocab,
    )
    constraint = apply_by_color_commitments(
        program, fs, constraint, palette=palette,
    )
    constraint = apply_swap_next_color_forbids(program, fs, constraint)
    constraint = apply_positional_commitments(program, fs, constraint)
    constraint = apply_execution_safety_constraints(program, fs, constraint)
    return constraint
