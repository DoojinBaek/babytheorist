"""Filter-side per-shape state, history tracking.

Wraps :func:`babytheorist.symbolic.apply_symbolic_step` with a snapshot
history so R3 (state revisit) can consult prior states.

Scope of this module
--------------------
- **D4-only milestone (Phase 6)**: current implementation is
  sufficient. Just stores states + per-step history.
- **Phase 7+ (translate)**: will grow a :class:`TranslateWindow`
  per-shape / per-axis for R5 dependency-aware cancellation.
  Intentionally deferred — adding the window machinery now, with
  no rule consuming it, buys nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from babytheorist.core import Step
from babytheorist.symbolic import (
    PreconditionFail,
    ShapeState,
    apply_symbolic_step,
    initial_states,
    resolve_symbolic,
)


__all__ = [
    "FilterState",
    "ShapeHistoryError",
    "PreconditionFail",  # re-export for convenience
]


class ShapeHistoryError(Exception):
    """Raised when history inconsistency is detected."""


@dataclass(frozen=True)
class FilterState:
    """Filter-side state: current per-shape states + full history.

    ``history[k]`` is the tuple of per-shape states BEFORE applying
    step ``k``. ``history[0]`` is therefore the initial state;
    ``current`` equals the state AFTER the most-recent step (i.e.
    what ``history[len(program)]`` would be if we appended it).

    ``self_resolutions[k]`` is the tuple of shape indices that step
    ``k`` resolves its ``selector`` to (the "self" write-role). For
    symbolically-resolvable selectors (``ALL``, ``by_index``,
    ``is_hollow``) this is just what :func:`resolve_symbolic` would
    return. For selectors that read scene-time data
    (``by_color``, positional), the sampler commits the resolution
    and records it here via :meth:`apply`'s ``override_indices``.

    History is kept as tuples of per-shape ``ShapeState`` (ordered
    by scene index) so equality comparisons are cheap and hashable.
    """

    current: tuple[ShapeState, ...]
    history: tuple[tuple[ShapeState, ...], ...]
    self_resolutions: tuple[tuple[int, ...], ...] = ()

    @classmethod
    def initial(
        cls, n_shapes: int, *, grid_shapes: list[str] | None = None,
    ) -> "FilterState":
        states = tuple(initial_states(n_shapes, grid_shapes=grid_shapes))
        return cls(current=states, history=(states,), self_resolutions=())

    def apply(
        self, step: Step, *, override_indices: list[int] | None = None,
    ) -> "FilterState":
        """Return the next state after applying ``step``.

        Raises :class:`PreconditionFail` if the step is symbolically
        infeasible (propagated from
        :func:`babytheorist.symbolic.apply_symbolic_step`).

        ``override_indices`` forwards to
        :func:`babytheorist.symbolic.apply_symbolic_step`. Used when the
        sampler commits a non-resolvable selector (``by_color`` /
        positional) to a specific shape subset. The chosen indices
        are recorded in :attr:`self_resolutions` so later analyses
        (trace-identity, dependency windows) can read them without
        re-resolving.
        """
        pre = list(self.current)
        new_states = tuple(apply_symbolic_step(
            step, pre, override_indices=override_indices,
        ))
        if override_indices is not None:
            resolution = tuple(override_indices)
        else:
            resolution = tuple(resolve_symbolic(step.selector, pre))
        return FilterState(
            current=new_states,
            history=self.history + (new_states,),
            self_resolutions=self.self_resolutions + (resolution,),
        )

    # --- queries for rules -----------------------------------------------

    def shape_state_at(self, step_idx: int, shape_idx: int) -> ShapeState:
        """Per-shape state before step ``step_idx`` (``0`` = initial)."""
        if not 0 <= step_idx < len(self.history):
            raise IndexError(f"step_idx {step_idx} out of history [0, {len(self.history)})")
        row = self.history[step_idx]
        if not 0 <= shape_idx < len(row):
            raise IndexError(f"shape_idx {shape_idx} out of [0, {len(row)})")
        return row[shape_idx]

    def shape_trajectory(self, shape_idx: int) -> tuple[ShapeState, ...]:
        """All snapshots of shape ``shape_idx``, initial-first."""
        return tuple(row[shape_idx] for row in self.history)

    @property
    def n_shapes(self) -> int:
        return len(self.current)

    @property
    def n_steps_applied(self) -> int:
        """Number of steps applied so far (history length - 1)."""
        return len(self.history) - 1
