"""Production filter input-constraint builder."""

from __future__ import annotations

from babytheorist.core import Program
from babytheorist.filter.constraint import InputConstraint
from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.filter.rules_input import accumulate_constraint as _shared_accumulate

from .state import FilterState


def accumulate_constraint(
    program: Program,
    fs: FilterState,
    *,
    vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    d4_vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    initial_grid_shapes: list[str] | None = None,
    palette: tuple[int, ...] | None = None,
) -> InputConstraint:
    """Build the production input constraint for a prefix.

    The candidate-enumerating sampler reuses the shared constraint math here.
    The separate function gives us a stable replacement point for a future
    incremental builder.
    """

    return _shared_accumulate(
        program,
        fs,
        vocab=vocab,
        d4_vocab=d4_vocab,
        initial_grid_shapes=initial_grid_shapes,
        palette=palette,
    )


__all__ = ["accumulate_constraint"]
