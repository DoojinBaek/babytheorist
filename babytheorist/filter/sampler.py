"""Candidate-enumerating production filter program sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from babytheorist.core import ALL, Program, RELATIONAL_PRIMITIVE_NAMES, Selector, Step
from babytheorist.dominant import AbstractProgram, AbstractRef
from babytheorist.filter.constraint import InfeasibleConstraint, InputConstraint
from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.program import SamplerDeadEnd
from babytheorist.symbolic import LinExpr

from .constraints import accumulate_constraint
from .grounding import (
    abstract_step_has_random_grounding_choices,
    ground_abstract_step,
    infer_slot_requirements,
)
from .rules import program_is_valid
from .state import FilterState, PreconditionFail
from .types import RejectReason, SamplerStats


__all__ = [
    "sample_program_with_constraint",
    "SamplerDeadEnd",
    "D4_TRANSLATE_VOCAB",
    "D4_TRANSLATE_COLOR_VOCAB",
    "NON_RELATIONAL_VOCAB",
    "FULL_VOCAB",
]


_SUPPORTED_SELECTORS = frozenset({
    "ALL", "by_index", "is_hollow", "by_color",
    "topmost", "bottommost", "leftmost", "rightmost",
})
_ROW_POSITIONAL = frozenset({"topmost", "bottommost"})
_COL_POSITIONAL = frozenset({"leftmost", "rightmost"})
_POSITIONAL_SELECTORS = _ROW_POSITIONAL | _COL_POSITIONAL
_EAGER_CANDIDATE_LIMIT = 8

#: D4 + translate vocabulary.
D4_TRANSLATE_VOCAB: frozenset[str] = D4_PRIMITIVE_VOCAB | frozenset({
    "translate_up", "translate_down", "translate_left", "translate_right",
})

#: D4 + translate + color vocabulary.
D4_TRANSLATE_COLOR_VOCAB: frozenset[str] = D4_TRANSLATE_VOCAB | frozenset({
    "next_color",
})

#: Every non-relational primitive.
NON_RELATIONAL_VOCAB: frozenset[str] = D4_TRANSLATE_COLOR_VOCAB | frozenset({
    "hollow", "fill_interior", "add_border",
})

#: Full production filter vocabulary.
FULL_VOCAB: frozenset[str] = NON_RELATIONAL_VOCAB | RELATIONAL_PRIMITIVE_NAMES


@dataclass(frozen=True)
class _Candidate:
    step: Step
    override_indices: list[int] | None
    weight: float = 1.0


def _check_selector_types(selector_types: frozenset[str]) -> None:
    unsupported = selector_types - _SUPPORTED_SELECTORS
    if unsupported:
        raise NotImplementedError(
            f"selectors {sorted(unsupported)} not yet supported by the filter"
        )


def _axis_undisturbed(fs: FilterState, axis: str) -> bool:
    attr = "row" if axis == "row" else "col"
    prefix = "r_" if axis == "row" else "c_"
    for i, s in enumerate(fs.current):
        if getattr(s, attr) != LinExpr.var(f"{prefix}{i}"):
            return False
    return True


def _sample_primitive(
    vocab: frozenset[str], rng: np.random.Generator,
) -> str:
    names = sorted(vocab)
    return names[int(rng.integers(0, len(names)))]


def _sample_relational_step(
    prim: str, n_shapes: int, rng: np.random.Generator,
) -> Step:
    src_k = int(rng.integers(1, n_shapes + 1))
    tgt_candidates = [k for k in range(1, n_shapes + 1) if k != src_k]
    tgt_k = tgt_candidates[int(rng.integers(0, len(tgt_candidates)))]
    return Step(
        selector=Selector("by_index", param=src_k),
        primitive=prim,
        target=Selector("by_index", param=tgt_k),
    )


def _sample_non_relational_step(
    primitive: str,
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
    rng: np.random.Generator,
) -> tuple[Step, list[int] | None]:
    options: list[tuple[str, object]] = []
    weights: list[float] = []
    if "ALL" in selector_types:
        options.append(("static", ALL))
        weights.append(1.0)
    if "by_index" in selector_types:
        for k in range(1, n_shapes + 1):
            options.append(("static", Selector("by_index", param=k)))
            weights.append(1.0)
    if "is_hollow" in selector_types:
        options.append(("static", Selector("is_hollow")))
        weights.append(1.0)
    if "by_color" in selector_types:
        if palette is None:
            raise ValueError(
                "by_color selector requires palette -- pass palette to "
                "sample_program_with_constraint"
            )
        for color in sorted(palette):
            for mask in range(1, 1 << n_shapes):
                indices = tuple(i for i in range(n_shapes) if mask & (1 << i))
                options.append(("by_color", (color, indices)))
                weights.append(1.0 / len(indices))
    if n_shapes >= 2:
        row_ok = _axis_undisturbed(fs, "row")
        col_ok = _axis_undisturbed(fs, "col")
        for sel_type in sorted(selector_types & _POSITIONAL_SELECTORS):
            axis = "row" if sel_type in _ROW_POSITIONAL else "col"
            if axis == "row" and not row_ok:
                continue
            if axis == "col" and not col_ok:
                continue
            for winner in range(n_shapes):
                options.append(("positional", (sel_type, winner)))
                weights.append(1.0)
    if not options:
        raise ValueError("selector_types is empty")

    probs = np.asarray(weights, dtype=np.float64)
    probs /= probs.sum()
    kind, payload = options[int(rng.choice(len(options), p=probs))]
    if kind == "static":
        return Step(selector=payload, primitive=primitive), None
    if kind == "by_color":
        color, indices = payload
        return (
            Step(selector=Selector("by_color", param=int(color)), primitive=primitive),
            list(indices),
        )
    sel_type, winner = payload
    return Step(selector=Selector(sel_type), primitive=primitive), [winner]


def _static_selector_candidates(
    selector_types: frozenset[str],
    n_shapes: int,
) -> list[tuple[Selector, list[int] | None, float]]:
    out: list[tuple[Selector, list[int] | None, float]] = []
    if "ALL" in selector_types:
        out.append((ALL, None, 1.0))
    if "by_index" in selector_types:
        for k in range(1, n_shapes + 1):
            out.append((Selector("by_index", param=k), None, 1.0))
    if "is_hollow" in selector_types:
        out.append((Selector("is_hollow"), None, 1.0))
    return out


def _by_color_candidates(
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
) -> list[tuple[Selector, list[int] | None, float]]:
    if "by_color" not in selector_types:
        return []
    if palette is None:
        raise ValueError(
            "by_color selector requires palette -- pass palette to "
            "sample_program_with_constraint"
        )
    out: list[tuple[Selector, list[int] | None, float]] = []
    for color in sorted(palette):
        for mask in range(1, 1 << n_shapes):
            indices = [i for i in range(n_shapes) if mask & (1 << i)]
            out.append((
                Selector("by_color", param=int(color)),
                indices,
                1.0 / len(indices),
            ))
    return out


def _positional_candidates(
    selector_types: frozenset[str],
    n_shapes: int,
    fs: FilterState,
) -> list[tuple[Selector, list[int] | None, float]]:
    if n_shapes < 2:
        return []
    row_ok = _axis_undisturbed(fs, "row")
    col_ok = _axis_undisturbed(fs, "col")
    out: list[tuple[Selector, list[int] | None, float]] = []
    for sel_type in sorted(selector_types & _POSITIONAL_SELECTORS):
        axis = "row" if sel_type in _ROW_POSITIONAL else "col"
        if axis == "row" and not row_ok:
            continue
        if axis == "col" and not col_ok:
            continue
        for winner in range(n_shapes):
            out.append((Selector(sel_type), [winner], 1.0))
    return out


def _non_relational_candidates(
    prim: str,
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
) -> list[_Candidate]:
    selector_options = []
    selector_options.extend(_static_selector_candidates(selector_types, n_shapes))
    selector_options.extend(_by_color_candidates(selector_types, n_shapes, palette))
    selector_options.extend(_positional_candidates(selector_types, n_shapes, fs))
    return [
        _Candidate(Step(selector=selector, primitive=prim), override, weight)
        for selector, override, weight in selector_options
    ]


def _relational_candidates(prim: str, n_shapes: int) -> list[_Candidate]:
    out: list[_Candidate] = []
    for src_k in range(1, n_shapes + 1):
        for tgt_k in range(1, n_shapes + 1):
            if src_k == tgt_k:
                continue
            out.append(_Candidate(
                Step(
                    selector=Selector("by_index", param=src_k),
                    primitive=prim,
                    target=Selector("by_index", param=tgt_k),
                ),
                None,
                1.0,
            ))
    return out


def _candidate_steps(
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for prim in sorted(vocab):
        if prim in RELATIONAL_PRIMITIVE_NAMES:
            if n_shapes >= 2 and "by_index" in selector_types:
                candidates.extend(_relational_candidates(prim, n_shapes))
        else:
            candidates.extend(_non_relational_candidates(
                prim, selector_types, n_shapes, palette, fs,
            ))
    return candidates


def _candidate_count_estimate(
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
) -> int:
    non_rel_count = sum(1 for p in vocab if p not in RELATIONAL_PRIMITIVE_NAMES)
    rel_count = sum(1 for p in vocab if p in RELATIONAL_PRIMITIVE_NAMES)
    selector_count = 0
    if "ALL" in selector_types:
        selector_count += 1
    if "by_index" in selector_types:
        selector_count += n_shapes
    if "is_hollow" in selector_types:
        selector_count += 1
    if "by_color" in selector_types:
        selector_count += (len(palette or ()) * ((1 << n_shapes) - 1))
    if n_shapes >= 2:
        row_ok = _axis_undisturbed(fs, "row")
        col_ok = _axis_undisturbed(fs, "col")
        for sel_type in selector_types & _POSITIONAL_SELECTORS:
            if sel_type in _ROW_POSITIONAL and row_ok:
                selector_count += n_shapes
            elif sel_type in _COL_POSITIONAL and col_ok:
                selector_count += n_shapes
    total = non_rel_count * selector_count
    if n_shapes >= 2 and "by_index" in selector_types:
        total += rel_count * n_shapes * (n_shapes - 1)
    return total


def _weighted_choice_index(
    items: list[_Candidate],
    rng: np.random.Generator,
) -> int:
    weights = np.asarray([max(0.0, item.weight) for item in items], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        return int(rng.integers(0, len(items)))
    return int(rng.choice(len(items), p=weights / total))


def _validate_applied_candidate(
    candidate: _Candidate,
    new_fs: FilterState,
    program: list[Step],
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    stats: SamplerStats | None,
) -> tuple[_Candidate, FilterState, InputConstraint] | None:
    new_program = program + [candidate.step]
    try:
        constraint = accumulate_constraint(
            new_program,
            new_fs,
            vocab=vocab,
            d4_vocab=vocab & D4_PRIMITIVE_VOCAB,
            initial_grid_shapes=initial_grid_shapes,
            palette=palette,
        )
    except InfeasibleConstraint:
        if stats is not None:
            stats.record_reject(RejectReason.CONSTRAINT_INFEASIBLE.value)
        return None
    if constraint.is_infeasible():
        if stats is not None:
            stats.record_reject(RejectReason.CONSTRAINT_INFEASIBLE.value)
        return None

    violation = program_is_valid(
        new_program,
        new_fs,
        selector_types=selector_types,
        constraint=constraint,
        palette=palette,
    )
    if violation is not None:
        if stats is not None:
            stats.record_reject(violation.rule)
        return None
    return candidate, new_fs, constraint


def _try_candidate(
    candidate: _Candidate,
    program: list[Step],
    fs: FilterState,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    stats: SamplerStats | None,
) -> tuple[_Candidate, FilterState, InputConstraint] | None:
    try:
        new_fs = fs.apply(
            candidate.step,
            override_indices=candidate.override_indices,
        )
    except PreconditionFail:
        if stats is not None:
            stats.record_reject(RejectReason.PRECONDITION.value)
        return None
    return _validate_applied_candidate(
        candidate,
        new_fs,
        program,
        vocab=vocab,
        selector_types=selector_types,
        initial_grid_shapes=initial_grid_shapes,
        palette=palette,
        stats=stats,
    )


def _sample_from_eager_candidates(
    candidates: list[_Candidate],
    program: list[Step],
    fs: FilterState,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    rng: np.random.Generator,
    max_full_checks: int,
    stats: SamplerStats | None,
) -> tuple[_Candidate, FilterState, InputConstraint] | None:
    cheap_valid: list[tuple[_Candidate, FilterState]] = []
    for candidate in candidates:
        try:
            new_fs = fs.apply(
                candidate.step,
                override_indices=candidate.override_indices,
            )
        except PreconditionFail:
            if stats is not None:
                stats.record_reject(RejectReason.PRECONDITION.value)
            continue
        cheap_valid.append((candidate, new_fs))

    if stats is not None:
        stats.record_candidates(len(candidates), len(cheap_valid))
    full_checks = min(max_full_checks, len(cheap_valid))
    for _ in range(full_checks):
        chosen_idx = _weighted_choice_index(
            [item[0] for item in cheap_valid], rng,
        )
        candidate, new_fs = cheap_valid.pop(chosen_idx)
        accepted = _validate_applied_candidate(
            candidate,
            new_fs,
            program,
            vocab=vocab,
            selector_types=selector_types,
            initial_grid_shapes=initial_grid_shapes,
            palette=palette,
            stats=stats,
        )
        if accepted is not None:
            return accepted
    return None


def _sample_random_candidate(
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
    rng: np.random.Generator,
) -> _Candidate | None:
    prim = _sample_primitive(vocab, rng)
    if prim in RELATIONAL_PRIMITIVE_NAMES:
        if n_shapes < 2 or "by_index" not in selector_types:
            return None
        return _Candidate(_sample_relational_step(prim, n_shapes, rng), None, 1.0)
    step, override = _sample_non_relational_step(
        prim,
        selector_types,
        n_shapes,
        palette,
        fs,
        rng,
    )
    return _Candidate(step, override, 1.0)


def _sample_from_lazy_candidates(
    program: list[Step],
    fs: FilterState,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    rng: np.random.Generator,
    max_step_retries: int,
    candidate_count_estimate: int,
    stats: SamplerStats | None,
) -> tuple[_Candidate, FilterState, InputConstraint] | None:
    if stats is not None:
        stats.record_candidates(candidate_count_estimate, 0)
    for _ in range(max_step_retries):
        candidate = _sample_random_candidate(
            vocab, selector_types, n_shapes, palette, fs, rng,
        )
        if candidate is None:
            continue
        accepted = _try_candidate(
            candidate,
            program,
            fs,
            vocab=vocab,
            selector_types=selector_types,
            initial_grid_shapes=initial_grid_shapes,
            palette=palette,
            stats=stats,
        )
        if accepted is not None:
            if stats is not None:
                stats.valid_candidate_counts[-1] += 1
            return accepted
    return None


def _sample_hybrid_candidate(
    program: list[Step],
    fs: FilterState,
    *,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    n_shapes: int,
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    rng: np.random.Generator,
    max_step_retries: int,
    stats: SamplerStats | None,
) -> tuple[_Candidate, FilterState, InputConstraint] | None:
    candidate_count = _candidate_count_estimate(
        vocab, selector_types, n_shapes, palette, fs,
    )
    if candidate_count <= _EAGER_CANDIDATE_LIMIT:
        candidates = _candidate_steps(vocab, selector_types, n_shapes, palette, fs)
        return _sample_from_eager_candidates(
            candidates,
            program,
            fs,
            vocab=vocab,
            selector_types=selector_types,
            initial_grid_shapes=initial_grid_shapes,
            palette=palette,
            rng=rng,
            max_full_checks=max_step_retries,
            stats=stats,
        )
    return _sample_from_lazy_candidates(
        program,
        fs,
        vocab=vocab,
        selector_types=selector_types,
        n_shapes=n_shapes,
        initial_grid_shapes=initial_grid_shapes,
        palette=palette,
        rng=rng,
        max_step_retries=max_step_retries,
        candidate_count_estimate=candidate_count,
        stats=stats,
    )


def sample_program_with_constraint(
    depth: int,
    n_shapes: int,
    *,
    vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    rng: Optional[np.random.Generator] = None,
    max_step_retries: int = 100,
    initial_grid_shapes: list[str] | None = None,
    palette: tuple[int, ...] | None = None,
    abstract_program: AbstractProgram | None = None,
    ref_initial_grid_shapes: dict[AbstractRef, str] | None = None,
    stats: SamplerStats | None = None,
) -> tuple[Program, InputConstraint]:
    """Sample a valid depth-``depth`` program and scene constraint.

    For non-dominant generation, the sampler enumerates all prefix candidates,
    filters them once, and then samples from the valid continuation set.  That
    removes repeated random draws of already-invalid continuations. Abstract
    dominant grounding still uses randomized slot binding, with each grounded
    prefix validated immediately.
    """

    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if n_shapes < 1:
        raise ValueError(f"n_shapes must be >= 1, got {n_shapes}")
    if initial_grid_shapes is not None and len(initial_grid_shapes) != n_shapes:
        raise ValueError(
            f"initial_grid_shapes length {len(initial_grid_shapes)} "
            f"!= n_shapes {n_shapes}"
        )
    _check_selector_types(selector_types)
    if "by_color" in selector_types and palette is None:
        raise ValueError(
            "by_color in selector_types but palette not provided"
        )
    if rng is None:
        rng = np.random.default_rng()
    if abstract_program is not None:
        if len(abstract_program) != depth:
            raise ValueError(
                f"abstract_program length {len(abstract_program)} "
                f"!= depth {depth}"
            )
        unknown = [
            step.primitive for step in abstract_program
            if step.primitive not in vocab
        ]
        if unknown:
            raise ValueError(
                f"abstract_program uses primitives outside vocab: {unknown!r}"
            )

    program: list[Step] = []
    fs = FilterState.initial(n_shapes, grid_shapes=initial_grid_shapes)
    slot_bindings: dict[AbstractRef, tuple[Selector, list[int] | None]] = {}
    requirements = (
        infer_slot_requirements(
            abstract_program,
            ref_initial_grid_shapes=ref_initial_grid_shapes,
        )
        if abstract_program is not None else {}
    )

    while len(program) < depth:
        if abstract_program is None:
            accepted = _sample_hybrid_candidate(
                program,
                fs,
                vocab=vocab,
                selector_types=selector_types,
                n_shapes=n_shapes,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
                rng=rng,
                max_step_retries=max_step_retries,
                stats=stats,
            )
            if accepted is None:
                if stats is not None:
                    stats.record_dead_end()
                    stats.record_reject(RejectReason.DEAD_END.value)
                raise SamplerDeadEnd(
                    f"filter found no valid continuation at step "
                    f"{len(program)} (depth target {depth})"
                )
            candidate, fs, _constraint = accepted
            program.append(candidate.step)
            continue

        committed = False
        attempts = 0
        abstract_step = abstract_program[len(program)]
        retry_budget = (
            max_step_retries
            if abstract_step_has_random_grounding_choices(
                abstract_step, slot_bindings,
            )
            else 1
        )
        for _ in range(retry_budget):
            attempts += 1
            trial_bindings = dict(slot_bindings)
            try:
                step, override = ground_abstract_step(
                    abstract_step,
                    trial_bindings,
                    selector_types=selector_types,
                    n_shapes=n_shapes,
                    palette=palette,
                    fs=fs,
                    rng=rng,
                    requirements=requirements,
                    initial_grid_shapes=initial_grid_shapes,
                )
            except PreconditionFail:
                if stats is not None:
                    stats.record_reject(RejectReason.PRECONDITION.value)
                continue
            accepted = _try_candidate(
                _Candidate(step, override, 1.0),
                program,
                fs,
                vocab=vocab,
                selector_types=selector_types,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
                stats=stats,
            )
            if accepted is None:
                continue
            candidate, fs, _constraint = accepted
            program.append(candidate.step)
            slot_bindings = trial_bindings
            committed = True
            break
        if stats is not None:
            stats.record_candidates(attempts, 1 if committed else 0)
        if not committed:
            if stats is not None:
                stats.record_dead_end()
                stats.record_reject(RejectReason.DEAD_END.value)
            raise SamplerDeadEnd(
                f"filter abstract grounding exhausted "
                f"{max_step_retries} retries at step {len(program)} "
                f"(depth target {depth})"
            )

    final_constraint = accumulate_constraint(
        program,
        fs,
        vocab=vocab,
        d4_vocab=vocab & D4_PRIMITIVE_VOCAB,
        initial_grid_shapes=initial_grid_shapes,
        palette=palette,
    )
    if stats is not None:
        stats.accepted_depth = len(program)
    return program, final_constraint
