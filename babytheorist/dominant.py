"""Abstract dominant-program pool.

Dominant programs are latent generator-side templates.  They keep primitive
co-occurrence structure, but bind primitives to abstract object slots instead
of concrete selectors.  Concrete selectors are sampled later by the filter
sampler and then validated by the normal filter stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from babytheorist.core import ALL, RELATIONAL_PRIMITIVE_NAMES, Program, Selector, Step
from babytheorist.filter.constraint import InfeasibleConstraint
from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.filter.rules_input import accumulate_constraint
from babytheorist.filter.rules_program import program_is_valid
from babytheorist.filter.state import FilterState, PreconditionFail
from babytheorist.program import SamplerDeadEnd


__all__ = [
    "AbstractRef",
    "AbstractStep",
    "AbstractProgram",
    "AbstractPool",
    "AbstractProvenance",
    "AbstractPoolBuild",
    "build_abstract_pool",
    "build_append_abstract_pool",
    "sample_abstract_program",
    "canonical_ground",
    "validate_canonical",
    "abstract_program_to_strings",
    "abstract_pool_to_jsonable",
    "abstract_pool_from_jsonable",
    "abstract_provenance_to_jsonable",
    "abstract_provenance_from_jsonable",
    "abstract_provenance_entry_ids",
]


RECURSIVE_COMPOSITION_POLICY = "recursive_frequency_v1"
RECURSIVE_COMPOSITION_SAMPLE_COUNT = 1000
DOMINANT_PROVENANCE_POLICY = "entry_id_parent_counts_v1"
_COMPOSITION_MAX_ATTEMPT_FACTOR = 100
_COMPOSITION_MIN_ATTEMPTS = 10_000
_COMPOSITION_MODES = ("reuse", "fresh")


@dataclass(frozen=True, order=True)
class AbstractRef:
    """A chunk-local latent object or object-group slot."""

    scope: int
    slot: str
    kind: str = "single"  # "single" | "group" | "any"

    def __post_init__(self) -> None:
        if self.kind not in {"single", "group", "any"}:
            raise ValueError(f"unknown abstract slot kind {self.kind!r}")


@dataclass(frozen=True)
class AbstractStep:
    """One abstract dominant step."""

    source: AbstractRef
    primitive: str
    target: AbstractRef | None = None

    def __post_init__(self) -> None:
        if self.primitive in RELATIONAL_PRIMITIVE_NAMES:
            if self.target is None:
                raise ValueError(
                    f"relational primitive {self.primitive!r} requires target"
                )
            if self.source.kind != "single" or self.target.kind != "single":
                raise ValueError(
                    f"relational primitive {self.primitive!r} requires single slots"
                )
        elif self.target is not None:
            raise ValueError(
                f"non-relational primitive {self.primitive!r} cannot have target"
            )


AbstractProgram = tuple[AbstractStep, ...]
AbstractPool = dict[int, dict[AbstractProgram, float]]
AbstractProvenance = dict[int, dict[AbstractProgram, dict[str, Any]]]
AbstractEntryFilter = Callable[
    [int, dict[AbstractProgram, float]], dict[AbstractProgram, float]
]


@dataclass
class AbstractPoolBuild:
    """Pool plus exact parent counts recorded during recursive construction."""

    pool: AbstractPool
    provenance: AbstractProvenance


def _make_non_rel_step(scope: int, slot: str, kind: str, primitive: str) -> AbstractStep:
    return AbstractStep(AbstractRef(scope, slot, kind), primitive)


def _make_rel_step(scope: int, primitive: str) -> AbstractStep:
    return AbstractStep(
        AbstractRef(scope, "A", "single"),
        primitive,
        AbstractRef(scope, "B", "single"),
    )


def _with_fresh_scopes(
    left: AbstractProgram,
    right: AbstractProgram,
) -> AbstractProgram:
    if not left:
        return right
    max_scope = max(
        ref.scope
        for step in left
        for ref in ((step.source,) if step.target is None else (step.source, step.target))
    )
    offset = max_scope + 1
    remap: dict[AbstractRef, AbstractRef] = {}

    def fresh(ref: AbstractRef) -> AbstractRef:
        if ref not in remap:
            remap[ref] = AbstractRef(ref.scope + offset, ref.slot, ref.kind)
        return remap[ref]

    shifted = tuple(
        AbstractStep(fresh(step.source), step.primitive, fresh(step.target) if step.target else None)
        for step in right
    )
    return left + shifted


def _canonicalize_scopes(program: AbstractProgram) -> AbstractProgram:
    """Renumber scopes by first occurrence while preserving slot identity."""
    if not program:
        return program
    remap: dict[int, int] = {}
    next_scope = 0

    def canonical(ref: AbstractRef) -> AbstractRef:
        nonlocal next_scope
        if ref.scope not in remap:
            remap[ref.scope] = next_scope
            next_scope += 1
        return AbstractRef(remap[ref.scope], ref.slot, ref.kind)

    return tuple(
        AbstractStep(
            canonical(step.source),
            step.primitive,
            canonical(step.target) if step.target else None,
        )
        for step in program
    )


def _compose_programs(
    left: AbstractProgram,
    right: AbstractProgram,
    mode: str,
) -> AbstractProgram:
    if mode == "reuse":
        return _canonicalize_scopes(left + right)
    if mode == "fresh":
        return _canonicalize_scopes(_with_fresh_scopes(left, right))
    raise ValueError(f"unknown composition mode {mode!r}")


def _single_ref_count(program: AbstractProgram) -> int:
    refs = {
        ref
        for step in program
        for ref in ((step.source,) if step.target is None else (step.source, step.target))
        if ref.kind in {"single", "any"}
    }
    return len(refs)


def _sample_composition(
    left: AbstractProgram,
    right: AbstractProgram,
    rng: np.random.Generator,
    *,
    n_shapes: int,
) -> tuple[AbstractProgram, str] | None:
    candidates = [
        (candidate, mode)
        for mode in _COMPOSITION_MODES
        for candidate in (_compose_programs(left, right, mode),)
        if _single_ref_count(candidate) <= n_shapes
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _program_sort_key(program: AbstractProgram) -> tuple:
    def ref_key(ref: AbstractRef | None) -> tuple:
        if ref is None:
            return (-1, "", "")
        return (ref.scope, ref.slot, ref.kind)

    return tuple(
        (ref_key(step.source), step.primitive, ref_key(step.target))
        for step in program
    )


def _entry_id(id_prefix: str, depth: int, index: int) -> str:
    return f"{id_prefix}d{depth}.{index:02d}"


def _depth_entry_ids(
    entries: dict[AbstractProgram, float],
    depth: int,
    id_prefix: str,
) -> dict[AbstractProgram, str]:
    return {
        program: _entry_id(id_prefix, depth, index)
        for index, program in enumerate(entries, start=1)
    }


def _pool_entry_ids(
    pool: AbstractPool,
    id_prefix: str,
) -> dict[int, dict[AbstractProgram, str]]:
    return {
        depth: _depth_entry_ids(entries, depth, id_prefix)
        for depth, entries in pool.items()
    }


def abstract_provenance_entry_ids(
    provenance: AbstractProvenance,
) -> dict[int, dict[AbstractProgram, str]]:
    """Return ``program -> stable entry id`` lookup maps from provenance."""
    return {
        depth: {
            program: str(record["id"])
            for program, record in entries.items()
        }
        for depth, entries in provenance.items()
    }


def _parent_key_to_record(
    key: tuple[str, int, str, str, str],
    count: int,
) -> dict[str, Any]:
    source, split, mode, left_id, right_id = key
    return {
        "source": source,
        "split": int(split),
        "mode": mode,
        "left_id": left_id,
        "right_id": right_id,
        "count": int(count),
    }


def _depth_provenance(
    entries: dict[AbstractProgram, float],
    *,
    depth: int,
    ids: dict[AbstractProgram, str],
    valid_samples: int | None,
    counts: dict[AbstractProgram, int] | None = None,
    parent_counts: dict[AbstractProgram, dict[tuple[str, int, str, str, str], int]] | None = None,
) -> dict[AbstractProgram, dict[str, Any]]:
    out: dict[AbstractProgram, dict[str, Any]] = {}
    for index, (program, score) in enumerate(entries.items(), start=1):
        parents = []
        if parent_counts is not None:
            parents = [
                _parent_key_to_record(key, count)
                for key, count in sorted(
                    parent_counts.get(program, {}).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ]
        record: dict[str, Any] = {
            "id": ids[program],
            "depth": depth,
            "index": index,
            "score": float(score),
            "parents": parents,
        }
        if valid_samples is not None:
            record["valid_samples"] = int(valid_samples)
        if counts is not None and program in counts:
            record["count"] = int(counts[program])
        out[program] = record
    return out


def _boundary_weight(
    left: AbstractProgram,
    right: AbstractProgram,
    bigram: np.ndarray | None,
    primitive_index: dict[str, int] | None,
) -> float:
    if bigram is None or primitive_index is None or not left or not right:
        return 1.0
    return float(bigram[primitive_index[left[-1].primitive], primitive_index[right[0].primitive]])


def _sample_entry(
    entries: dict[AbstractProgram, float],
    rng: np.random.Generator,
    *,
    left_context: AbstractProgram | None = None,
    bigram: np.ndarray | None = None,
    primitive_index: dict[str, int] | None = None,
) -> AbstractProgram | None:
    items = list(entries.items())
    if not items:
        return None
    weights = np.array([max(float(score), 0.0) for _, score in items], dtype=np.float64)
    if left_context is not None:
        weights *= np.array(
            [
                _boundary_weight(left_context, program, bigram, primitive_index)
                for program, _score in items
            ],
            dtype=np.float64,
        )
    total = weights.sum()
    if total <= 0:
        return None
    return items[int(rng.choice(len(items), p=weights / total))][0]


def _top_frequency_pool(
    counts: dict[AbstractProgram, int],
    valid_samples: int,
    pool_size: int,
) -> dict[AbstractProgram, float]:
    if not counts or valid_samples <= 0:
        return {}
    ranked = sorted(
        counts.items(),
        key=lambda kv: (-kv[1], _program_sort_key(kv[0])),
    )[:pool_size]
    return {
        program: float(count) / float(valid_samples)
        for program, count in ranked
    }


def _introduced_step_variants(primitive: str) -> tuple[AbstractProgram, ...]:
    """One-step abstract variants for an introduced primitive.

    The current curriculum only introduces non-relational primitives.  Keep
    introduced primitives kind-neutral; grounding chooses single/group later.
    """
    if primitive in RELATIONAL_PRIMITIVE_NAMES:
        return (_make_rel_step(0, primitive),)
    return ((_make_non_rel_step(0, "A", "any", primitive),),)


def _count_primitive(program: AbstractProgram, primitive: str) -> int:
    return sum(1 for step in program if step.primitive == primitive)


_ROW_TRANSLATES = frozenset({"translate_up", "translate_down"})
_TRANSLATE_SIGN = {
    "translate_up": -1,
    "translate_down": +1,
    "translate_left": -1,
    "translate_right": +1,
}


def _has_opposing_translate_pair(program: AbstractProgram) -> bool:
    """Reject dominant templates with backtracking movement on one slot.

    The symbolic filter can consider some ``up``/``down`` mixtures minimal
    when rotations change the active height/width phase. For curriculum
    dominant chunks, though, a single latent object moving in both directions
    on the same world axis is not a clean depth signal, so keep movement
    monotone per abstract slot and axis.
    """
    signs: dict[tuple[AbstractRef, str], set[int]] = {}
    for step in program:
        if step.primitive not in _TRANSLATE_SIGN:
            continue
        axis = "row" if step.primitive in _ROW_TRANSLATES else "col"
        key = (step.source, axis)
        bucket = signs.setdefault(key, set())
        bucket.add(_TRANSLATE_SIGN[step.primitive])
        if len(bucket) > 1:
            return True
    return False


def _fill_initial_grid_shapes(
    program: AbstractProgram,
    n_shapes: int,
    fallback: list[str] | None,
) -> list[str] | None:
    fill_refs = sorted({
        step.source
        for step in program
        if step.primitive == "fill_interior"
    })
    if not fill_refs:
        return fallback
    if len(fill_refs) > n_shapes:
        return []
    initial = list(fallback) if fallback is not None else ["solid"] * n_shapes
    for i, _ref in enumerate(fill_refs):
        initial[i] = "hollow"
    return initial


def build_append_abstract_pool(
    inherited_pool: AbstractPool,
    introduced_primitive: str,
    *,
    max_depth: int,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    pool_size: int = 5,
    rng: np.random.Generator | None = None,
    initial_grid_shapes: list[str] | None = None,
    palette: tuple[int, ...] | None = None,
    bigram: np.ndarray | None = None,
    primitives: list[str] | None = None,
    sample_count: int = RECURSIVE_COMPOSITION_SAMPLE_COUNT,
    inherited_entry_ids: dict[int, dict[AbstractProgram, str]] | None = None,
    track_provenance: bool = False,
    id_prefix: str = "",
    entry_filter: AbstractEntryFilter | None = None,
) -> AbstractPool | AbstractPoolBuild:
    """Build append-only entries containing exactly one new primitive.

    Depth-one entries are introduced atomic steps.  Higher depths are sampled by
    recursively composing one lower-depth append entry with one inherited entry,
    so every admitted candidate still contains the introduced primitive exactly
    once and is made from earlier dominant entries.
    """
    if introduced_primitive not in vocab:
        raise ValueError(
            f"introduced primitive {introduced_primitive!r} is outside vocab"
        )
    pool: AbstractPool = {}
    provenance: AbstractProvenance = {}
    entry_ids: dict[int, dict[AbstractProgram, str]] = {}
    inherited_ids = inherited_entry_ids
    if track_provenance and inherited_ids is None:
        inherited_ids = _pool_entry_ids(inherited_pool, "inherited:")
    sample_rng = rng if rng is not None else np.random.default_rng(0)
    primitive_index = (
        {p: i for i, p in enumerate(primitives)}
        if bigram is not None and primitives is not None else None
    )

    for depth in range(1, max_depth + 1):
        if depth == 1:
            pool[depth] = _canonical_depth_one_pool(
                [introduced_primitive],
                n_shapes=n_shapes,
                selector_types=selector_types,
                vocab=vocab,
                primitive_weights=None,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
                fill_initial_grid_shapes=True,
            )
            if entry_filter is not None:
                pool[depth] = entry_filter(depth, pool[depth])
            if track_provenance:
                entry_ids[depth] = _depth_entry_ids(
                    pool[depth], depth, id_prefix,
                )
                provenance[depth] = _depth_provenance(
                    pool[depth],
                    depth=depth,
                    ids=entry_ids[depth],
                    valid_samples=None,
                )
            continue

        sources: list[tuple[str, int]] = []
        for split in range(1, depth):
            if pool.get(split) and inherited_pool.get(depth - split):
                sources.append(("append_left", split))
            if inherited_pool.get(split) and pool.get(depth - split):
                sources.append(("append_right", split))
        counts: dict[AbstractProgram, int] = {}
        parent_counts: dict[
            AbstractProgram, dict[tuple[str, int, str, str, str], int]
        ] = {}
        validation_cache: dict[tuple[AbstractProgram, tuple[str, ...] | None], bool] = {}
        valid_samples = 0
        attempts = 0
        max_attempts = max(
            sample_count * _COMPOSITION_MAX_ATTEMPT_FACTOR,
            _COMPOSITION_MIN_ATTEMPTS,
        )
        while sources and valid_samples < sample_count and attempts < max_attempts:
            attempts += 1
            source, split = sources[int(sample_rng.integers(0, len(sources)))]
            if source == "append_left":
                left_entries = pool[split]
                right_entries = inherited_pool[depth - split]
            else:
                left_entries = inherited_pool[split]
                right_entries = pool[depth - split]
            left = _sample_entry(left_entries, sample_rng)
            if left is None:
                continue
            right = _sample_entry(
                right_entries,
                sample_rng,
                left_context=left,
                bigram=bigram,
                primitive_index=primitive_index,
            )
            if right is None:
                continue
            composition = _sample_composition(
                left, right, sample_rng, n_shapes=n_shapes,
            )
            if composition is None:
                continue
            candidate, mode = composition
            if _count_primitive(candidate, introduced_primitive) != 1:
                continue
            candidate_initial_grid_shapes = _fill_initial_grid_shapes(
                candidate, n_shapes, initial_grid_shapes,
            )
            if (
                candidate_initial_grid_shapes is not None
                and len(candidate_initial_grid_shapes) != n_shapes
            ):
                continue
            cache_key = (
                candidate,
                tuple(candidate_initial_grid_shapes)
                if candidate_initial_grid_shapes is not None else None,
            )
            accepted = validation_cache.get(cache_key)
            if accepted is None:
                accepted = validate_canonical(
                    candidate,
                    n_shapes=n_shapes,
                    vocab=vocab,
                    selector_types=selector_types,
                    initial_grid_shapes=candidate_initial_grid_shapes,
                    palette=palette,
                )
                validation_cache[cache_key] = accepted
            if not accepted:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            if track_provenance and inherited_ids is not None:
                if source == "append_left":
                    left_id = entry_ids[split][left]
                    right_id = inherited_ids[depth - split][right]
                else:
                    left_id = inherited_ids[split][left]
                    right_id = entry_ids[depth - split][right]
                parent_key = (source, split, mode, left_id, right_id)
                bucket = parent_counts.setdefault(candidate, {})
                bucket[parent_key] = bucket.get(parent_key, 0) + 1
            valid_samples += 1
        pool[depth] = _top_frequency_pool(counts, valid_samples, pool_size)
        if entry_filter is not None:
            pool[depth] = entry_filter(depth, pool[depth])
        if track_provenance:
            entry_ids[depth] = _depth_entry_ids(pool[depth], depth, id_prefix)
            provenance[depth] = _depth_provenance(
                pool[depth],
                depth=depth,
                ids=entry_ids[depth],
                valid_samples=valid_samples,
                counts=counts,
                parent_counts=parent_counts,
            )
    if track_provenance:
        return AbstractPoolBuild(pool=pool, provenance=provenance)
    return pool


def canonical_ground(
    program: AbstractProgram,
    n_shapes: int,
    *,
    any_kind: str = "single",
) -> Program:
    """Ground an abstract program to deterministic by_index selectors."""
    if any_kind not in {"single", "group"}:
        raise ValueError(f"unknown canonical any_kind {any_kind!r}")
    single_refs = sorted({
        ref
        for step in program
        for ref in ((step.source,) if step.target is None else (step.source, step.target))
        if ref.kind == "single" or (ref.kind == "any" and any_kind == "single")
    })
    if len(single_refs) > n_shapes:
        raise SamplerDeadEnd(
            f"canonical grounding needs {len(single_refs)} shapes, got {n_shapes}"
        )
    mapping = {
        ref: Selector("by_index", param=i + 1)
        for i, ref in enumerate(single_refs)
    }
    out: Program = []
    for step in program:
        if step.source.kind == "group" or (
            step.source.kind == "any" and any_kind == "group"
        ):
            selector = ALL
        else:
            selector = mapping[step.source]
        target = mapping[step.target] if step.target is not None else None
        out.append(Step(selector, step.primitive, target))
    return out


def validate_canonical(
    program: AbstractProgram,
    *,
    n_shapes: int,
    vocab: frozenset[str],
    selector_types: frozenset[str],
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
) -> bool:
    """Run canonical grounding through the same filter stack as generation."""
    if not program:
        return False
    if _has_opposing_translate_pair(program):
        return False
    if any(step.primitive not in vocab for step in program):
        return False
    refs = [
        ref
        for step in program
        for ref in ((step.source,) if step.target is None else (step.source, step.target))
    ]
    single_selector_types = {
        "by_index", "topmost", "bottommost", "leftmost", "rightmost", "by_color",
    }
    group_selector_types = {"ALL", "by_color", "is_hollow"}
    if any(ref.kind == "group" for ref in refs):
        if not selector_types & group_selector_types:
            return False
    if any(ref.kind == "single" for ref in refs):
        if not selector_types & single_selector_types:
            return False
    any_refs = any(ref.kind == "any" for ref in refs)
    any_kind = "single"
    if any_refs:
        if selector_types & single_selector_types:
            any_kind = "single"
        elif selector_types & group_selector_types:
            any_kind = "group"
        else:
            return False
    if any(step.primitive in RELATIONAL_PRIMITIVE_NAMES for step in program):
        if "by_index" not in selector_types:
            return False
    try:
        concrete = canonical_ground(program, n_shapes, any_kind=any_kind)
    except SamplerDeadEnd:
        return False
    fs = FilterState.initial(n_shapes, grid_shapes=initial_grid_shapes)
    prefix: Program = []
    for step in concrete:
        try:
            fs = fs.apply(step)
            prefix.append(step)
            constraint = accumulate_constraint(
                prefix,
                fs,
                vocab=vocab,
                d4_vocab=vocab & D4_PRIMITIVE_VOCAB,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
            )
        except (PreconditionFail, InfeasibleConstraint, ValueError):
            return False
        if constraint.is_infeasible():
            return False
        if program_is_valid(
            prefix,
            fs,
            selector_types=selector_types,
            constraint=constraint,
            palette=palette,
        ) is not None:
            return False
    return True


def _canonical_depth_one_pool(
    primitives: list[str],
    *,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    primitive_weights: dict[str, float] | None,
    initial_grid_shapes: list[str] | None,
    palette: tuple[int, ...] | None,
    fill_initial_grid_shapes: bool = False,
) -> dict[AbstractProgram, float]:
    """Canonical valid atomic entries for uniform depth-1 sampling."""
    weights = primitive_weights or {}
    entries: dict[AbstractProgram, float] = {}
    for primitive in primitives:
        if primitive in RELATIONAL_PRIMITIVE_NAMES:
            variants = ((_make_rel_step(0, primitive),),)
        else:
            variants = ((_make_non_rel_step(0, "A", "any", primitive),),)
        for abstract in variants:
            candidate_initial_grid_shapes = (
                _fill_initial_grid_shapes(abstract, n_shapes, initial_grid_shapes)
                if fill_initial_grid_shapes else initial_grid_shapes
            )
            if (
                candidate_initial_grid_shapes is not None
                and len(candidate_initial_grid_shapes) != n_shapes
            ):
                continue
            if not validate_canonical(
                abstract,
                n_shapes=n_shapes,
                vocab=vocab,
                selector_types=selector_types,
                initial_grid_shapes=candidate_initial_grid_shapes,
                palette=palette,
            ):
                continue
            entries[abstract] = float(weights.get(primitive, 1.0))
    return entries


def build_abstract_pool(
    bigram: np.ndarray,
    primitives: list[str],
    *,
    max_depth: int,
    n_shapes: int,
    selector_types: frozenset[str],
    vocab: frozenset[str],
    pool_size: int = 20,
    primitive_weights: dict[str, float] | None = None,
    rng: np.random.Generator | None = None,
    initial_grid_shapes: list[str] | None = None,
    palette: tuple[int, ...] | None = None,
    sample_count: int = RECURSIVE_COMPOSITION_SAMPLE_COUNT,
    track_provenance: bool = False,
    id_prefix: str = "",
    entry_filter: AbstractEntryFilter | None = None,
) -> AbstractPool | AbstractPoolBuild:
    """Build a filter-validated abstract dominant pool.

    Depth one contains valid atomic abstract steps.  Each higher-depth pool is
    estimated from repeated recursive compositions of lower-depth pool entries;
    the stored entries are the most frequent valid samples.
    """
    sample_rng = rng if rng is not None else np.random.default_rng(0)
    primitive_index = {p: i for i, p in enumerate(primitives)}
    pool: AbstractPool = {}
    provenance: AbstractProvenance = {}
    entry_ids: dict[int, dict[AbstractProgram, str]] = {}
    for depth in range(1, max_depth + 1):
        if depth == 1:
            pool[depth] = _canonical_depth_one_pool(
                primitives,
                n_shapes=n_shapes,
                selector_types=selector_types,
                vocab=vocab,
                primitive_weights=primitive_weights,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
            )
            if entry_filter is not None:
                pool[depth] = entry_filter(depth, pool[depth])
            if track_provenance:
                entry_ids[depth] = _depth_entry_ids(
                    pool[depth], depth, id_prefix,
                )
                provenance[depth] = _depth_provenance(
                    pool[depth],
                    depth=depth,
                    ids=entry_ids[depth],
                    valid_samples=None,
                )
            continue

        splits = [
            split for split in range(1, depth)
            if pool.get(split) and pool.get(depth - split)
        ]
        counts: dict[AbstractProgram, int] = {}
        parent_counts: dict[
            AbstractProgram, dict[tuple[str, int, str, str, str], int]
        ] = {}
        validation_cache: dict[AbstractProgram, bool] = {}
        valid_samples = 0
        attempts = 0
        max_attempts = max(
            sample_count * _COMPOSITION_MAX_ATTEMPT_FACTOR,
            _COMPOSITION_MIN_ATTEMPTS,
        )
        while splits and valid_samples < sample_count and attempts < max_attempts:
            attempts += 1
            split = splits[int(sample_rng.integers(0, len(splits)))]
            left = _sample_entry(pool[split], sample_rng)
            if left is None:
                continue
            right = _sample_entry(
                pool[depth - split],
                sample_rng,
                left_context=left,
                bigram=bigram,
                primitive_index=primitive_index,
            )
            if right is None:
                continue
            composition = _sample_composition(
                left, right, sample_rng, n_shapes=n_shapes,
            )
            if composition is None:
                continue
            candidate, mode = composition
            accepted = validation_cache.get(candidate)
            if accepted is None:
                accepted = validate_canonical(
                    candidate,
                    n_shapes=n_shapes,
                    vocab=vocab,
                    initial_grid_shapes=initial_grid_shapes,
                    selector_types=selector_types,
                    palette=palette,
                )
                validation_cache[candidate] = accepted
            if not accepted:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            if track_provenance:
                left_id = entry_ids[split][left]
                right_id = entry_ids[depth - split][right]
                parent_key = ("base", split, mode, left_id, right_id)
                bucket = parent_counts.setdefault(candidate, {})
                bucket[parent_key] = bucket.get(parent_key, 0) + 1
            valid_samples += 1

        pool[depth] = _top_frequency_pool(counts, valid_samples, pool_size)
        if entry_filter is not None:
            pool[depth] = entry_filter(depth, pool[depth])
        if track_provenance:
            entry_ids[depth] = _depth_entry_ids(pool[depth], depth, id_prefix)
            provenance[depth] = _depth_provenance(
                pool[depth],
                depth=depth,
                ids=entry_ids[depth],
                valid_samples=valid_samples,
                counts=counts,
                parent_counts=parent_counts,
            )
    if track_provenance:
        return AbstractPoolBuild(pool=pool, provenance=provenance)
    return pool


def sample_abstract_program(
    pool: AbstractPool,
    bigram: np.ndarray,
    primitives: list[str],
    depth: int,
    rng: np.random.Generator,
    p_direct: float = 0.5,
) -> AbstractProgram:
    """Sample a length-``depth`` abstract program from the pool."""
    nonempty = [d for d, entries in pool.items() if entries]
    if not nonempty:
        raise SamplerDeadEnd("abstract pool has no non-empty levels")
    max_level = max(nonempty)
    idx = {p: i for i, p in enumerate(primitives)}

    def first_primitive(program: AbstractProgram) -> str:
        return program[0].primitive

    def last_primitive(program: AbstractProgram) -> str:
        return program[-1].primitive

    def pick(d: int, prev: str | None) -> AbstractProgram:
        items = list(pool.get(d, {}).items())
        if not items:
            raise SamplerDeadEnd(f"abstract pool[{d}] is empty")
        weights = np.array([score for _, score in items], dtype=np.float64)
        if prev is not None:
            weights *= bigram[idx[prev], [idx[first_primitive(p)] for p, _ in items]]
        total = weights.sum()
        if total <= 0:
            raise SamplerDeadEnd(
                f"zero-weight abstract continuation at prev={prev!r}, pool[{d}]"
            )
        return items[int(rng.choice(len(items), p=weights / total))][0]

    if depth == 1 or (depth <= max_level and rng.random() < p_direct):
        return pick(depth, None)

    out: AbstractProgram = ()
    prev: str | None = None
    remaining = depth
    while remaining > 0:
        max_chunk = min(remaining, max_level)
        if not out and remaining == depth and depth <= max_level and depth > 1:
            max_chunk = depth - 1
        d = int(rng.integers(1, max_chunk + 1))
        chunk = pick(d, prev)
        out = _with_fresh_scopes(out, chunk)
        prev = last_primitive(chunk)
        remaining -= d
    return out


def abstract_program_to_strings(program: AbstractProgram) -> list[str]:
    """Stable human-readable representation for metadata."""
    rows: list[str] = []
    for step in program:
        src = f"s{step.source.scope}.{step.source.slot}:{step.source.kind}"
        if step.target is None:
            rows.append(f"{src}.{step.primitive}")
        else:
            tgt = f"s{step.target.scope}.{step.target.slot}:{step.target.kind}"
            rows.append(f"{src}.{step.primitive}({tgt})")
    return rows


def _ref_to_jsonable(ref: AbstractRef) -> dict:
    return {"scope": ref.scope, "slot": ref.slot, "kind": ref.kind}


def _ref_from_jsonable(raw: dict) -> AbstractRef:
    return AbstractRef(
        scope=int(raw["scope"]),
        slot=str(raw["slot"]),
        kind=str(raw["kind"]),
    )


def _step_to_jsonable(step: AbstractStep) -> dict:
    return {
        "source": _ref_to_jsonable(step.source),
        "primitive": step.primitive,
        "target": (
            _ref_to_jsonable(step.target)
            if step.target is not None else None
        ),
    }


def _step_from_jsonable(raw: dict) -> AbstractStep:
    target = raw.get("target")
    return AbstractStep(
        source=_ref_from_jsonable(raw["source"]),
        primitive=str(raw["primitive"]),
        target=_ref_from_jsonable(target) if target is not None else None,
    )


def abstract_pool_to_jsonable(pool: AbstractPool) -> dict:
    """Convert an abstract pool to a portable JSON-compatible structure."""
    return {
        str(depth): [
            {
                "score": float(score),
                "program": [_step_to_jsonable(step) for step in program],
            }
            for program, score in entries.items()
        ]
        for depth, entries in sorted(pool.items())
    }


def abstract_pool_from_jsonable(raw: dict) -> AbstractPool:
    """Restore an abstract pool serialized by ``abstract_pool_to_jsonable``."""
    pool: AbstractPool = {}
    for depth_s, entries in raw.items():
        depth = int(depth_s)
        pool[depth] = {
            tuple(_step_from_jsonable(step) for step in entry["program"]): float(
                entry["score"]
            )
            for entry in entries
        }
    return pool


def abstract_provenance_to_jsonable(provenance: AbstractProvenance) -> dict:
    """Convert sampled parent provenance to a portable JSON structure."""
    return {
        str(depth): [
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "parents"
                },
                "parents": [dict(parent) for parent in record.get("parents", [])],
                "program": [_step_to_jsonable(step) for step in program],
            }
            for program, record in entries.items()
        ]
        for depth, entries in sorted(provenance.items())
    }


def abstract_provenance_from_jsonable(raw: dict) -> AbstractProvenance:
    """Restore provenance serialized by ``abstract_provenance_to_jsonable``."""
    provenance: AbstractProvenance = {}
    for depth_s, entries in raw.items():
        depth = int(depth_s)
        bucket: dict[AbstractProgram, dict[str, Any]] = {}
        for entry in entries:
            program = tuple(_step_from_jsonable(step) for step in entry["program"])
            record = {
                key: value for key, value in entry.items() if key != "program"
            }
            record["parents"] = [
                dict(parent) for parent in record.get("parents", [])
            ]
            bucket[program] = record
        provenance[depth] = bucket
    return provenance
