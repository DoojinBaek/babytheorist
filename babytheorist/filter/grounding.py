"""Dominant abstract grounding helpers for the production filter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from babytheorist.core import ALL, RELATIONAL_PRIMITIVE_NAMES, Selector, Step
from babytheorist.dominant import AbstractProgram, AbstractRef, AbstractStep
from babytheorist.program import SamplerDeadEnd

from .state import FilterState, PreconditionFail


GROUNDING_SYSTEM_VERSION = "filter.abstract_grounding.v2"

Grounding = tuple[Selector, list[int] | None]


@dataclass(frozen=True)
class SlotRequirement:
    """Concrete selector constraints inferred for one abstract slot."""

    ref: AbstractRef
    force_by_index: bool = False
    required_grid_shape: str | None = None


def abstract_relational_refs(
    abstract_program: AbstractProgram | None,
) -> frozenset[AbstractRef]:
    if abstract_program is None:
        return frozenset()
    refs: set[AbstractRef] = set()
    for step in abstract_program:
        if step.primitive in RELATIONAL_PRIMITIVE_NAMES:
            refs.add(step.source)
            refs.add(step.target)
    return frozenset(refs)


def infer_slot_requirements(
    abstract_program: AbstractProgram,
    *,
    ref_initial_grid_shapes: dict[AbstractRef, str] | None = None,
) -> dict[AbstractRef, SlotRequirement]:
    """Infer static grounding requirements from an abstract program."""

    relational_refs = abstract_relational_refs(abstract_program)
    refs = {
        ref
        for step in abstract_program
        for ref in (
            (step.source,)
            if step.target is None
            else (step.source, step.target)
        )
    }
    shape_requirements = ref_initial_grid_shapes or {}
    return {
        ref: SlotRequirement(
            ref=ref,
            force_by_index=ref in relational_refs,
            required_grid_shape=shape_requirements.get(ref),
        )
        for ref in refs
    }


def abstract_step_has_random_grounding_choices(
    abstract_step: AbstractStep,
    bindings: dict[AbstractRef, Grounding],
) -> bool:
    """True iff retrying the step can sample a different slot binding."""

    if abstract_step.primitive in RELATIONAL_PRIMITIVE_NAMES:
        return (
            abstract_step.source not in bindings
            or abstract_step.target not in bindings
        )
    return abstract_step.source not in bindings


def _allowed_indices_for_grid_shape(
    requirement: SlotRequirement,
    initial_grid_shapes: list[str] | None,
) -> set[int] | None:
    required = requirement.required_grid_shape
    if required is None:
        return None
    if initial_grid_shapes is None:
        raise SamplerDeadEnd(
            f"abstract slot {requirement.ref!r} requires initial {required}"
        )
    allowed = {
        i for i, grid_shape in enumerate(initial_grid_shapes)
        if grid_shape == required
    }
    if not allowed:
        raise SamplerDeadEnd(
            f"no shapes have initial {required} for slot {requirement.ref!r}"
        )
    return allowed


def slot_grounding_options(
    requirement: SlotRequirement,
    *,
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    initial_grid_shapes: list[str] | None,
) -> list[tuple[Selector, list[int] | None, float]]:
    """Return weighted concrete selector choices for one abstract slot."""

    ref = requirement.ref
    allowed_indices = _allowed_indices_for_grid_shape(
        requirement, initial_grid_shapes,
    )
    single_options: list[tuple[Selector, list[int] | None, float]] = []
    group_options: list[tuple[Selector, list[int] | None, float]] = []

    if ref.kind in {"single", "any"}:
        if "by_index" in selector_types:
            for k in range(1, n_shapes + 1):
                if allowed_indices is not None and (k - 1) not in allowed_indices:
                    continue
                single_options.append((Selector("by_index", param=k), None, 1.0))
        if requirement.force_by_index:
            return single_options
        if "by_color" in selector_types:
            if palette is None:
                raise ValueError(
                    "by_color selector requires palette -- pass palette to "
                    "sample_program_with_constraint"
                )
            for color in sorted(palette):
                for i in range(n_shapes):
                    if allowed_indices is not None and i not in allowed_indices:
                        continue
                    single_options.append((
                        Selector("by_color", param=int(color)),
                        [i],
                        0.5,
                    ))
        for sel_type in ("topmost", "bottommost", "leftmost", "rightmost"):
            if sel_type not in selector_types:
                continue
            for i in range(n_shapes):
                if allowed_indices is not None and i not in allowed_indices:
                    continue
                single_options.append((Selector(sel_type), [i], 1.0))
        if ref.kind == "single":
            return single_options

    if ref.kind in {"group", "any"}:
        if "ALL" in selector_types and (
            allowed_indices is None or len(allowed_indices) == n_shapes
        ):
            group_options.append((ALL, None, 1.0))
        if (
            "is_hollow" in selector_types
            and requirement.required_grid_shape in (None, "hollow")
        ):
            group_options.append((Selector("is_hollow"), None, 0.75))
        if "by_color" in selector_types:
            if palette is None:
                raise ValueError(
                    "by_color selector requires palette -- pass palette to "
                    "sample_program_with_constraint"
                )
            for color in sorted(palette):
                for mask in range(1, 1 << n_shapes):
                    indices = [i for i in range(n_shapes) if mask & (1 << i)]
                    if (
                        allowed_indices is not None
                        and not set(indices) <= allowed_indices
                    ):
                        continue
                    group_options.append((
                        Selector("by_color", param=int(color)),
                        indices,
                        1.0 / len(indices),
                    ))
        if ref.kind == "group":
            return group_options

    def normalize_family(
        options: list[tuple[Selector, list[int] | None, float]],
        family_weight: float,
    ) -> list[tuple[Selector, list[int] | None, float]]:
        total = sum(weight for _selector, _override, weight in options)
        if total <= 0.0:
            return []
        return [
            (selector, override, family_weight * weight / total)
            for selector, override, weight in options
        ]

    if ref.kind == "any":
        if single_options and group_options:
            return (
                normalize_family(single_options, 0.5)
                + normalize_family(group_options, 0.5)
            )
        if single_options:
            return normalize_family(single_options, 1.0)
        if group_options:
            return normalize_family(group_options, 1.0)
        return []
    raise ValueError(
        f"unknown abstract slot kind {ref.kind!r} for slot {ref!r}"
    )


def sample_slot_grounding(
    requirement: SlotRequirement,
    *,
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    initial_grid_shapes: list[str] | None,
    rng: np.random.Generator,
) -> Grounding:
    options = slot_grounding_options(
        requirement,
        selector_types=selector_types,
        n_shapes=n_shapes,
        palette=palette,
        initial_grid_shapes=initial_grid_shapes,
    )
    if not options:
        raise SamplerDeadEnd(
            f"no concrete selector options for abstract slot "
            f"{requirement.ref!r}"
        )
    weights = np.asarray([weight for _, _, weight in options], dtype=np.float64)
    selector, override, _ = options[
        int(rng.choice(len(options), p=weights / weights.sum()))
    ]
    return selector, override


def bind_relational_ref(
    ref: AbstractRef,
    bindings: dict[AbstractRef, Grounding],
    *,
    n_shapes: int,
    rng: np.random.Generator,
    exclude_index: int | None = None,
) -> None:
    """Ensure a relational abstract ref is grounded to by_index."""

    if ref in bindings:
        selector, override = bindings[ref]
        if selector.type != "by_index" or override is not None:
            raise PreconditionFail(
                f"relational ref {ref!r} already bound to {selector!r}"
            )
        if exclude_index is not None and selector.param == exclude_index:
            raise PreconditionFail(
                f"relational refs resolved to same by_index {selector.param}"
            )
        return
    candidates = [k for k in range(1, n_shapes + 1) if k != exclude_index]
    if not candidates:
        raise PreconditionFail("no by_index candidate for relational ref")
    k = candidates[int(rng.integers(0, len(candidates)))]
    bindings[ref] = (Selector("by_index", param=k), None)


def ground_abstract_step(
    abstract_step: AbstractStep,
    bindings: dict[AbstractRef, Grounding],
    *,
    selector_types: frozenset[str],
    n_shapes: int,
    palette: tuple[int, ...] | None,
    fs: FilterState,
    rng: np.random.Generator,
    requirements: dict[AbstractRef, SlotRequirement],
    initial_grid_shapes: list[str] | None,
) -> tuple[Step, list[int] | None]:
    """Ground one abstract step using the current slot bindings."""

    _ = fs  # Reserved for future state-dependent slot constraints.
    prim = abstract_step.primitive
    if prim in RELATIONAL_PRIMITIVE_NAMES:
        if "by_index" not in selector_types or n_shapes < 2:
            raise PreconditionFail("relational abstract step requires by_index")
        bind_relational_ref(
            abstract_step.source,
            bindings,
            n_shapes=n_shapes,
            rng=rng,
        )
        src_selector, _src_override = bindings[abstract_step.source]
        bind_relational_ref(
            abstract_step.target,
            bindings,
            n_shapes=n_shapes,
            rng=rng,
            exclude_index=int(src_selector.param),
        )
        tgt_selector, _tgt_override = bindings[abstract_step.target]
        return Step(src_selector, prim, tgt_selector), None

    if abstract_step.source not in bindings:
        requirement = requirements[abstract_step.source]
        bindings[abstract_step.source] = sample_slot_grounding(
            requirement,
            selector_types=selector_types,
            n_shapes=n_shapes,
            palette=palette,
            initial_grid_shapes=initial_grid_shapes,
            rng=rng,
        )
    selector, override = bindings[abstract_step.source]
    return Step(selector=selector, primitive=prim), override


__all__ = [
    "GROUNDING_SYSTEM_VERSION",
    "Grounding",
    "SlotRequirement",
    "abstract_relational_refs",
    "abstract_step_has_random_grounding_choices",
    "bind_relational_ref",
    "ground_abstract_step",
    "infer_slot_requirements",
    "sample_slot_grounding",
    "slot_grounding_options",
]
