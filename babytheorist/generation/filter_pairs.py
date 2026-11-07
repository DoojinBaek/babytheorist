"""Filter-backend pair sampling and conversion helpers."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from babytheorist.core import (
    Program, Shape, TaskPair,
    RELATIONAL_PRIMITIVE_NAMES,
)
from babytheorist.engine import apply_program_with_failure_reason
from babytheorist.primitives import category_of

from babytheorist.generation.config import (
    _FILTER_PAIR_EXECUTION_RETRIES,
    _PAIR_FAILURE_ENGINE_NONE,
    _PAIR_FAILURE_IDENTITY,
    _PAIR_FAILURE_RASTER_ORDER,
    _PAIR_FAILURE_SCENE_NONE,
)
from babytheorist.generation.rejection import _task_from_pairs_with_shapes


def _filter_task_from_mpt(mpt, level_cfg, task_cfg, uses_by_index):
    """Convert a filter ``MultiPairTask`` to the public ``Task`` contract."""
    if mpt is None:
        return None
    if uses_by_index and any(
        not _is_raster_ordered(shs) for _, shs, _ in mpt.pairs
    ):
        return None
    pairs_with_shapes = [
        (TaskPair(input_grid=ig, output_grid=og, n_shapes=len(shs)), shs)
        for ig, shs, og in mpt.pairs
    ]
    return _task_from_pairs_with_shapes(
        mpt.program, pairs_with_shapes, level_cfg, task_cfg, uses_by_index,
    )


def _resolve_int_or_range(v: int | tuple[int, int], rng: np.random.Generator) -> int:
    if isinstance(v, int):
        return v
    lo, hi = v
    return int(rng.integers(lo, hi + 1))


def _is_raster_ordered(shapes: list[Shape]) -> bool:
    return all(
        shapes[i].position <= shapes[i + 1].position
        for i in range(len(shapes) - 1)
    )


def _with_raster_order_constraint(constraint):
    for i in range(constraint.n_shapes - 1):
        constraint = constraint.add_raster_ordering(i, i + 1)
    return constraint


def _filter_sampler_module():
    from babytheorist.filter import sampler
    return sampler


def _filter_generate_module():
    from babytheorist.filter import generate as filter_generate
    return filter_generate


def _filter_scene_sampler_module():
    from babytheorist.filter import scene_sampler
    return scene_sampler


def _filter_rules_version() -> str:
    from babytheorist.filter.types import FILTER_RULES_VERSION
    return FILTER_RULES_VERSION


def _filter_grounding_version() -> str:
    from babytheorist.filter.grounding import GROUNDING_SYSTEM_VERSION
    return GROUNDING_SYSTEM_VERSION


def _primitive_signature_name(primitive: str) -> str:
    if primitive in RELATIONAL_PRIMITIVE_NAMES:
        return "relational"
    if primitive in {"translate_up", "translate_down", "translate_left", "translate_right"}:
        return "translate"
    if primitive in {"rot90", "flip_h", "flip_v"}:
        return "d4"
    if primitive in {"next_color", "swap_colors"}:
        return "color"
    if primitive in {"hollow", "fill_interior", "add_border"}:
        return "shape_axis"
    return str(category_of(primitive))


def _program_grounding_signature(program: Program) -> str:
    """Compact stable signature for analyzing grounded dominant programs."""

    source_selectors = Counter(step.selector.type for step in program)
    target_selectors = Counter(
        step.target.type for step in program if step.target is not None
    )
    primitive_families = Counter(
        _primitive_signature_name(step.primitive) for step in program
    )
    primitives = Counter(step.primitive for step in program)
    payload = {
        "depth": len(program),
        "source_selectors": dict(sorted(source_selectors.items())),
        "target_selectors": dict(sorted(target_selectors.items())),
        "primitive_families": dict(sorted(primitive_families.items())),
        "primitives": dict(sorted(primitives.items())),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _decode_grounding_signature(signature: str) -> dict[str, object]:
    try:
        decoded = json.loads(signature)
        if isinstance(decoded, dict):
            return decoded
    except json.JSONDecodeError:
        pass
    return {"raw": signature}


def _sample_filter_pair_with_reason(
    constraint, program, factory, level_cfg, task_cfg, rng,
    *,
    uses_by_index: bool,
    gap: int,
    execution_retries: int = _FILTER_PAIR_EXECUTION_RETRIES,
) -> tuple[tuple[TaskPair, list[Shape]] | None, str | None]:
    sample_scene = _filter_scene_sampler_module().sample_scene

    last_reason: str | None = None
    for _ in range(max(1, int(execution_retries))):
        scene = sample_scene(
            constraint,
            factory=factory,
            palette=level_cfg.palette,
            world_size=task_cfg.world_size,
            rng=rng,
            gap=gap,
            max_total_retries=task_cfg.max_pair_retries,
        )
        if scene is None:
            last_reason = _PAIR_FAILURE_SCENE_NONE
            break
        input_grid, input_shapes = scene
        if uses_by_index and not _is_raster_ordered(input_shapes):
            last_reason = _PAIR_FAILURE_RASTER_ORDER
            continue
        output_grid, _seq, failure_reason = apply_program_with_failure_reason(
            program, input_shapes, task_cfg.world_size,
        )
        if output_grid is None:
            last_reason = failure_reason or _PAIR_FAILURE_ENGINE_NONE
            continue
        if np.array_equal(input_grid, output_grid):
            last_reason = _PAIR_FAILURE_IDENTITY
            continue
        return (
            TaskPair(input_grid=input_grid, output_grid=output_grid,
                     n_shapes=len(input_shapes)),
            input_shapes,
        ), None
    return None, last_reason or _PAIR_FAILURE_ENGINE_NONE


def _sample_filter_pair(
    constraint, program, factory, level_cfg, task_cfg, rng,
    *,
    uses_by_index: bool,
    gap: int,
):
    pair, _reason = _sample_filter_pair_with_reason(
        constraint, program, factory, level_cfg, task_cfg, rng,
        uses_by_index=uses_by_index,
        gap=gap,
    )
    return pair


__all__ = [
    "_filter_task_from_mpt",
    "_resolve_int_or_range",
    "_is_raster_ordered",
    "_with_raster_order_constraint",
    "_filter_sampler_module",
    "_filter_generate_module",
    "_filter_scene_sampler_module",
    "_filter_rules_version",
    "_filter_grounding_version",
    "_primitive_signature_name",
    "_program_grounding_signature",
    "_decode_grounding_signature",
    "_sample_filter_pair_with_reason",
    "_sample_filter_pair",
]
