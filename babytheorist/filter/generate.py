"""End-to-end task generation through the production filter sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from babytheorist.core import Program, Shape
from babytheorist.engine import apply_program
from babytheorist.filter.constraint import InfeasibleConstraint, InputConstraint
from babytheorist.filter.d4_tables import D4_PRIMITIVE_VOCAB
from babytheorist.filter.scene_sampler import sample_scene
from babytheorist.program import SamplerDeadEnd
from babytheorist.shapes import ShapeFactory

from .sampler import sample_program_with_constraint


__all__ = [
    "GeneratedTask",
    "MultiPairTask",
    "generate_task",
    "generate_multipair_task",
]


@dataclass
class MultiPairTask:
    """A single filtered program paired with multiple scenes."""

    program: Program
    constraint: InputConstraint
    pairs: list[tuple[np.ndarray, list[Shape], np.ndarray]]


@dataclass
class GeneratedTask:
    program: Program
    constraint: InputConstraint
    input_shapes: list[Shape]
    input_grid: np.ndarray
    output_grid: np.ndarray


def _resolve_range(
    v: int | tuple[int, int], rng: np.random.Generator, label: str,
) -> int:
    if isinstance(v, int):
        if v < 1:
            raise ValueError(f"{label} must be >= 1, got {v}")
        return v
    lo, hi = v
    if not (isinstance(lo, int) and isinstance(hi, int)):
        raise TypeError(f"{label} range must be (int, int); got {v!r}")
    if lo < 1 or hi < lo:
        raise ValueError(
            f"{label} range must satisfy 1 <= lo <= hi, got ({lo}, {hi})"
        )
    return int(rng.integers(lo, hi + 1))


def _with_raster_order_constraint(
    constraint: InputConstraint,
) -> InputConstraint:
    for i in range(constraint.n_shapes - 1):
        constraint = constraint.add_raster_ordering(i, i + 1)
    return constraint


def _sample_initial_grid_shapes(
    n_shapes: int,
    initial_hollow_prob: float,
    rng: np.random.Generator,
) -> list[str]:
    return [
        "hollow" if rng.random() < initial_hollow_prob else "solid"
        for _ in range(n_shapes)
    ]


def _sample_scene_pair(
    program: Program,
    constraint: InputConstraint,
    *,
    factory: ShapeFactory,
    palette: tuple[int, ...],
    world_size: int,
    rng: np.random.Generator,
    gap: int,
    max_scene_retries: int,
) -> tuple[np.ndarray, list[Shape], np.ndarray] | None:
    scene = sample_scene(
        constraint,
        factory=factory,
        palette=palette,
        world_size=world_size,
        rng=rng,
        gap=gap,
        max_total_retries=max_scene_retries,
    )
    if scene is None:
        return None
    input_grid, input_shapes = scene
    output_grid, _ = apply_program(program, input_shapes, world_size)
    if output_grid is None:
        return None
    if np.array_equal(input_grid, output_grid):
        return None
    return input_grid, input_shapes, output_grid


def generate_task(
    *,
    depth: int | tuple[int, int],
    n_shapes: int | tuple[int, int],
    factory: ShapeFactory,
    palette: tuple[int, ...],
    world_size: int,
    rng: np.random.Generator,
    vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    max_program_retries: int = 50,
    max_scene_retries: int = 300,
    gap: int = 2,
    initial_hollow_prob: float = 0.0,
    raster_order: bool = False,
) -> Optional[GeneratedTask]:
    """Generate one task with the production filter, returning ``None`` on exhaustion."""

    if not 0.0 <= initial_hollow_prob <= 1.0:
        raise ValueError(
            f"initial_hollow_prob must be in [0, 1], got {initial_hollow_prob}"
        )
    for _ in range(max_program_retries):
        depth_k = _resolve_range(depth, rng, "depth")
        n_shapes_k = _resolve_range(n_shapes, rng, "n_shapes")
        initial_grid_shapes = _sample_initial_grid_shapes(
            n_shapes_k, initial_hollow_prob, rng,
        )
        try:
            program, constraint = sample_program_with_constraint(
                depth=depth_k,
                n_shapes=n_shapes_k,
                vocab=vocab,
                selector_types=selector_types,
                rng=rng,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
            )
            if raster_order:
                constraint = _with_raster_order_constraint(constraint)
        except (SamplerDeadEnd, InfeasibleConstraint):
            continue

        pair = _sample_scene_pair(
            program,
            constraint,
            factory=factory,
            palette=palette,
            world_size=world_size,
            rng=rng,
            gap=gap,
            max_scene_retries=max_scene_retries,
        )
        if pair is None:
            continue
        input_grid, input_shapes, output_grid = pair
        return GeneratedTask(
            program=program,
            constraint=constraint,
            input_shapes=input_shapes,
            input_grid=input_grid,
            output_grid=output_grid,
        )
    return None


def generate_multipair_task(
    *,
    n_pairs: int,
    depth: int | tuple[int, int],
    n_shapes: int | tuple[int, int],
    factory: ShapeFactory,
    palette: tuple[int, ...],
    world_size: int,
    rng: np.random.Generator,
    vocab: frozenset[str] = D4_PRIMITIVE_VOCAB,
    selector_types: frozenset[str] = frozenset({"ALL", "by_index"}),
    max_program_retries: int = 50,
    max_scene_retries: int = 300,
    gap: int = 2,
    initial_hollow_prob: float = 0.0,
    raster_order: bool = False,
) -> Optional[MultiPairTask]:
    """Generate one filtered program with ``n_pairs`` independent scenes."""

    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not 0.0 <= initial_hollow_prob <= 1.0:
        raise ValueError(
            f"initial_hollow_prob must be in [0, 1], got {initial_hollow_prob}"
        )

    for _ in range(max_program_retries):
        depth_k = _resolve_range(depth, rng, "depth")
        n_shapes_k = _resolve_range(n_shapes, rng, "n_shapes")
        initial_grid_shapes = _sample_initial_grid_shapes(
            n_shapes_k, initial_hollow_prob, rng,
        )
        try:
            program, constraint = sample_program_with_constraint(
                depth=depth_k,
                n_shapes=n_shapes_k,
                vocab=vocab,
                selector_types=selector_types,
                rng=rng,
                initial_grid_shapes=initial_grid_shapes,
                palette=palette,
            )
            if raster_order:
                constraint = _with_raster_order_constraint(constraint)
        except (SamplerDeadEnd, InfeasibleConstraint):
            continue

        pairs: list[tuple[np.ndarray, list[Shape], np.ndarray]] = []
        for _pair_idx in range(n_pairs):
            pair = _sample_scene_pair(
                program,
                constraint,
                factory=factory,
                palette=palette,
                world_size=world_size,
                rng=rng,
                gap=gap,
                max_scene_retries=max_scene_retries,
            )
            if pair is None:
                break
            pairs.append(pair)

        if len(pairs) == n_pairs:
            return MultiPairTask(
                program=program, constraint=constraint, pairs=pairs,
            )
    return None
