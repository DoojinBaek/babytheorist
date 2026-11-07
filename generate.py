"""Public BabyTheorist dataset generation interface.

The implementation lives under ``babytheorist.generation``:

- ``config``: generation dataclasses and depth policy
- ``filter_dataset``: official constraint-forward generator
- ``rejection``: legacy rejection sampler used for tests/debugging
- ``cli``: command-line argument parsing and artifact writing

This file intentionally stays as a thin top-level facade:

    python generate.py --level 1 --n 50 --benchmark-mode single-A

    from generate import LevelConfig, TaskConfig, generate_filter_dataset
"""

from babytheorist.benchmark import (
    PrimitiveCurriculum,
    benchmark_mode_default_n_shapes,
    benchmark_mode_preset,
    resolve_primitive_curriculum,
)
from babytheorist.generation import (
    FilterGenerationStats,
    LevelConfig,
    TaskConfig,
    generate_dataset,
    generate_filter_dataset,
    generate_task,
    load_excluded_pairs,
    random_program_filtered,
)
from babytheorist.generation.cli import main
from babytheorist.generation.filter_dataset import (
    _FilterAttemptTimedOut,
    _FilterAttemptTimeoutError,
    _generate_filter_attempt,
    _generate_filter_attempt_with_timeout,
)

__all__ = [
    "FilterGenerationStats",
    "LevelConfig",
    "PrimitiveCurriculum",
    "TaskConfig",
    "_FilterAttemptTimedOut",
    "_FilterAttemptTimeoutError",
    "_generate_filter_attempt",
    "_generate_filter_attempt_with_timeout",
    "benchmark_mode_default_n_shapes",
    "benchmark_mode_preset",
    "generate_dataset",
    "generate_filter_dataset",
    "generate_task",
    "load_excluded_pairs",
    "main",
    "random_program_filtered",
    "resolve_primitive_curriculum",
]


if __name__ == "__main__":
    raise SystemExit(main())
