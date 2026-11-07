"""BabyTheorist support library — single-responsibility modules.

The user-facing data-generation entry point lives outside this package, in
`generate.py` at the project root, so the only file at the top level is the
one users actually call.

Module dependency graph (acyclic, layered):

    core ────────────────────────────────────┐
       ↑                                     │
       ├── primitives ── engine              │
       │        ↑           ↑                │
       ├── selectors ───────┘                │
       │                                     │
       ├── shapes                            │
       │                                     │
       ├── program                           │
       │                                     │
       └── review ──────────────────────────┘
                                             │
                                             ▼
                                       generate.py

Runtime dependencies: numpy, scipy.ndimage. Matplotlib is lazy-imported
only by `utils.viz`.
"""

from __future__ import annotations

from .version import (
    BENCHMARK_SPEC_VERSION,
    __version__,
    default_dataset_dir_name,
    default_dataset_release_version,
)
from .core import (
    # types
    Grid, Program, Shape, Step,
    Task, TaskPair,
    # selectors
    Selector, MULTI_SELECTORS, SINGLE_SELECTORS, SELECTOR_TYPES,
    ALL, by_color, is_hollow, topmost, bottommost, leftmost, rightmost,
    RELATIONAL_PRIMITIVE_NAMES,
)
from .primitives import (
    CATEGORIES, Category, Primitive, PRIMITIVES, primitive,
    category_of, primitives_by_category,
    canceling_pairs, involutions, idempotents, cyclic_orders,
    apply_primitive,
)
from .selectors import select, select_one
from .engine import apply_step, apply_program, apply_program_with_scenes
from .shapes import (
    DEFAULT_GENERATORS, PATTERN_TYPES, ShapeFactory,
    bordered, hollow, patterned, place_shapes, random_walk, solid,
)
from .program import effective_depth, is_valid, simplify

__all__ = [
    # data types
    "Grid", "Program", "Shape", "Step",
    "Task", "TaskPair",
    # selectors
    "Selector", "MULTI_SELECTORS", "SINGLE_SELECTORS", "SELECTOR_TYPES",
    "ALL", "by_color", "is_hollow",
    "topmost", "bottommost", "leftmost", "rightmost",
    "select", "select_one",
    # primitives
    "CATEGORIES", "Category", "Primitive", "PRIMITIVES", "primitive",
    "category_of", "primitives_by_category",
    "canceling_pairs", "involutions", "idempotents", "cyclic_orders",
    "apply_primitive",
    # engine
    "apply_step", "apply_program", "apply_program_with_scenes",
    # shapes
    "DEFAULT_GENERATORS", "PATTERN_TYPES", "ShapeFactory",
    "solid", "bordered", "hollow", "patterned",
    "place_shapes", "random_walk",
    # program validation
    "simplify", "effective_depth", "is_valid",
    # misc
    "RELATIONAL_PRIMITIVE_NAMES",
    # versioning
    "__version__", "BENCHMARK_SPEC_VERSION",
    "default_dataset_dir_name", "default_dataset_release_version",
]
