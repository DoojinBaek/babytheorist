# Abstract Dominant Programs

Status: v1 implemented on `dev/dominant_program`.

This note records the implemented v1 design for BabyTheorist generation: dominant
programs are filter-aware abstract programs over latent object slots, rather
than primitive-only sequences or concrete selector templates.

## Current Implementation

The v1 implementation adds:

- `babytheorist.dominant.AbstractRef`, `AbstractStep`, and `AbstractProgram`.
- A filter-validated abstract dominant pool builder.
- Direct and walking sampling from the abstract pool.
- Chunk-local scope freshening during walking composition.
- Canonical pseudo-grounding for pool-stage validation.
- Concrete grounding inside `sample_program_with_constraint`.
- Slot-level selector/resolution reuse for repeated `(scope, slot)` references.
- `by_color` committed-resolution reuse.
- Positional selector commitments for abstract single slots using the `0.5.5`
  grounding semantics.
- Relational slot prebinding to stable `by_index` selectors.
- Filter-backend integration in `generate_filter_dataset`.
- `--pool-seed` support for the default filter backend.
- Dominant viability preflight metadata in generated sidecars and cache files.
- HDF5 sidecar metadata for abstract dominant generation.

The public task schema, HDF5 arrays, `BabyTheoristEnv`, and `BabyTheoristGym` APIs are
unchanged. Abstract dominant programs are latent generator metadata, not
model-facing supervision.

## Goal

Dominant programs are meant to create recurring observable transformation
families that encourage abstraction. A primitive-only sequence is too weak:

```text
translate_up -> translate_right
```

can mean either one object moves up and right, or one object moves up while a
different object moves right. A concrete selector template is too specific:

```text
by_index(1).translate_up -> by_index(1).translate_right
```

risks turning the benchmark distribution into template matching. The intended
middle layer is an abstract program:

```text
A.translate_up -> A.translate_right -> B.translate_left -> B.translate_left
```

Here `A` and `B` are latent slots. During generation they are grounded to
concrete selectors and committed shape resolutions.

## Core Semantics

An abstract dominant program is a sequence of abstract steps:

```text
AbstractStep(source=A, primitive=translate_up)
AbstractStep(source=A, primitive=translate_right)
AbstractStep(source=B, primitive=translate_left)
```

Each slot has a kind:

- `single`: exactly one object.
- `group`: one or more objects.

Relational primitive source and target slots must be `single`.

Scopes are chunk-local namespaces. The same `(scope, slot)` refers to the same
latent object or group, while the same slot name in a different scope is
independent:

```text
scope0.A.translate_up
scope0.A.translate_right
scope1.A.translate_left
```

`scope0.A` and `scope1.A` are different latent bindings.

## Pool Construction

`pool[d]` should contain length-`d` abstract programs, not primitive tuples.
Pool entries may themselves be built compositionally from smaller chunks. This
preserves the existing dominant-program strategy:

- Direct path, probability `0.5`: sample one already-composed entry from
  `pool[d]`.
- Walking path, probability `0.5`: sample chunk lengths such as `3 / 4 / 3`,
  sample `pool[3]`, `pool[4]`, `pool[3]`, then concatenate them with fresh
  scopes.

Pool scoring retains the current primitive bigram and primitive weights.
Filtering is a hard constraint: a candidate abstract program is kept only if it
passes filter-aware validation. Official cache builds then apply a viability
preflight: the abstract entry must ground to concrete selectors and generate at
least one lightweight `(x, y)` pair under the current task shape/world config.

## Filter-Aware Validation

Dominant pool construction should use the filtering system as a prefilter.
For each candidate abstract program:

1. Convert abstract slots into a canonical pseudo-grounded concrete program.
2. Run the same symbolic filter stack used during generation:
   `FilterState.apply`, `accumulate_constraint`, and `program_is_valid`.
3. Keep the candidate only if every prefix remains valid and the requested
   depth is not shortcutable.

This pool-stage validation is not the final guarantee. Concrete selector
grounding can introduce new aliasing or shortcuts, so generated tasks must run
the filter again after grounding.

## Concrete Grounding

During task generation, an abstract program is grounded into concrete `Step`s.
Each `(scope, slot)` receives one selector plus a committed resolution, and that
grounding is reused for every occurrence of the slot.

Examples:

```text
scope0.A -> by_index(1)
scope0.B -> by_color(3), committed resolution = [0, 2]
```

The existing filter machinery already supports committed resolutions for a
subset of non-relational dynamic selectors:

- `by_color`: committed subset becomes primary-color requirements and forbids.
- `is_hollow`: symbolic grid-shape resolution and constraints.

Positional selectors are supported for abstract single slots. A slot may bind
to `topmost`, `bottommost`, `leftmost`, or `rightmost` with one committed
winner for filter constraints, and repeated occurrences reuse the same concrete
selector. Runtime execution still evaluates positional selectors as selectors
at each step, so repeated uses such as
`A.translate_down -> A.translate_down` preserve the concrete selector template
without materializing the committed winner as a stable `by_index` program.
This is the `0.5.5` generation contract; changing it would be a benchmark
semantics change.

The concrete grounded program must then pass the existing filter stack and scene
realization checks before the task is saved.

## Generation Integration

The official filter backend uses abstract dominant programs by default:

- `LevelConfig.use_dominant=True`: sample abstract programs from the dominant
  pool.
- `--no-dominant`: fall back to the current uniform filter sampler.
- `--pool-seed`: seed abstract pool construction. If omitted, it defaults to
  `--seed`.
- `primitive_weights` and `bigram_weights`: apply to abstract dominant scoring.

The public train/eval HDF5 schema and `BabyTheoristEnv` / `BabyTheoristGym` APIs should remain
unchanged. Abstract dominant programs are generator-side latent metadata, not
model-facing supervision.

Sidecar `.info.json` files should record generation metadata such as:

```text
dominant_kind = "abstract_slots"
dominant_pool_size
dominant_p_direct
dominant_depths
representative abstract programs
empirical slot-kind statistics
```

Current metadata fields include:

- `dominant_kind`
- `dominant_pool_seed`
- `dominant_pool_size`
- `dominant_p_direct`
- `dominant_depths`
- `dominant_programs`
- `dominant_slot_kinds`

## Dynamic Selector Roadmap

The implementation supports `by_color` dynamic commitments, symbolic
`is_hollow` commitments, and positional commitments using the `0.5.5`
runtime-selector semantics described above.

Relational dynamic selectors are a follow-up. The current symbolic relational
path does not carry source and target overrides, so cases such as:

```text
topmost.align_row(rightmost)
```

need additional filter support. The later extension should add relational
self/target override plumbing to `FilterState.apply`, symbolic IO reads/writes,
and align/swap constraint rules.

## Deferred Work

The following items are intentionally left for later iterations:

- Stable materialization of positional winners in abstract dominant grounding.
  The current support preserves concrete positional selector templates. If a
  future release wants repeated positional slots to keep addressing a committed
  object after movement, `add_border`, or relational alignment, it needs either
  linear ordering constraints over symbolic positions or stable selector
  materialization.
- Relational dynamic selectors.
  V1 relational slots are prebound to `by_index`. Dynamic relational selectors
  need source/target override plumbing through the symbolic and rule layers.
- Richer abstract binding grammars.
  V1 uses deterministic binding variants such as one-slot, A/B split, group,
  and fresh-per-step patterns. More structured or learned binding-pattern
  distributions can be added later.
- Empirical task-level dominant usage analytics.
  V1 records pool-level metadata. Detailed statistics over the actually
  generated tasks can be added if needed for dataset auditing.
- Large-depth throughput and quality sweeps.
  Level-3 performance is measured and usable; deeper levels should still be
  benchmarked separately for yield, depth distribution, and shortcut quality.

## Test Plan

Abstract dominant tests:

- `pool[d]` entries have length `d`.
- Pool entries preserve scope and slot bindings.
- Pool entries pass canonical pseudo-grounding validation.
- Direct and walking samples preserve total depth.
- Walking samples use independent chunk scopes.
- Fixed seeds produce deterministic pools and samples.

Grounding and filter tests:

- `[A.up, A.right]` reuses one selector and resolution.
- `[A.up, A.right, B.left, B.left]` separates A and B within a scope.
- Group slots can affect multiple shapes.
- Repeated `by_color` slots reuse the committed subset.
- Repeated positional slots reuse the concrete positional selector and preserve
  the `0.5.5` runtime re-resolution semantics.
- Concrete grounding failures retry cleanly without corrupting sampler state.

Generation tests:

- `generate_filter_dataset(..., use_dominant=True)` works for train and eval.
- `use_dominant=False` preserves the current uniform filter behavior.
- Small easy and hard level-3 generations succeed.
- Parallel generation remains deterministic.
- `.info.json` includes abstract dominant metadata.

Regression tests:

- Existing filter, generation, env, and HDF5 tests continue to pass.

## Validation Snapshot

Validation run on `dev/dominant_program`:

- `python -m py_compile -q generate.py babytheorist/dominant.py babytheorist/filter/sampler.py tests/babytheorist/test_dominant.py`
- `bash -n sample_data_gen.sh`
- `pytest -q tests/babytheorist/test_dominant.py`
- `pytest -q tests/env tests/babytheorist/test_generate.py tests/babytheorist/test_parallel.py tests/babytheorist/filter/test_sampler.py`
- `pytest -q tests/babytheorist`

Observed results:

- Dominant regression tests: `10 passed`.
- Core env/generate/parallel/filter subset: `145 passed`.
- Full `tests/babytheorist`: `885 passed`.

Generation smoke tests:

- Hard level-3 train generation succeeds with all primitives/selectors.
- Hard level-3 eval generation produces unique query outputs.
- `workers=1` and `workers=4` produce identical program sequences for the same
  seed and pool seed.
- CLI HDF5 export writes abstract dominant metadata to `.info.json`.

## Performance Snapshot

Measured on a 256-core host with hard mode, level 3, depth 3, `n_shapes=3`,
and abstract dominant generation enabled:

| Mode | Workers | Tasks | Throughput |
| --- | ---: | ---: | ---: |
| train | 8 | 1,000 | ~238 tasks/s |
| train | 32 | 1,000 | ~392 tasks/s |
| train | 64 | 1,000 | ~343 tasks/s |
| train | 32 | 5,000 | ~900 tasks/s |
| eval unique-support | 32 | 1,000 | ~173 tasks/s |

`workers=32` is the recommended setting for this host. `workers=128` finished
generation progress but stalled during multiprocessing pool teardown, so it is
not recommended.
