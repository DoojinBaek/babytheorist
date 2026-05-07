# BabyTheorist production filter and Dominant Abstract v2 Implementation Plan

## Summary

- The candidate-enumerating filter is now the production filter path.
- The temporary filter-system rollout switch has been removed from public
  generation and env auto-generation. `--backend filter` always uses the
  production filter.
- Improve dominant abstract programs only after production filter is stable, because the filter is the source of truth for accept/reject semantics.

## Key Changes

- Add `babytheorist/filter/` with a structured rule engine:
  - `types.py`: `FilterConfig`, `FilterContext`, `RuleResult`, `RejectReason`, `SamplerStats`
  - `state.py`: prefix state and selector commitments around existing symbolic state objects
  - `rules.py`: named rule implementations
  - `constraints.py`: incremental constraint builder compatible with existing `InputConstraint`
  - `sampler.py`: production filter `sample_program_with_constraint(...)`
- Replace history-driven rule names such as `R1`, `R3`, `I6`, and `R7'` with meaning-based names:
  - `PrimitiveCountBound`
  - `NoStateCycle`
  - `D4MinimalWord`
  - `NoDeadWrites`
  - `NoAxisCancellation`
  - `NoJointSelectorShortcut`
  - `NoD4CosetShortcut`
  - `NoTraceMergeShortcut`
  - `SelectorCommitmentConsistency`
  - `SceneConstraintFeasibility`
- Rewrite the sampler around valid-candidate enumeration instead of repeated random retries:
  - enumerate prefix-valid primitive/selector/target candidates before sampling;
  - remove symbolic precondition failures before candidate selection;
  - catch `fill_interior`, relational, positional, and `by_color` commitment conflicts early;
  - cache expensive shortcut-rule decisions by prefix signature.
- Add generation metadata:
  - `filter_system`
  - `filter_rules_version`
  - `filter_reject_stats`
  - `filter_generation_stats`
  - `dominant_system` when dominant sampling is enabled

## Implementation Plan

1. Freeze and measure the legacy baseline.
   - Measure `single_easy`, `multi_easy`, `single_hard`, `multi_hard`, and `multi_hard fixed_depth=5`.
   - Record tasks/sec, accepted programs/sec, program retries, scene retries, rule reject counts, and oracle validation time.

2. Build the production filter core.
   - Add the shared context/result types.
   - Reuse existing `InputConstraint`, `ShapeConstraint`, and `sample_scene` initially.
   - Make every rule return a structured pass/reject result with a stable rule name.

3. Implement the retry-reducing prefix sampler.
   - Construct candidate sets from the current prefix state.
   - Filter candidates with cheap precondition and constraint checks before weighted sampling.
   - Keep `max_step_retries` only as a fallback budget.
   - Emit `SamplerStats` for candidate counts, reject reasons, dead-end locations, and accepted depth.

4. Optimize incremental rule and constraint evaluation.
   - Stop rebuilding full constraints from scratch on every prefix where possible.
   - Maintain cached D4 summaries, write traces, per-axis targets, and selector commitments.
   - Gate and cache BFS-style shortcut checks.

5. Integrate with generation.
   - Route the filter generation path through the production filter.
   - Remove the temporary CLI flag `--filter-system`.
   - Preserve the public `Task`, HDF5, eval `Y_q`, suite routing, and metadata contracts.

6. Validate soundness and speed.
   - Any task accepted by production filter must pass brute-force shortcut oracle checks.
   - production filter should keep shortcut violations within the accepted
     audit budget.
   - Keep at least `2x` tasks/sec on `multi_hard fixed_depth=5` as the speed
     regression guard.
   - Continue optimizing beyond `2x` where candidate pruning and caching make it feasible.

7. Build Dominant Abstract v2.
   - Add `babytheorist/dominant_v2` or an equivalent versioned module after production filter is stable.
   - Keep selector-free abstract templates.
   - Add typed slots with slot constraints:
     - `single` or `group`
     - relational eligibility
     - required initial grid shape
     - selector capability
   - Replace `fill_interior` special handling with slot-level initial-state requirements.
   - Store the new cache format separately, for example `babytheorist.abstract_dominant_library.v4`.
   - Validate every dominant candidate through production filter.

8. Clean up after the switch.
   - Make the candidate-enumerating filter the only filter system after validation gates pass.
   - Remove or migrate legacy sampler/generator internals.
   - Update README and rulebook docs to use production filter terminology.
   - Remove the temporary legacy switch once it is no longer needed.

## Current Progress

Status as of 2026-04-30:

- Added the parallel `babytheorist/filter/` package with named rule wrappers,
  structured sampler stats, and a hybrid eager/lazy candidate sampler.
- Moved abstract-slot grounding into `babytheorist/filter/grounding.py`, with
  explicit `SlotRequirement` records, relational `by_index` forcing, initial
  grid-shape requirements, and `filter_grounding_version` metadata.
- Removed the temporary `--filter-system` rollout switch. `generate.py
  --backend filter` and env auto-generation now use the production filter
  directly.
- Added `filter_system`, `filter_rules_version`, `filter_generation_stats`,
  `filter_grounding_version`, `filter_reject_stats`, `dominant_system`, and
  `dominant_grounding_system` metadata hooks.
- Added bounded pair execution retries: the filter now retries scene realization
  for the same grounded program before discarding it, with
  `pair_execution_retries` recorded in generation metadata.
- Updated `scripts/bench_filter_system.py` for local production-filter
  throughput probes. Benchmark JSONL records include the exact run config for
  provenance.
- Added a saved benchmark report at
  `docs/filter_v2_benchmark_2026-04-30.jsonl`.
- The production filter is faster than the removed legacy filter on the
  targeted probe. On a warm single-worker `multi_hard`, level 4,
  `fixed_depth=5`, `n=20`, `warmups=1`, `repeats=3` run, it measured about
  `2.06x` the legacy throughput.
- Timing stats show the warm run is split mostly between abstract grounding
  sampler time and pair/scene realization time. Retrying scene realization
  reduces wasted dominant grounding on scene-specific execution failures.
- Pair-failure taxonomy is now emitted in `filter_generation_stats` using
  structured engine reasons. In the current `multi_hard` depth-5 probe, pair
  failures dropped from `58` to `2`; the remaining observed reasons are
  `engine_collision_or_oob` and `engine_noop`, which points the next speed pass
  toward movement/placement constraints and no-op/symmetry constraints.
- Full Dominant Abstract v2 cache/library format has not started yet. It should
  follow the remaining production-filter soundness/speed gates.

## Test Plan

- Add focused production filter tests:
  - `tests/babytheorist/filter/test_rules_meaning.py`
  - `tests/babytheorist/filter/test_candidate_sampler.py`
  - `tests/babytheorist/filter/test_grounding.py`
  - `tests/babytheorist/filter/test_canonical_rule_names.py`
  - `tests/babytheorist/filter/test_oracle_e2e_*.py`
  - `tests/babytheorist/test_filter_system_selection.py`
- Add dominant v2 tests:
  - `tests/babytheorist/test_dominant_v2.py`
- Run focused validation:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/babytheorist/filter
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/babytheorist/test_generate.py tests/babytheorist/test_dominant.py tests/babytheorist/test_parallel.py tests/env
```

- Run full validation before default switch:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

- Benchmark acceptance:
  - save production-filter throughput reports for the same seeds;
  - track shortcut violations in oracle e2e and DP sweep outputs;
  - keep at least `2x` tasks/sec on `multi_hard fixed_depth=5` as the speed
    regression guard.

## Assumptions

- The public benchmark contract remains unchanged: training data must not expose programs, depths, dominant templates, or same-program task grouping as fair supervision.
- `--backend filter` remains the production backend name.
- The public `--filter-system` switch has been removed.
- The production filter may reject more if that improves soundness, but default
  generation still requires acceptable yield and clear speed.
- Dominant Abstract v2 follows production filter. Dominant changes should not move first because they depend on the filter as the accept/reject source of truth.
