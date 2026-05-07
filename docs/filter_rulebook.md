# BabyTheorist Filter Rulebook

This document explains the current rule-based symbolic filtering system in this branch.

Goal:
- Guarantee intended program depth without brute-force search in the generation path.
- Use symbolic rules and input constraints to reject shortcut-able programs and scenes early.


## 1. High-level Responsibility Split

Generation path uses five layers:

1. Step sampler and prefix builder
- File: babytheorist/filter/sampler.py
- Entry: sample_program_with_constraint
- Responsibility:
	- Build a program prefix one step at a time.
	- Reject bad candidates immediately by symbolic execution, program rules, and input constraints.

2. Symbolic state transition and history
- File: babytheorist/filter/state.py
- Core type: FilterState
- Responsibility:
	- Hold current symbolic per-shape state.
	- Hold full state history for rule checks.
	- Record committed selector resolutions for non-symbolically-resolvable selectors.

3. Program-level redundancy rules (R rules)
- File: babytheorist/filter/rules_program.py
- Entry: program_is_valid
- Responsibility:
	- Reject prefixes that are non-minimal at the program level.

4. Input-level scene constraints (I rules and commitments)
- File: babytheorist/filter/rules_input.py
- Entry: accumulate_constraint
- Responsibility:
	- Compute constraints that input scenes must satisfy so the sampled program remains depth-valid.

5. Scene realization under constraints
- File: babytheorist/filter/scene_sampler.py
- Entry: sample_scene
- Responsibility:
	- Draw concrete shapes and placements that satisfy accumulated constraints.

Verification-only layer:
- Files: babytheorist/filter/oracle.py, babytheorist/filter/oracle_bfs.py, babytheorist/gpu_engine.py
- Responsibility:
	- Reliability checking and analysis only.
	- Not part of normal generation-time filtering.


## 2. Program-Level Rules (R Rules)

Program rules are evaluated in this order inside program_is_valid:

1) R1 count limits
- Function: check_count_limits
- File: babytheorist/filter/rules_program.py
- Role:
	- Enforce algebraic max-count per primitive per shape.
	- Examples:
		- rot90 max 3 (cyclic order 4)
		- flip_h/flip_v max 1 (involution)
		- next_color max 8 (cyclic order 9)
		- hollow/fill_interior max 1 (idempotent axis behavior)
- Why:
	- Removes obvious cycles and repeated no-op-like repetitions per shape.

2) R3 state revisit
- Function: check_state_revisit
- File: babytheorist/filter/rules_program.py
- Role:
	- Reject if any shape revisits a previous full symbolic state in its trajectory.
- Why:
	- Prevents identity cycles and many hidden shortcut patterns.
- Design note:
	- Current implementation is intentionally strict and may over-reject.
	- This is used as a conservative guard until stronger joint constraints are complete.

3) R6 D4 minimal-word check
- Function: check_minimal_word
- File: babytheorist/filter/rules_program.py
- Role:
	- For D4-active part of each shape, require:
		- number of applied D4 primitives == shortest D4 word length of net effect.
- Why:
	- Catches non-adjacent and alternative-path D4 shortcuts not removed by simple local rewrites.

4) R5 translate cancellation
- Function: check_translate_cancellation
- File: babytheorist/filter/rules_program.py
- Role:
	- On each shape and each axis, reject if translate writes exceed minimum needed for final linear offset.
	- Minimum is abs(coeff_h) + abs(coeff_w).
- Why:
	- Removes cancellations through transparent intermediate steps.

5) R5b grid-shape cancellation
- Function: check_grid_shape_cancellation
- File: babytheorist/filter/rules_program.py
- Role:
	- Reject when a shape gets both hollow and fill_interior in a way that creates a zero-net grid-shape round trip.
- Why:
	- Catches a shape-axis shortcut family that can survive full-state checks when other axes change.

6) R4 dead writes
- Function: check_dead_writes
- File: babytheorist/filter/rules_program.py
- Role:
	- Reject when a step writes only values that are overwritten before any relevant read.
- Why:
	- Eliminates overwrite-dead prefixes, including relational dead-code patterns.

7) R7 prime trace identity (selector-union aliasing)
- Function: check_trace_identity (delegates to babytheorist/filter/trace_identity.py)
- Files: babytheorist/filter/rules_program.py, babytheorist/filter/trace_identity.py
- Role:
	- Build per-shape write traces as ordered pairs (primitive, role).
	- If a group of shapes has identical trace and that group is expressible by a wider selector across relevant steps, reject by_index-split programs that are collapsible.
- Why:
	- Removes subset-collapse shortcuts, such as per-shape duplication that can be replaced by ALL, is_hollow, or deterministic by_color group selector.


## 3. Input-Level Rules and Commitments (I Side)

Input constraints are accumulated from the current prefix in accumulate_constraint.

### I1 stabilizer-depth compatibility
- Function: apply_rule_i1
- Files: babytheorist/filter/rules_input.py, babytheorist/filter/d4_tables.py
- Role:
	- For each D4-touched shape with net effect g and D4 depth d, restrict allowed stabilizers H so:
		- min over h in gH of word_length(h) >= d
- Why:
	- Forbid scene symmetries that would allow shorter equivalent rendering.

### Vocab-wide structural constraints
- Function: apply_vocab_requirements
- File: babytheorist/filter/rules_input.py
- Role:
	- Add constraints derived from primitive vocabulary interactions, independent of specific sampled steps.
	- Current enforced constraints:
		- If rot90 and translate coexist: require non-square bbox per shape.
		- If add_border and translate coexist: require min bbox dimension >= 3.
- Why:
	- Close known symbolic blind spots caused by shape geometry coincidences.

### by_color commitments
- Function: apply_by_color_commitments
- File: babytheorist/filter/rules_input.py
- Role:
	- Convert committed by_color(c) subset decisions into per-shape required/forbidden primary colors.
	- Adjust target color with current symbolic color offset.
- Why:
	- Make scene-time color-based selector behavior deterministic and compatible with sampled commits.

### positional commitments
- Function: apply_positional_commitments
- Files: babytheorist/filter/rules_input.py, babytheorist/filter/constraint.py
- Role:
	- Convert committed positional selector winners into strict row/col ordering constraints.
	- Detect and reject ordering cycles during accumulation.
- Why:
	- Ensure scene placement realizes chosen topmost/bottommost/leftmost/rightmost commitments.

### initial grid-shape requirement
- Integration point: accumulate_constraint
- File: babytheorist/filter/rules_input.py
- Role:
	- If initial_grid_shapes is provided, pin each shape to required initial grid_shape.
- Why:
	- Keep symbolic initial assumptions and scene sampler behavior aligned.


## 4. Constraint Data Model and Feasibility

Core model:
- File: babytheorist/filter/constraint.py
- Types:
	- ShapeConstraint
	- InputConstraint
	- RowCoincidenceForbid, ColCoincidenceForbid, PrimaryColorCoincidenceForbid, PositionalOrdering

Monotonic accumulation principle:
- Every update narrows or intersects constraints.
- No update relaxes previously added information.

Immediate infeasibility:
- Triggered when per-shape allowed stabilizer set becomes empty.

Deferred infeasibility:
- Cross-shape constraints are checked during concrete scene sampling and placement.


## 5. Scene Sampler Responsibilities

File: babytheorist/filter/scene_sampler.py

What sample_scene enforces:

1. Per-shape checks
- Stabilizer membership
- Required or forbidden primary color
- Required grid_shape
- Non-square requirement
- Minimum bbox dimension

2. Placement checks
- Uses place_shapes for spacing and fit in world grid.

3. Cross-shape checks
- Row/col coincidence forbids
- Primary color coincidence forbids
- Positional ordering constraints


## 6. Selector Handling Model

Symbolically resolvable selectors:
- ALL, by_index, is_hollow

Committed selectors (resolved by sampler, stored in state self_resolutions):
- by_color
- topmost, bottommost, leftmost, rightmost

Implementation path:
- sampler chooses step and optional override indices
- FilterState.apply forwards override_indices to symbolic step application
- rules consume history plus self_resolutions to reason consistently


## 7. D4 Table Layer (Precomputed Group Logic)

File: babytheorist/filter/d4_tables.py

What it provides:
- D4 elements and composition/inverse tables
- Word lengths under primitive generators
- Stabilizer subgroup list
- Coset minimum-word lookup helpers

Why:
- Keep rule checks constant-time table lookups instead of repeated online BFS/group recomputation.


## 8. End-to-End Generation Flow (Actual Call Path)

1. generate_filter_dataset in generate.py chooses filter backend.
2. generate_multipair_task or generate_task in babytheorist/filter/generate.py.
3. sample_program_with_constraint in babytheorist/filter/sampler.py.
4. Inside each candidate step:
	 - symbolic apply via FilterState
	 - program_is_valid (R rules)
	 - accumulate_constraint (I side)
5. After full depth, sample_scene under final InputConstraint.
6. Run apply_program to produce output and reject identity outputs.


## 9. What Is Explicitly Deferred or Conservative

Documented deferred items include:
- Joint stabilizer tuple style constraints (I6-style extensions)
- Additional input-level coincidence rules tied to future expansions
- Possible completeness improvements for dynamic selector expressibility

Current stance:
- Prioritize soundness, allow bounded over-rejection.
- Validate reliability with external oracle tests rather than using oracle in generation loop.


## 10. Verification Boundary (Important)

Generation-time filtering:
- Pure symbolic rule and constraint pipeline.
- No brute-force Q search in normal generation.

Verification-time reliability checks:
- oracle.py and oracle_bfs.py in tests and analysis scripts.
- Optional GPU acceleration still CPU-reverified for soundness.


## 11. Main Files to Read by Responsibility

Program sampling and orchestration:
- babytheorist/filter/sampler.py
- babytheorist/filter/generate.py

Program-level rules:
- babytheorist/filter/rules_program.py
- babytheorist/filter/trace_identity.py

Input-level rules and constraints:
- babytheorist/filter/rules_input.py
- babytheorist/filter/constraint.py

Symbolic state and selector commit tracking:
- babytheorist/filter/state.py
- babytheorist/symbolic.py

Concrete scene enforcement:
- babytheorist/filter/scene_sampler.py
- babytheorist/filter/shape_attrs.py

Verification only:
- babytheorist/filter/oracle.py
- babytheorist/filter/oracle_bfs.py
- babytheorist/gpu_engine.py


## 12. Future Work After Initial Main Merge

The current filter backend is ready to use as the default generation path,
but several improvements are intentionally left for follow-up work:

1. More constructive scene realization
- Current status:
	- Primary color, hollow/solid requirements, and placement constraints are
	  partially generated under the accumulated InputConstraint.
	- Some shape-attribute requirements, especially exact D4 stabilizer
	  subgroup requirements, still rely on bounded sampling and verification.
- Next step:
	- Add stabilizer-aware shape templates or generator modes so required
	  symmetry classes can be sampled directly instead of discovered by retry.

2. Better dynamic selector modeling
- Current status:
	- by_color and positional selector commits are converted into concrete input
	  constraints.
	- This is sound but can over-reject when selector expressibility is
	  conservatively approximated.
- Next step:
	- Tighten subset expressibility rules for by_color/is_hollow/positional
	  selectors so the sampler rejects fewer valid programs.

3. Broader coincidence constraints
- Current status:
	- Align/translate row/col coincidences and existing color/position forbids
	  are covered.
- Next step:
	- Extend linear and modular coincidence constraints for additional
	  add_border/translate/color interactions found by oracle sweeps.

4. Throughput engineering for 50k-scale datasets
- Current status:
	- Filter backend supports multiprocessing workers and bounded retry budgets.
	- Generation no longer depends on brute-force shortcut enumeration.
- Next step:
	- Add long-running dataset generation scripts with resumable shards,
	  per-depth metrics, and oracle sampling reports.

5. Verification cadence
- Current status:
	- Unit tests and oracle end-to-end tests cover the known shortcut families.
- Next step:
	- Keep oracle and GPU oracle tooling verification-only, and run them as
	  scheduled audits on sampled shards rather than inside generation.
