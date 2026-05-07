# BabyTheorist 코드베이스 구조 메모

작성 기준: `v0.5.5` 현재 코드. 이 문서는 다음에 코드를 다시 인덱싱하지 않고도
벤치마크의 주요 계약과 구현 위치를 빠르게 복구하기 위한 지도이다.

## 0. 한 문장 요약

BabyTheorist, 논문명 BabyTheorist Environment, 는 독립 관측쌍 `(x, y)`로부터
재사용 가능한 변화 이론을 학습하는 observation-to-theory benchmark이다.
공식 실행 루프는 `env.BabyTheoristGym`이고, 데이터 생성의 production 경로는
`generate.py --backend filter`이다.

훈련에서 모델이 보아야 하는 것은 독립 row 단위 관측쌍뿐이다. 프로그램,
depth, dominant template, 같은 프로그램으로 생성된 row grouping은
generator/audit metadata이며 fair training supervision이 아니다.

## 1. 공식 벤치마크 경로

공식 benchmark interaction은 `BabyTheoristEnv`를 직접 호출하는 것이 아니라
`BabyTheoristGym`을 통해 돌아간다.

```text
model
  -> BabyTheoristGym.interact()
      -> BabyTheoristEnv.eval(level) or BabyTheoristEnv.train(level)
  -> model prediction
  -> BabyTheoristGym.submit(preds)
      -> BabyTheoristEnv.score_predictions(preds, level)
      -> io_h5.read_Y_q(...)
```

중요한 의미:

- `BabyTheoristGym`은 curriculum state machine이다.
- 각 level은 먼저 zero-observation entry eval로 시작한다.
- entry eval에 실패하면 그 level의 train split이 한 번 노출된다.
- 그 뒤 같은 level eval을 최대 5번 charged submission으로 시도한다.
- pass threshold는 `env/curriculum.py`의 `OFFICIAL_PASS_THRESHOLD = 0.80`이다.
- official score는 level별 best eval score를 point 단위로 누적한다.

채점은 `BabyTheoristGym.submit(...)` 내부에서 `BabyTheoristEnv.score_predictions(...)`를 부른다.
코드 계약은 prediction이 `Y_q` 안의 어떤 grid와 일치하면 정답이다. 다만 현재
official generation은 unique-support eval을 사용해 `len(Y_q) == 1`인 row를
저장하므로, 실제 official suite에서는 거의 `y_q` exact match처럼 동작한다.
따라서 논문에서는 다음처럼 쓰는 것이 가장 정확하다.

```text
The official BabyTheoristGym evaluator accepts any prediction in Y_q. Under the current
unique-support official suites, generation enforces singleton Y_q, so scoring
reduces to exact match against the canonical y_q.
```

## 2. 주요 공개 표면

### `generate.py`

CLI와 library generation entry point이다. production 데이터 생성은
`--backend filter`가 기본이다. `--backend rejection`은 legacy/comparison 경로이다.

주요 책임:

- benchmark mode와 primitive curriculum resolve
- level별 depth prior sampling
- abstract dominant library 준비
- filter backend 호출
- train/eval `Task` 생성
- HDF5/JSON 저장용 sidecar metadata 구성

특히 다음 함수들이 큰 흐름의 anchor이다.

- `_resolve_primitive_curriculum(...)`: suite, level -> primitive curriculum
- `_sample_piecewise_geometric_depth(...)`: official depth prior
- `generate_filter_dataset(...)`: production filter backend dataset generation
- `main(...)`: CLI parsing, config assembly, metadata, save

### `env/curriculum.py`

`BabyTheoristGym` 구현이다. 공식 benchmark protocol의 state machine이다.

중요 클래스/상수:

- `BabyTheoristGym`
- `GymState`
- `GymReport`
- `GymResult`
- `OFFICIAL_PASS_THRESHOLD = 0.80`
- `OFFICIAL_LEVEL_COUNT = 8`
- `OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL = 1`

`BabyTheoristGym`은 level별로 `BabyTheoristEnv`를 lazy-load/cache한다. 기본 factory는
`BabyTheoristEnv(data_dir=data_dir, levels=(level,), mode=mode, generation=generation)`을
만든다.

### `env/env.py`

`BabyTheoristEnv`는 static HDF5 train/eval loader와 scorer이다. 공식 사용자는
직접 쓸 수도 있지만, benchmark protocol 자체는 `BabyTheoristGym`이 감싼다.

주요 책임:

- `data/<mode>-v<version>/` resolver
- missing HDF5 auto-generation
- sidecar `.info.json` mode/depth/shape metadata validation
- train/eval dataset object 제공
- eval answer oracle API 제공
- prediction validation and scoring

중요한 fairness boundary:

- `metadata(split="train", ...)`은 train row program metadata를 반환한다.
  이것은 audit/debug용이며 fair training supervision이 아니다.
- `metadata(split="eval", ...)`은 막혀 있다.
- `answers(level, task_id)`는 eval-only oracle/debug API이다.
- `score_predictions(...)`는 `Y_q` membership으로 채점한다.

### `env/io_h5.py`

HDF5 read contract의 single source of truth이다. 다른 loader가 직접 HDF5 schema를
중복 구현하지 않도록 이 파일을 통한다.

Eval row contract:

```text
inputs, outputs: (N, P, H, W)
support slots:   [0, K)
query slot:      K
n_support[i]:    K, if present
Y_q:             valid_query_outputs[i]
```

`read_eval_row(...)`는 `support_x`, `support_y`, `query_x`, `y_q`, `Y_q`를 읽는다.
`read_Y_q(...)`는 scoring fast path이며 support를 읽지 않는다.

### `babytheorist/review.py`

`Task`를 JSON 또는 HDF5로 저장한다. HDF5 writer는 `inputs`와 `outputs`를
contiguous/uncompressed dataset으로 만든다. 모델 학습 throughput 때문에 이 계약을
깨지 않는 것이 중요하다.

HDF5 sidecar `.info.json`에는 다음 audit metadata가 들어간다.

- level, n_tasks, grid, depth histogram
- primitive histogram
- support histogram
- `babytheorist_version`, `benchmark_spec_version`
- `dataset_release_version`
- backend/filter/dominant metadata
- mode, benchmark suite, vocabulary, depth prior, shape generation settings

## 3. 생성 pipeline

Production filter backend의 큰 흐름은 다음과 같다.

```text
benchmark mode / level
  -> primitive curriculum
  -> depth prior sample
  -> abstract dominant library sample
  -> concrete selector grounding
  -> prefix symbolic state update
  -> program-rule validation
  -> input-constraint accumulation
  -> constraint-aware scene sampling
  -> engine execution
  -> Task
  -> HDF5/JSON export
```

관련 파일:

- `generate.py`: orchestration
- `babytheorist/dominant.py`: abstract dominant library
- `babytheorist/filter/grounding.py`: abstract slot -> concrete selector binding
- `babytheorist/filter/sampler.py`: candidate-enumerating production sampler
- `babytheorist/filter/state.py`: symbolic per-shape state
- `babytheorist/filter/rules_program.py`: program shortcut/minimality rules
- `babytheorist/filter/rules_input.py`: input scene constraints
- `babytheorist/filter/scene_sampler.py`: constrained shape/placement sampler
- `babytheorist/engine.py`: actual program execution and collision/OOB checks
- `babytheorist/evaluation.py`: eval `Y_q` computation

## 4. Program language

Core data types live in `babytheorist/core.py`.

```text
Shape
Selector
Step(selector, primitive, target?)
TaskPair
Task
```

Selectors:

- multi-select: `ALL`, `by_color(c)`, `is_hollow`
- single-select: `by_index(k)`, `topmost`, `bottommost`, `leftmost`, `rightmost`

`by_index(k)` is 1-based. When `by_index` is enabled, generation enforces input
raster ordering so index identity stays stable through program execution.

Primitives live in `babytheorist/primitives.py`. There are 14 primitives:

- transform: `rot90`, `flip_h`, `flip_v`, `next_color`, `fill_interior`,
  `hollow`, `add_border`
- movement: `translate_up`, `translate_down`, `translate_left`,
  `translate_right`
- relational: `swap_colors`, `align_row`, `align_col`

Relational primitives require source and target single-select selectors at the
`Step` type level.

## 5. Benchmark suites

Current supported suite names:

| suite | objects | selectors | primitive policy |
| --- | ---: | --- | --- |
| `multi_dev` | 2 | `by_index` | translate only |
| `single_easy` | 1 | `by_index` | single base only |
| `multi_easy` | 2 | `by_index` | multi base only |
| `single_hard` | 1 | `by_index` | single base plus staged unlocks |
| `multi_hard` | 3 | hard selector set | multi base plus staged unlocks |

Important: `single_hard` is not hard-selector. It is one object with `by_index`
only, plus staged primitive unlocks. `multi_hard` is the suite that uses the
full hard selector set.

Hard unlock schedule:

- `L1`: base primitives
- `L2-L3`: add `hollow`
- `L4-L5`: add `fill_interior`
- `L6-L7`: add `add_border`
- `L8+`: all unlocked primitives

`multi_dev` is the translate-only development/control suite. Some paper-facing
tables may intentionally focus on four main suites, but code and public API
support five named suites including `multi_dev`.

## 6. Depth prior

Official depth sampling is `piecewise_geometric`.

For level `L`:

- center `c = L + 1`
- 80% mass on `d <= c`
- 15% mass on shoulder depths `c+1`, `c+2`
- 5% mass on a geometrically decaying hard tail after `c+2`
- tail is truncated after six hard-tail depths
- support is `1..L+9`

Non-official recorded depth distributions should not be silently mixed with
official data.

## 7. Dominant programs

Dominant programs are generator-side latent structure, not model-facing
supervision.

The current implementation uses selector-free abstract slot programs in
`babytheorist/dominant.py`:

```text
AbstractRef(scope, slot, kind)
AbstractStep(source, primitive, target?)
AbstractProgram = tuple[AbstractStep, ...]
```

Concrete selectors are not stored in dominant templates. They are chosen later
in `babytheorist/filter/grounding.py`, then validated through the same filter stack as
ordinary sampled programs.

Hard suites use an append-only dominant library:

```text
base
append:hollow
append:fill_interior
append:add_border
```

Each append segment contains entries with exactly one newly introduced primitive.
Prefixes and suffixes are composed from the inherited library. This preserves
older dominant abstractions when new primitives unlock.

Direct abstract pools are capped at depth 10. Deeper sampled depths are generated
by composing cached chunks.

Depth 1 has a special policy in `v0.5.5`: `_canonical_depth_one_pool()` keeps one
canonical entry per valid primitive to avoid selector variant multiplicity making
some primitives appear more often at depth 1.

## 8. Eval and `Y_q`

Eval tasks are support/query transfer probes.

```text
support: (x_1, y_1), ..., (x_K, y_K)
query:   x_q
answer:  any y in Y_q
```

`babytheorist/evaluation.py` computes alternatives by replacing selectors with selectors
that agree on the support scene(s), then executing those alternatives on the
query scene. The result set is `Y_q`.

Current official generation uses adaptive unique support:

- minimum support: `K >= 2`
- maximum support: `K <= 5`
- generation accepts an eval row only when `len(Y_q) == 1`
- `n_support` records the row-specific `K`

Therefore:

- Code-level scorer is `Y_q` membership.
- Current official suite behaves like canonical `y_q` exact match because
  `Y_q` is singleton.
- Legacy or custom ambiguous eval files with larger `Y_q` remain scoreable.

## 9. Data layout and versioning

Version source of truth:

- `babytheorist/version.py`
- current code/spec version: `0.5.5`

Default suite directories:

```text
data/single_easy-v<version>/
data/multi_dev-v<version>/
data/multi_easy-v<version>/
data/single_hard-v<version>/
data/multi_hard-v<version>/
```

Dominant caches:

```text
data/dominant_pools-v<version>/
```

Do not hard-code versioned names when helper functions exist. Local `data/` may
contain stale data from old versions; always inspect `.info.json` before using it
as evidence.

Current observed local caveat at the time this doc was written:

- local `data/multi_hard-v0.5.0/` exists
- sidecars report `babytheorist_version = 0.5.0`
- this is stale relative to current code `v0.5.5`

## 10. Generation scripts

`generate.py` is the public generation entry point for individual JSON and HDF5
files. Compact review samples can be generated with:

- `sample_data_gen.sh`
- `sample_single_a.sh`
- `sample_multi_a.sh`
- `sample_single_b.sh`
- `sample_multi_b.sh`

Official generation defaults are:

- levels `1..8`
- train `250,000` per level
- eval `10,000` per level
- `SHAPE_PROFILE=standard`
- `--backend filter`
- dominant pool cache under `data/dominant_pools-v<version>/`

## 11. Shape generation

Shared shape profile defaults live in `shape_profiles.py`.

Current default profile is `standard`:

- shape size range `4..16`
- compactness `8.0`
- bordered/hollow minimum target size `9`
- randomized bordered/hollow inner walk sizing
- max object bounding-box span `5`

`BabyTheoristEnv` validates shape profile metadata before reusing existing generated
files when auto-generation is active.

## 12. Tests and validation

Full validation:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q
```

Focused generation/env validation:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q \
  tests/env \
  tests/babytheorist/test_generate.py \
  tests/babytheorist/test_dominant.py \
  tests/babytheorist/test_parallel.py \
  tests/babytheorist/filter/test_sampler.py
```

Script syntax validation:

```bash
bash -n sample_data_gen.sh
bash -n sample_single_a.sh
bash -n sample_multi_a.sh
bash -n sample_single_b.sh
bash -n sample_multi_b.sh
```

At the time this document was written, full test baseline was:

```text
1061 passed
```

## 13. Known paper/docs drift to check

These are not necessarily code bugs, but they are important for paper writing.

- Generated documentation may lag `babytheorist/version.py`; check version before trusting it.
- `00_PARER_WRITING/docs/sketch.md` may still mention `v0.5.1` paths.
- Any statement saying `single_hard` uses the hard selector set is stale.
  Current code makes `single_hard` `by_index` only.
- Any statement saying official scoring is only target-grid exact match should
  clarify the `Y_q` membership contract and the singleton-`Y_q` official-suite
  special case.
- Some paper-facing tables may omit `multi_dev`; decide whether that is an
  intentional main-suite presentation or an accidental omission.

## 14. Quick file index

```text
generate.py
  CLI, suite/curriculum/depth resolution, generation orchestration, metadata

env/curriculum.py
  BabyTheoristGym official state machine and scoring protocol wrapper

env/env.py
  BabyTheoristEnv static loader, auto-generation, metadata validation, scorer

env/io_h5.py
  HDF5 read contract and Y_q decoding/canonicalization

babytheorist/review.py
  JSON/HDF5 writer and sidecar metadata writer

babytheorist/core.py
  Shape, Selector, Step, TaskPair, Task

babytheorist/primitives.py
  Primitive registry and primitive semantics

babytheorist/selectors.py
  Concrete selector resolution

babytheorist/engine.py
  Program execution, collision/OOB/no-op/precondition failure semantics

babytheorist/evaluation.py
  Support/query selector ambiguity and Y_q computation

babytheorist/dominant.py
  Abstract dominant program/library construction and cache codec

babytheorist/filter/sampler.py
  Candidate-enumerating production program sampler

babytheorist/filter/grounding.py
  Abstract slot grounding to concrete selectors

babytheorist/filter/state.py
  Symbolic state history for filter validation

babytheorist/filter/rules_program.py
  Program-level shortcut/minimality rules

babytheorist/filter/rules_input.py
  Input scene constraints

babytheorist/filter/scene_sampler.py
  Constraint-aware concrete scene sampling

shape_profiles.py
  Shared shape-generation profile defaults

babytheorist/version.py
  Code/spec version source of truth
```
