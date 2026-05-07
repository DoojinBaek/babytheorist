# BabyTheorist Filter 시스템 가이드

이 문서의 목적은 하나입니다.

> 누군가 필터 코드를 처음 열었을 때, "이게 왜 이렇게 생겼고 무엇을 막고 있는가"를
> 20분 안에 납득할 수 있게 한다.

난이도 순서: 문제 정의 → 필터의 연산 장치(symbolic state) → 파이프라인 →
각 룰의 의미 → 데이터 구조 → 오라클.

---

## 0. 필터가 푸는 문제

### 0.1 "depth N 프로그램"이란?

BabyTheorist는 각 task에 **program depth**를 할당합니다. 이론상 level N 데이터셋의 모든
task는 길이 N 프로그램의 결과여야 합니다. 그런데 단순히 길이 N 프로그램을 쓴다고
해서 task의 실제 난이도가 N인 것은 아닙니다.

### 0.2 Shortcut 문제

예: 길이 3 프로그램 `P = [rot90, flip_v, rot90]`. 이게 shape에 주는 최종 효과는
`rot² · flip_v = flip_h`. 즉 **길이 1 프로그램** `[flip_h]`와 동일. 이러면 학습자
입장에서 task는 depth-1인데 label만 depth-3이 됩니다. 이런 **shortcut**(더 짧은
동등 프로그램이 존재하는 경우)을 생성 단계에서 걸러야 데이터셋의 depth 라벨이
신뢰할 수 있는 난이도 신호가 됩니다.

### 0.3 두 층위의 shortcut

| 종류 | 원인 | 방어 |
|---|---|---|
| **프로그램 자체** shortcut | 프로그램 구조가 대체 가능 (`rot×4 = id`, selector 축약 등) | **R 룰** (program-level) |
| **Scene-specific** shortcut | 프로그램은 정상인데 특정 scene에서 우연 shortcut (정사각형 shape, 색 일치 등) | **I 제약** (input-level) |

필터는 이 둘을 동시에 잡습니다. R 룰이 프로그램을 검사하고, I 제약이 "어떤 scene이
그 프로그램과 동거해도 안전한가"를 누적해서 scene 샘플러에게 넘깁니다.

### 0.4 핵심 원칙

1. **Rule-based, symbolic lookup**. 생성 중에는 BFS/brute-force 없음. 모든 룰은
   O(program length × n_shapes) 이내의 symbolic 테이블 질의.
2. **Forward-accumulating**. 매 step마다 상태 + 제약을 한 단계 전진시키고,
   통과한 것만 채택. 실패한 경로를 끝까지 돌려보지 않음.
3. **Oracle은 검증 전용**. 생성 파이프라인에 절대 들어가지 않음.

---

## 1. Symbolic state — 필터의 연산 장치

### 1.1 왜 symbolic인가?

대안은 "매 step마다 실제 픽셀 grid를 렌더해서 비교하기". 이게 안 되는 이유:

- scene마다 결과가 달라져서 "프로그램이 shortcut인지" 판정이 scene-dependent.
- 매 step × 매 scene 렌더 비용.

대신 필터는 **scene을 뽑기 전에** 각 shape를 "아직 크기·위치·색이 정해지지 않은
추상 객체"로 두고, 그 객체에 **어떤 변환이 어떤 순서로 누적됐는지**만 추적합니다.
그 누적 정보가 `ShapeState`.

### 1.2 ShapeState — 6 필드

`babytheorist/symbolic.py:ShapeState`

| 필드 | 타입 | 의미 | 연관된 primitive |
|---|---|---|---|
| `orientation` | `D4(rot ∈ 0-3, flipped ∈ {T, F})` | D4군 원소 (회전·뒤집기 누적) | rot90, flip_h, flip_v |
| `border_count` | `int` | add_border 누적 횟수 | add_border |
| `row` | `LinExpr` | 현재 y 좌표 (입력변수 선형식) | translate_{up,down}, add_border, align_row |
| `col` | `LinExpr` | 현재 x 좌표 (입력변수 선형식) | translate_{left,right}, add_border, align_col |
| `color` | `(base, offset mod 9)` | "어느 shape 초기색을 몇 번 next_color했나" | next_color, swap_colors |
| `grid_shape` | `"solid" \| "hollow"` | 내부 속·빈 상태 | hollow, fill_interior |

등식(`==`)은 **field-wise**. 두 상태가 같다 ⇔ 6 필드 전부 일치.

### 1.3 LinExpr — 좌표를 왜 선형식으로?

`row`, `col`을 정수로 저장하면 안 됩니다. 이유:

- `translate_up`은 "shape 높이만큼 row 감소". 그런데 shape 높이는 rotation 평행도에
  따라 바뀜 — rotation이 짝수면 `h_i`, 홀수면 `w_i`(h와 w가 rot90으로 맞바뀜).
- 즉 translate의 효과가 **앞선 rotation에 의존**. 정수 카운트로는 이 coupling을
  못 표현.

해법: `row`를 입력 변수 `{h_i, w_i, r_i, c_i}` 위의 정수 계수 선형식으로 저장.

```python
@dataclass(frozen=True)
class LinExpr:
    coeffs: tuple[tuple[str, int], ...]  # sorted, 0-coeff 제거
    const: int = 0
```

Canonical form이라 **syntactic equality = semantic equality**. 두 LinExpr이
`==`이면 모든 변수 assignment에서 값이 같음.

### 1.4 작은 walk-through (1 shape)

초기 상태
```
orientation = e,  border=0,  row=r_0,  col=c_0,  color=(0,0),  grid=solid
```

step 1 — `rot90`: orientation만 바뀜
```
orientation = r,  row=r_0,  col=c_0,  ...
```

step 2 — `translate_up`: 현재 rotation이 홀수라 `current_height = w_0`
```
orientation = r,  row = r_0 - w_0,  ...
```

step 3 — `rot90`: orientation = r² (row·col은 그대로; translate 효과는 이미 row에
구워짐)
```
orientation = r²,  row = r_0 - w_0,  ...
```

step 4 — `translate_up`: 이번엔 rotation이 짝수라 `current_height = h_0`
```
orientation = r²,  row = r_0 - w_0 - h_0,  ...
```

이 시점의 state tuple은 초기 tuple과 다릅니다 → R3(state revisit) 통과.

하지만 **정사각형 shape**(h_0 = w_0)라면 픽셀 수준에서 `translate_up` 2번과
구분이 안 됩니다. 이건 symbolic이 못 잡는 blind spot이고, 입력 측에서
`require_non_square`로 방어합니다 (§3.2 참고).

### 1.5 FilterState — ShapeState들의 orchestrator

`babytheorist/filter/state.py:FilterState`

세 필드만 기억하면 됩니다.

| 필드 | 내용 |
|---|---|
| `current` | 현재 per-shape ShapeState 튜플 (길이 = n_shapes) |
| `history` | step 0 전부터 현재까지 모든 snapshot. `history[k]` = step k 적용 **직전** 상태 |
| `self_resolutions[k]` | step k의 selector가 resolve된 shape index 튜플 |

`self_resolutions`가 따로 있는 이유:
- `ALL`, `by_index`, `is_hollow`는 **symbolic하게** 결정 가능 (state만 보면 됨).
- `by_color`, `topmost/...`는 scene 없이는 결정 불가 → 샘플러가 "이번 step은
  `{0, 2}`에 적용하기로 한다"고 **commit**하고 이 결정을 여기에 기록.
- 이후의 모든 룰·제약은 이 commit을 공유해서 해석이 엇갈리지 않음.

### 1.6 Symbolic state가 **일부러** 못 보는 것들

다음은 전부 scene-time에만 결정되므로 ShapeState가 모릅니다:
- shape의 실제 픽셀 모양, 구체적 크기, 구체적 시작 위치/색
- `r_0 = r_1` 같은 shape 간 위치 일치
- `h_i = w_i` 같은 scene-specific 정수 일치

이것들은 **모두 I 측**(I1 stabilizer, vocab requirements, by_color/positional
commitment)에서 "어떤 scene이면 이 프로그램이 유효한가"라는 제약 형태로 내려갑니다.

> 필터 구조의 핵심 분업:
> - **R 측** = symbolic state로 프로그램 구조 검증
> - **I 측** = scene 조건으로 프로그램-scene 결합 검증

---

## 2. 필터 파이프라인

`babytheorist/filter/sampler.py:sample_program_with_constraint`가 오케스트레이터. 의사코드:

```
fs := FilterState.initial(n_shapes)
constraint := InputConstraint.empty(n_shapes)
program := []

for depth_so_far in 1..target_depth:
    repeat until success or budget exhausted:
        step := _sample_primitive(...)           # 후보 primitive + selector
        try:
            fs_next := fs.apply(step, ...)        # symbolic 적용
        except PreconditionFail:
            continue
        v := program_is_valid(program+[step], fs_next, ...)   # R 룰 전부
        if v is not None: continue
        try:
            cnext := accumulate_constraint(program+[step], fs_next, ...)  # I 제약
        except InfeasibleConstraint: continue
        if cnext.is_infeasible(): continue
        fs, constraint, program := fs_next, cnext, program+[step]
        break
return program, fs, constraint
```

핵심 성질:
- 한 번에 한 step, 매 step마다 symbolic apply + R 검사 + I 누적 전부 통과해야 채택.
- 실패하면 해당 step만 다시 뽑음 (백트래킹 아님).
- 제약은 **단조 감소**(완화 없음). 한 번 infeasible이면 그 프로그램은 영구 탈락.

샘플링 후에는 `sample_scene`이 누적 constraint를 만족하는 scene을 뽑고, 최종
`apply_program` 결과가 input과 동일하면 폐기(식별 task는 쓸모없음).

---

## 3. Program 룰 (R 측)

모두 `babytheorist/filter/rules_program.py`. 실행 순서는 `program_is_valid` 안에서
**비용이 낮은 것부터** → R1, R3, R6, I6(joint orchestration), R5, R5b, R4, R7'.

각 룰은 `None`(통과) 또는 `RuleViolation(rule, reason)`을 반환.

### R1. Count limits

- **한 줄**: primitive마다 shape당 허용 횟수가 있고, 이를 넘으면 탈락.
- **근거**: `rot90` cyclic order 4 → 4번 적용은 identity로 복귀 (R3가 잡을 수도
  있지만 R1이 더 싸게 먼저 잡음). involution(flip, hollow)은 1번, cyclic
  (rot90, next_color)은 order-1번까지 허용.
- **예**: 같은 shape에 `rot90`이 4번 → 즉시 R1 위반.

### R3. State revisit

- **한 줄**: 어떤 shape라도 자기 ShapeState가 과거에 나온 값으로 돌아오면 탈락.
- **구현**: 각 shape의 trajectory에서 연속 중복(그 step에 건드려지지 않은 idle
  step)을 먼저 압축 → 집합에 넣으면서 중복 발생 시 violation.
- **왜 idle 압축을 먼저?**: selector가 그 shape를 건드리지 않은 step은 그냥
  "아무 일도 안 일어남"이라 재방문의 반례가 아니기 때문.
- **예**: `[rot90]×4` → 4번째에 초기 상태 재방문.

### R6. D4 minimal-word

- **한 줄**: 한 shape에 쓴 D4 primitive 수가, 그 shape에 누적된 D4 net 원소의
  **최소 단어 길이**와 같아야 함.
- **근거**: D4는 원소 8개, 최장 최소 단어 길이 3. 테이블로 미리 계산해둠.
- **왜 필요**: R3를 통과했어도 "다른 경로가 더 짧다"는 경우가 있음. 예:
  `[rot90, flip_v, rot90]` → net = flip_h (단어 길이 1). 3 step 써서 1 step에
  도달 가능한 결과를 만든 것 → 탈락.
- **구현**: 각 shape별 D4-부분프로그램의 net을 누적해 `word_length(net)`과 비교.

### I6. Joint orchestration shortcut (트레이스 정렬)

이름에 "I6"이 붙었지만 현 구현은 **R 측 룰**로 분류되어 `program_is_valid` 내부에서
돕니다 (판정이 프로그램만 보고 가능하기 때문).

- **한 줄**: 프로그램 P의 각 shape write trace가 있을 때, **더 짧은 프로그램 Q**가
  동일한 per-shape trace를 만들 수 있으면 탈락.
- **알고리즘**: 상태 `(c_0, ..., c_{n-1})` = 각 shape가 trace의 어느 위치까지
  소비했는지, 목표 `goal = (|trace_0|, ..., |trace_{n-1}|)`. BFS로 최소
  step 수를 구함. len(P)보다 작으면 shortcut 존재.
- **상태 공간**: `∏(|trace_i| + 1) ≤ (d+1)^n`. d ≤ 6, n ≤ 4면 ≤ 2401 — 미미함.
- **예**: P = `[by_idx(1).rot90, by_idx(2).rot90]` → Q = `[ALL.rot90]` (len 1) 존재.

> BFS 불가 원칙에 예외처럼 보일 수 있지만, state space가 ≤ 2401로 **유한하고 작은
> 테이블 탐색**이라 rule-based lookup과 비용상 동급. 상세 근거는
> `2026-04-22_rule_based_constraint_forward_design.md` 참조.

### R5. Translate cancellation

- **한 줄**: shape별, row/col축별로, **쓴 translate 수 > 최소 필요 수**면 탈락.
- **원리**: 최종 row의 계수가 `α·w_i + β·h_i`면 필요한 최소 translate 수는
  `|α| + |β|`. 현재 쓴 translate 수가 이보다 많으면 cancellation 존재.
- **왜 adjacency만 보면 부족?**: `[trans_right, trans_up, trans_left]` — 중간
  `trans_up`이 col축을 건드리지 않아서 right/left가 **투명하게 상쇄**. 인접이
  아니라 못 보는 것을 R5가 잡음.

### R5b. Grid-shape cancellation

- **한 줄**: 한 shape에 `hollow`와 `fill_interior`가 둘 다 쓰이면 탈락.
- **근거**: R1이 각각을 1번으로 캡하므로, 두 번째가 찍혔다는 건 왕복이라는 뜻.
  R3는 다른 축(border_count, orientation)이 사이에 바뀌면 full state 동등성이
  깨져서 못 잡음. R5b가 grid_shape 축만 보고 잡음.

### R4. Dead writes

- **한 줄**: 어떤 step이 쓴 **모든** 값이 read 없이 뒤에서 덮이면 탈락.
- **테이블**: `babytheorist/symbolic.py:PRIMITIVE_IO` — 각 primitive가 `(self|target,
  property)`를 어떻게 read/write하는지 정의. precondition-only read는 제외되어
  있음(die 결정에 영향 주면 안 됨).
- **예**: `align_row(A, B); align_row(A, C)` — 첫 align의 row write는 두 번째
  align이 read 없이 덮음 → 첫 step dead.

### R7'. Trace identity (selector union aliasing)

- **한 줄**: 서로 다른 shape들이 **같은 write 시퀀스**를 받고, 그 shape 집합이
  단일 더 넓은 selector로 표현 가능하면 탈락.
- **예**: shape 1, 2에 같은 rot90을 `by_index`로 각각 적용 → `ALL.rot90` 1 step으로
  축약 가능 → 탈락.
- **현재 지원 축약 selector**: `ALL`, `is_hollow`, `by_color` (후자는 palette +
  constraint 정보로 expressibility 체크).
- **dependency window**: relational primitive가 들어오면 trace 동등 판정 중간에
  "closing" 개념이 필요. D4/translate/색/형 vocab까지는 no-op이지만 R10' 이후
  활성화. 구현은 `babytheorist/filter/trace_identity.py`.

---

## 4. Input 제약 (I 측)

모두 `babytheorist/filter/rules_input.py`. `accumulate_constraint`가 프로그램 하나를
받아서 `InputConstraint` 객체를 누적 반환.

### I1. Stabilizer-depth 제약

- **한 줄**: 프로그램이 shape `i`에 쓴 D4 net이 `g_i`이고 그 shape에 D4 step 수가
  `d_i`면, scene-time shape의 stabilizer `H`는 반드시
  `min_{h ∈ g_i·H} word_length(h) ≥ d_i`를 만족.
- **풀이**: "shape가 너무 대칭이면 D4 net이 원래 shape 자체에 **같은 결과**를 주는
  더 짧은 경로가 생긴다". 예를 들어 shape가 `rot²`-대칭이면 `rot²`은 shape 본인에
  아무 효과가 없으므로 길이가 0으로 줄어들 여지 발생.
- **구현**: `stabilizers_allowed_at_depth(net, depth, d4_vocab)` 테이블 lookup.
  결과가 빈 집합이면 즉시 infeasible.

### Vocab-wide 제약

특정 primitive 조합이 symbolic이 못 보는 blind spot을 만들면 shape 조건을 선제
강제. 현재 2개:

| 조건 | 강제 |
|---|---|
| vocab에 `rot90` + 아무 translate가 둘 다 있음 | 모든 shape `require_non_square` |
| vocab에 `add_border` + 아무 translate가 둘 다 있음 | 모든 shape `min_bbox_dim ≥ 3` |

**왜 필요한가**:
- 정사각형(h=w) shape에서는 `rot90`이 translate와 우연히 상쇄됨. symbolic은 h와
  w를 다른 변수로 보므로 이 일치를 감지 불가.
- `h=2` shape에서는 `[trans_up×2, add_border]`와 `[add_border, trans_up]`이
  픽셀 수준에서 같아짐 (`2h + 1 = h + 3` when `h = 2`). LinExpr은 syntactic
  equality만 하므로 정수 방정식은 못 푼다.

### By_color commitment

`by_color(c)` selector는 **scene의 실제 primary color로만** resolve 가능합니다.
샘플러는 이 step에서 "어떤 shape 부분집합이 이 색에 해당하는가"를 임의로 commit:

- commit된 shape `i`: scene-time primary color가 `(c - offset_i) mod 9` 여야 함
  → `require_primary(i, ...)`.
- commit 안 된 shape `j`: palette에 `(c - offset_j) mod 9`가 포함되어 있으면
  → `forbid_primary(j, ...)`.

이렇게 하면 scene 샘플러가 shape를 뽑을 때 이 제약을 만족하는 shape만 통과합니다.

### Positional commitment

`topmost/bottommost/leftmost/rightmost` selector도 scene 없이는 resolve 불가.
샘플러가 winner shape를 임의 commit하면 그 결과를 ordering 제약으로 변환:

- `topmost` 이 shape `w`로 commit → `row(w) < row(j)` for all `j ≠ w`.
- 다른 3개도 대칭.

모든 ordering은 `PositionalOrdering(lt, gt, axis)`로 누적. 새 ordering이 들어왔을
때 기존과 **순환**(cycle)을 만들면 즉시 infeasible.

### 초기 grid_shape 고정

`initial_grid_shapes`가 주어지면(예: 프로그램에 `fill_interior`가 있어서 해당
shape가 hollow여야 함), scene 샘플러도 그 초기 타입 shape만 뽑게 됨.

---

## 5. InputConstraint 데이터 구조

`babytheorist/filter/constraint.py`

### ShapeConstraint (shape 하나의 제약)

| 필드 | 의미 |
|---|---|
| `allowed_stabilizers` | 허용 stabilizer 집합 (비면 infeasible) |
| `required_primary_color` | 강제 primary color (있을 수도 없을 수도) |
| `forbidden_primary_colors` | 금지된 primary color 집합 |
| `required_grid_shape` | `"solid"` 또는 `"hollow"` |
| `require_non_square` | True면 scene에서 h ≠ w 강제 |
| `min_bbox_dim` | 최소 bbox 변 길이 |

### InputConstraint (scene 전체의 제약)

- `shapes`: 길이 n의 `tuple[ShapeConstraint, ...]`.
- `row_forbids` / `col_forbids`: `(shape_i, shape_j, offset)` 형태의 좌표 일치
  금지. 후기 phase에서 활성화.
- `primary_forbids`: 색 일치 금지.
- `positional_orderings`: strict order edge 집합.

### 불변식

- 제약은 **한 방향으로만** 누적(완화 없음). 이게 수학적 정의를 깔끔하게 해줌.
- 특정 shape의 `allowed_stabilizers`가 비면 즉시 `is_infeasible()`.
- 좌표/색 일치류는 scene 샘플링 단계에서 실제 배치로 검증.

---

## 6. Scene 샘플러

`babytheorist/filter/scene_sampler.py:sample_scene`

한 번의 시도에서 셋 다 통과해야 성공:

1. **Shape 속성 체크** (shape 하나씩):
   - stabilizer 허용집합에 속하는가
   - primary color require/forbid 만족
   - grid_shape 일치
   - require_non_square 만족
   - min_bbox_dim 만족
2. **배치 가능성**: `place_shapes`가 world 안에서 간격·충돌 제약으로 실제 배치.
   전체를 못 배치하면 전부 버리고 재시도 (all-or-nothing).
3. **Cross-shape 체크** (배치 후):
   - row/col/primary coincidence forbid
   - positional ordering

총 retry budget이 소진되면 샘플러가 `None` 반환 → 호출자가 프로그램 단계부터
다시.

---

## 7. Selector 해석 vs commitment

Selector는 두 부류.

### Symbolic하게 해석되는 selector

`ALL`, `by_index`, `is_hollow`

이것들은 ShapeState만 보고 즉시 resolve 가능 (grid_shape는 ShapeState 필드).
`resolve_symbolic`이 그대로 반환.

### Commit이 필요한 selector

`by_color`, `topmost`, `bottommost`, `leftmost`, `rightmost`

이것들은 **scene-time 데이터**에 의존. 예를 들어 `by_color(3)`은 실제 primary color
3인 shape들을 찾아야 하는데, scene이 없으면 어떤 shape인지 결정 불가.

샘플러의 처리:
1. 후보 step의 selector가 commit-type이면 임의로 shape 부분집합을 뽑음.
2. 이 commit을 `override_indices`로 `apply_symbolic_step`에 넘겨 상태를 진행.
3. 동시에 `self_resolutions`에 기록해 이후 룰·I 제약이 공유.
4. 해당 commit을 강제하는 input constraint 추가 (by_color → require/forbid
   primary, positional → ordering).

---

## 8. D4 테이블은 왜 독립 모듈인가

`babytheorist/filter/d4_tables.py`가 제공:

- `compose(g, h)`, `inverse(g)` — D4 연산
- `word_length(g)` — 최소 단어 길이 (0..3)
- D4 subgroup 리스트 (전체 10개)
- `stabilizers_allowed_at_depth(net, depth, vocab)` — I1용

이유: R6·I1·I6·R7'이 전부 이 테이블을 공유. 룰 체크 안에서 BFS를 안 돌고, 상수
시간 lookup으로 처리.

---

## 9. End-to-end walk-through (depth=3, n_shapes=2)

1. step1 후보 `Step(ALL, rot90)` 뽑힘 → `fs.apply` 성공 → R 전부 통과 → I 누적.
2. step2 후보 `Step(by_index(1), translate_right)` 뽑힘 → 상태 진행 → R5/R4에서
   탈락하는 경우 재샘플.
3. 통과 후보로 depth 2짜리 prefix 확정.
4. step3까지 완료하면 최종 `InputConstraint` 고정.
5. `sample_scene`이 constraint 만족 scene 생성.
6. `apply_program(program, shapes)` 결과가 input과 같으면 폐기(의미 없는 task).
   다르면 task 확정.

---

## 10. 오라클 (검증 전용)

- **생성 중 사용 안 함**. 필터의 rule-based 판정이 정답이라는 전제.
- **신뢰성 테스트용**: 완성된 task에 대해 "정말 depth-N 미만 프로그램이 존재하지
  않는가"를 brute-force로 확인.

파일:
- `babytheorist/filter/oracle.py` — 열거 기반.
- `babytheorist/filter/oracle_bfs.py` — BFS 중복 제거로 속도 개선.
- `babytheorist/gpu_engine.py` — GPU 가속 오라클 (대량 검증용), CPU 재검증 포함.

정책:
> 생성은 symbolic rule-based. 오라클은 품질 보증용.

---

## 11. 코드 탐색 시작점

| 용도 | 파일 |
|---|---|
| 샘플링 메인 | `babytheorist/filter/sampler.py` |
| 프로그램 룰 | `babytheorist/filter/rules_program.py` |
| Input 제약 | `babytheorist/filter/rules_input.py` |
| Symbolic 상태 | `babytheorist/symbolic.py` |
| 상태 + 히스토리 wrapper | `babytheorist/filter/state.py` |
| Trace-identity (R7') | `babytheorist/filter/trace_identity.py` |
| Constraint 모델 | `babytheorist/filter/constraint.py` |
| Scene 구체화 | `babytheorist/filter/scene_sampler.py` |
| D4 테이블 | `babytheorist/filter/d4_tables.py` |
| 오라클 (CPU) | `babytheorist/filter/oracle.py`, `oracle_bfs.py` |
| 오라클 (GPU) | `babytheorist/gpu_engine.py` |

---

## 부록 A — 자주 헷갈리는 점

**Q1. `ShapeState.row`가 왜 LinExpr이지? scene마다 r_0, h_0 값이 다를 텐데?**
A. 맞습니다. 그래서 scene-**independent**한 구조입니다. `row = r_0 - w_0 - h_0`는
"어떤 scene이든 초기 row에서 w와 h만큼 뺀 위치"라는 식(equation)이고, 필터는
이 식의 **구조**만 비교합니다. 구체적 값은 scene을 뽑은 뒤 평가.

**Q2. R3와 R6의 차이?**
R3는 "**같은 state**로 돌아왔나"만 본다(strict 재방문). R6는 "같은 state로 돌아오지
않았지만 더 짧은 경로가 있었나" 본다. R3 통과 + R6 실패 프로그램이 존재한다:
`[rot90, flip_v, rot90]`는 trajectory가 다 다르지만 net = flip_h = 1-step.

**Q3. R7'와 I6 (joint orchestration)의 차이?**
R7'는 **트레이스 동등**(같은 write 시퀀스)을 찾아서 wider selector로 축약 가능한지
본다. I6는 더 일반적으로, **정해진 시퀀스 제약 하에 최소 step 수**를 BFS로 구해서
len(P)와 비교한다. R7'를 일반화한 것이 I6.

**Q4. 필터는 왜 commit을 샘플러가 하는가?**
`by_color` 같은 selector는 scene이 없으면 정의되지 않음. 그렇다고 scene을 먼저
뽑아 두기에는 scene 샘플링 비용이 너무 큼. 타협: 샘플러가 "이 step에서 이
선택이었다고 가정하자"고 commit해서 프로그램 진행을 symbolic하게 끝내고, 그
commit을 input 제약으로 내려 scene 샘플링 단계에서 "그 commit을 실현하는 scene"을
뽑게 함. 이렇게 하면 프로그램 루프는 scene 독립적으로 돌고, scene 루프는
프로그램-aware하게 돔.

---

## 부록 B — main merge 이후 발전 방향

이번 PR 버전은 필터 backend를 기본 생성 경로로 쓰기 위한 최소 안정 버전이다.
다만 대량 생성과 rule coverage를 더 높이려면 아래 항목을 후속 작업으로 진행한다.

### 1. 더 constructive한 scene 실현

현재:
- primary color, hollow/solid, 일부 placement constraint는 `InputConstraint`에 맞춰
  직접 생성하거나 배치 중 가지치기한다.
- 하지만 정확한 D4 stabilizer subgroup 같은 shape attribute 조건은 아직 제한된
  sampling + 검증에 의존한다.

후속:
- stabilizer-aware shape template 또는 generator mode를 추가해서 필요한 대칭군을
  retry로 찾지 않고 직접 샘플한다.

### 2. Dynamic selector 모델 개선

현재:
- `by_color`, positional selector commit은 input constraint로 내려 scene에서 실현한다.
- soundness를 우선해서 일부 subset expressibility는 보수적으로 근사한다.

후속:
- `by_color` / `is_hollow` / positional selector의 표현 가능 subset을 더 정밀하게
  모델링해서 유효한 프로그램의 과도한 탈락을 줄인다.

### 3. Coincidence constraint 확장

현재:
- align/translate의 row/col coincidence, color/position forbid는 커버한다.

후속:
- oracle sweep에서 발견되는 `add_border`/translate/color 상호작용을 linear/modular
  coincidence constraint로 추가한다.

### 4. 5만 장 단위 throughput 엔지니어링

현재:
- filter backend는 multiprocessing worker와 bounded retry budget을 지원한다.
- 생성 경로에는 brute-force shortcut enumeration이 없다.

후속:
- depth별 shard 생성 스크립트, resume 가능한 output, per-depth acceptance metric,
  oracle sampling report를 붙인다.

### 5. 검증 cadence 유지

현재:
- unit test와 oracle end-to-end test가 알려진 shortcut family를 커버한다.

후속:
- oracle/GPU oracle은 생성 경로에 넣지 않고 verification-only로 유지한다.
- 대량 shard에서 주기적으로 audit sample을 뽑아 shortcut 누락 여부를 점검한다.
