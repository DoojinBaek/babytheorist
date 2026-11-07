"""Official BabyTheorist benchmark interaction wrapper.

`BabyTheoristGym` owns the curriculum protocol. A benchmark participant usually chooses
the benchmark mode and data root:

    gym = BabyTheoristGym(mode="single-A", data_dir="data")
    while gym.active:
        data, phase = gym.interact()
        if phase == "train":
            model.fit(data)
        else:
            preds = model.predict(data)
            report = gym.submit(preds)

Each level starts with a zero-observation test. If that fails, the model
receives one train split, followed by one same-level eval submission. Passing a level
immediately advances to the next level's test. Failing all allowed eval
submissions terminates the session.

The entry test at each level is diagnostic and does not consume that level's
submission budget. Official scoring still uses the best eval score observed at
each level, including entry tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np

from env.env import (
    BenchmarkMode,
    GenerationConfig,
    OFFICIAL_TRAIN_SIZE,
    BabyTheoristEnv,
    ScoreReport,
    _EvalDataset,
    _TrainDataset,
    benchmark_mode_dir_label,
    normalize_benchmark_mode,
)


Grid = np.ndarray
Phase = Literal["train", "test"]
OFFICIAL_PASS_THRESHOLD = 0.80
OFFICIAL_LEVEL_COUNT = 8
OFFICIAL_MAX_SCORE = OFFICIAL_LEVEL_COUNT * 100.0
OFFICIAL_MAX_SCORE_AUC = sum(
    100.0 * level for level in range(1, OFFICIAL_LEVEL_COUNT + 1)
)
OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL = 1


@dataclass(frozen=True)
class LevelScore:
    level: int
    score: float


@dataclass(frozen=True)
class GymScorePoint:
    level: int
    score: float


@dataclass(frozen=True)
class GymState:
    """Public snapshot of the next interaction state.

    `phase` names what the next successful `interact()` call will serve.
    After a train payload is served, `phase` becomes "test" immediately
    because the following interaction must be a same-level eval phase.
    """

    active: bool
    level: int | None
    phase: Phase | None
    train_calls: int
    n_exposed: int
    awaiting_submit: bool
    entry_eval_pending: bool = False
    eval_submissions: int = 0
    max_eval_submissions_per_level: int = OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    charged_eval_trials: int = 0
    score: float = 0.0
    level_best_score: float = 0.0
    efficiency: float = 0.0


@dataclass(frozen=True)
class GymReport:
    level: int
    n: int
    correct: int
    accuracy: float
    passed: bool
    train_calls: int
    n_exposed: int
    eval_submissions: int = 0
    max_eval_submissions_per_level: int = OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    entry_eval: bool = False
    score: float = 0.0
    level_best_score: float = 0.0
    cumulative_score: float = 0.0
    charged_eval_trials: int = 0
    efficiency: float = 0.0


@dataclass(frozen=True)
class GymResult:
    mode: BenchmarkMode
    data_dir: Path
    highest_cleared_level: int
    failed_level: int | None
    history: tuple[GymReport, ...]
    active: bool
    max_eval_submissions_per_level: int = OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    score: float = 0.0
    efficiency: float = 0.0
    charged_eval_trials: int = 0
    score_auc: float = 0.0
    level_scores: tuple[LevelScore, ...] = ()
    score_curve: tuple[GymScorePoint, ...] = ()


EnvFactory = Callable[[Path, BenchmarkMode, int], BabyTheoristEnv]


def _default_env_factory(
    data_dir: Path,
    mode: str,
    level: int,
    generation: GenerationConfig | None = None,
) -> BabyTheoristEnv:
    return BabyTheoristEnv(data_dir=data_dir, levels=(level,), mode=mode, generation=generation)


class BabyTheoristGym:
    """Stateful official BabyTheorist curriculum interaction API."""

    def __init__(
        self,
        mode: BenchmarkMode,
        data_dir: str | Path = "data",
        *,
        generation: GenerationConfig | None = None,
        max_eval_submissions_per_level: int = OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL,
        _env_factory: EnvFactory | None = None,
    ):
        requested_mode = mode
        mode = normalize_benchmark_mode(mode)
        self._mode_dir_label = benchmark_mode_dir_label(requested_mode)
        if max_eval_submissions_per_level < 1:
            raise ValueError(
                "max_eval_submissions_per_level must be >= 1, "
                f"got {max_eval_submissions_per_level}"
            )
        self._mode = mode
        self._data_dir = Path(data_dir)
        self._generation = generation
        self._max_eval_submissions_per_level = max_eval_submissions_per_level
        self._env_factory = _env_factory or _default_env_factory
        self._envs: dict[int, BabyTheoristEnv] = {}
        self.reset()

    def reset(self) -> None:
        self._active = True
        self._level = 1
        self._phase: Phase = "test"
        self._train_calls = 0
        self._n_exposed = 0
        self._eval_submissions = 0
        self._awaiting_submit = False
        self._pending_entry_eval = False
        self._highest_cleared_level = 0
        self._failed_level: int | None = None
        self._history: list[GymReport] = []
        self._level_best_scores: dict[int, float] = {}
        self._cumulative_score = 0.0
        self._charged_eval_trials = 0

    @property
    def active(self) -> bool:
        return self._active

    @property
    def done(self) -> bool:
        return not self._active

    @property
    def mode(self) -> BenchmarkMode:
        return self._mode

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    @property
    def state(self) -> GymState:
        return GymState(
            active=self._active,
            level=self._level if self._active else None,
            phase=self._phase if self._active else None,
            train_calls=self._train_calls,
            n_exposed=self._n_exposed,
            awaiting_submit=self._awaiting_submit,
            entry_eval_pending=self._pending_entry_eval,
            eval_submissions=self._eval_submissions,
            max_eval_submissions_per_level=self._max_eval_submissions_per_level,
            charged_eval_trials=self._charged_eval_trials,
            score=self._cumulative_score,
            level_best_score=self._level_best_scores.get(self._level, 0.0)
            if self._active else 0.0,
            efficiency=self._efficiency(),
        )

    def interact(self) -> tuple[_TrainDataset | _EvalDataset, Phase]:
        """Return the next benchmark interaction payload and its phase."""
        self._ensure_active()
        if self._awaiting_submit:
            raise RuntimeError("submit predictions before requesting another interaction")

        env = self._env_for_level(self._level)
        if self._phase == "test":
            data = env.eval(level=self._level)
            self._awaiting_submit = True
            self._pending_entry_eval = (
                self._train_calls == 0 and self._eval_submissions == 0
            )
            return data, "test"

        if self._train_calls >= 1:
            raise RuntimeError("train split already served for this level")
        next_train_calls = self._train_calls + 1
        train_size = getattr(env, "train_size", None)
        if callable(train_size):
            available_n = train_size(self._level)
        else:
            available_n = OFFICIAL_TRAIN_SIZE
        next_n_exposed = min(OFFICIAL_TRAIN_SIZE, available_n)
        data = env.train(level=self._level, n=next_n_exposed)
        self._train_calls = next_train_calls
        self._n_exposed = next_n_exposed
        self._phase = "test"
        return data, "train"

    def submit(self, preds: Sequence[tuple[int, Grid]]) -> GymReport:
        """Score predictions for the pending test and advance the state."""
        self._ensure_active()
        if not self._awaiting_submit:
            raise RuntimeError("submit() requires a pending test from interact()")

        env = self._env_for_level(self._level)
        score = env.score_predictions(preds, level=self._level)
        entry_eval = self._pending_entry_eval
        if not entry_eval:
            self._eval_submissions += 1
            self._charged_eval_trials += 1
        report = self._to_gym_report(score, entry_eval=entry_eval)
        self._history.append(report)
        self._awaiting_submit = False
        self._pending_entry_eval = False

        if report.passed:
            self._highest_cleared_level = max(
                self._highest_cleared_level, self._level,
            )
            if self._level >= OFFICIAL_LEVEL_COUNT:
                self._active = False
                return report
            self._level += 1
            self._train_calls = 0
            self._n_exposed = 0
            self._eval_submissions = 0
            self._phase = "test"
        elif self._eval_submissions >= self._max_eval_submissions_per_level:
            self._failed_level = self._level
            self._active = False
        elif self._train_calls < 1:
            self._phase = "train"
        else:
            self._phase = "test"

        return report

    def result(self) -> GymResult:
        return GymResult(
            mode=self._mode,
            data_dir=self._data_dir,
            highest_cleared_level=self._highest_cleared_level,
            failed_level=self._failed_level,
            history=tuple(self._history),
            active=self._active,
            max_eval_submissions_per_level=self._max_eval_submissions_per_level,
            score=self._cumulative_score,
            efficiency=self._efficiency(),
            charged_eval_trials=self._charged_eval_trials,
            score_auc=self._score_auc(),
            level_scores=self._level_scores(),
            score_curve=self._level_score_curve(),
        )

    def summary(self) -> GymResult:
        return self.result()

    def _env_for_level(self, level: int) -> BabyTheoristEnv:
        env = self._envs.get(level)
        if env is None:
            if self._env_factory is _default_env_factory:
                env = _default_env_factory(
                    self._data_dir, self._mode_dir_label, level, self._generation,
                )
            else:
                env = self._env_factory(self._data_dir, self._mode, level)
            self._envs[level] = env
        return env

    def _to_gym_report(self, score: ScoreReport, *, entry_eval: bool) -> GymReport:
        passed = score.accuracy >= OFFICIAL_PASS_THRESHOLD
        eval_score = score.accuracy * 100.0
        level_best = max(
            self._level_best_scores.get(score.level, 0.0), eval_score,
        )
        previous_best = self._level_best_scores.get(score.level, 0.0)
        if level_best > previous_best:
            self._level_best_scores[score.level] = level_best
            self._cumulative_score += level_best - previous_best
        else:
            level_best = previous_best
        return GymReport(
            level=score.level,
            n=score.n,
            correct=score.correct,
            accuracy=score.accuracy,
            passed=passed,
            train_calls=self._train_calls,
            n_exposed=self._n_exposed,
            eval_submissions=self._eval_submissions,
            max_eval_submissions_per_level=self._max_eval_submissions_per_level,
            entry_eval=entry_eval,
            score=eval_score,
            level_best_score=level_best,
            cumulative_score=self._cumulative_score,
            charged_eval_trials=self._charged_eval_trials,
            efficiency=self._efficiency(),
        )

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("session terminated; call reset() to restart")

    def _level_scores(self) -> tuple[LevelScore, ...]:
        return tuple(
            LevelScore(level=level, score=score)
            for level, score in sorted(self._level_best_scores.items())
        )

    def _level_score_curve(self) -> tuple[GymScorePoint, ...]:
        cumulative = 0.0
        points = []
        for level in range(1, OFFICIAL_LEVEL_COUNT + 1):
            cumulative += self._level_best_scores.get(level, 0.0)
            points.append(GymScorePoint(level=level, score=cumulative))
        return tuple(points)

    def _score_auc(self) -> float:
        return sum(point.score for point in self._level_score_curve())

    def _efficiency(self) -> float:
        auc = self._score_auc()
        if auc <= 0.0:
            return 0.0
        return auc / OFFICIAL_MAX_SCORE_AUC
