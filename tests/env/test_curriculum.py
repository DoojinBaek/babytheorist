"""Unit tests for the official BabyTheoristGym interaction protocol."""

from __future__ import annotations

from pathlib import Path

import pytest

from env import (
    GenerationConfig,
    GymReport,
    GymScorePoint,
    LevelScore,
    BabyTheoristGym,
    ScoreReport,
)
from env.curriculum import (
    OFFICIAL_LEVEL_COUNT,
    OFFICIAL_MAX_SCORE_AUC,
    OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL,
    OFFICIAL_TRAIN_SIZE,
)


class ScriptedFactory:
    def __init__(
        self,
        accuracies,
        *,
        train_size=OFFICIAL_TRAIN_SIZE,
        eval_size=100,
    ):
        self._accuracies = iter(accuracies)
        self.train_size = train_size
        self.eval_size = eval_size
        self.calls = []

    def __call__(self, data_dir: Path, mode: str, level: int):
        self.calls.append((Path(data_dir), mode, level))
        return StubEnv(level, self, self.train_size, self.eval_size)

    def next_score(self, level: int) -> ScoreReport:
        acc = next(self._accuracies)
        correct = round(acc * self.eval_size)
        return ScoreReport(
            level=level,
            n=self.eval_size,
            correct=correct,
            accuracy=correct / self.eval_size,
        )


class StubEnv:
    def __init__(self, level: int, factory: ScriptedFactory,
                 train_size: int, eval_size: int):
        self.level = level
        self._factory = factory
        self._train_size = train_size
        self._eval_size = eval_size

    def train(self, level: int, n: int | None = None):
        assert level == self.level
        if n is None:
            n = self._train_size
        if not 0 <= n <= self._train_size:
            raise ValueError(f"n={n} out of [0, {self._train_size}]")
        return StubDataset(level, n, "train")

    def train_size(self, level: int) -> int:
        assert level == self.level
        return self._train_size

    def eval(self, level: int):
        assert level == self.level
        return StubDataset(level, self._eval_size, "test")

    def score_predictions(self, preds, level: int):
        assert level == self.level
        return self._factory.next_score(level)


class StubDataset:
    def __init__(self, level: int, n: int, split: str):
        self.level = level
        self._n = n
        self.split = split

    def __len__(self):
        return self._n


def _gym(
    accuracies,
    *,
    train_size=OFFICIAL_TRAIN_SIZE,
    data_dir="bench-root",
    max_eval_submissions_per_level=OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL,
):
    factory = ScriptedFactory(accuracies, train_size=train_size)
    gym = BabyTheoristGym(
        mode="single-A",
        data_dir=data_dir,
        max_eval_submissions_per_level=max_eval_submissions_per_level,
        _env_factory=factory,
    )
    return gym, factory


def test_initial_interact_returns_level_one_test():
    gym, factory = _gym([])
    data, phase = gym.interact()

    assert phase == "test"
    assert data.level == 1
    assert data.split == "test"
    assert gym.state.level == 1
    assert gym.state.phase == "test"
    assert gym.state.train_calls == 0
    assert gym.state.n_exposed == 0
    assert gym.state.eval_submissions == 0
    assert gym.state.entry_eval_pending is True
    assert gym.state.charged_eval_trials == 0
    assert gym.state.score == 0.0
    assert gym.state.level_best_score == 0.0
    assert (
        gym.state.max_eval_submissions_per_level
        == OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    )
    assert gym.state.awaiting_submit is True
    assert factory.calls == [(Path("bench-root"), "single_a", 1)]


def test_babytheorist_gym_passes_generation_config_to_default_env(monkeypatch):
    import env.curriculum as curriculum

    calls = []

    class FakeEnv:
        def __init__(self, data_dir, levels, mode, generation):
            calls.append((Path(data_dir), tuple(levels), mode, generation))

        def eval(self, level: int):
            return StubDataset(level, 1, "test")

    monkeypatch.setattr(curriculum, "BabyTheoristEnv", FakeEnv)
    cfg = GenerationConfig(train_n=1, eval_n=1)

    gym = BabyTheoristGym(mode="multi-A", data_dir="bench-root", generation=cfg)
    data, phase = gym.interact()

    assert phase == "test"
    assert data.level == 1
    assert calls == [(Path("bench-root"), (1,), "multi_a", cfg)]


def test_submit_pass_advances_immediately_to_next_level_test():
    gym, _ = _gym([1.0])
    gym.interact()
    report = gym.submit([])

    assert isinstance(report, GymReport)
    assert report.level == 1
    assert report.passed is True
    assert report.entry_eval is True
    assert report.eval_submissions == 0
    assert report.n_exposed == 0
    assert report.score == 100.0
    assert report.level_best_score == 100.0
    assert report.cumulative_score == 100.0
    assert report.charged_eval_trials == 0
    assert report.efficiency == pytest.approx(800.0 / OFFICIAL_MAX_SCORE_AUC)
    assert gym.state.level == 2
    assert gym.state.phase == "test"
    assert gym.state.eval_submissions == 0

    data, phase = gym.interact()
    assert phase == "test"
    assert data.level == 2


def test_submit_passes_at_eighty_percent_threshold():
    gym, _ = _gym([0.80])
    gym.interact()
    report = gym.submit([])

    assert report.passed is True
    assert report.accuracy == 0.80
    assert gym.state.level == 2


def test_passing_level_eight_completes_session():
    gym, factory = _gym([1.0] * OFFICIAL_LEVEL_COUNT)

    while gym.active:
        _, phase = gym.interact()
        assert phase == "test"
        gym.submit([])

    result = gym.result()
    assert result.active is False
    assert result.failed_level is None
    assert result.highest_cleared_level == OFFICIAL_LEVEL_COUNT
    assert result.score == pytest.approx(800.0)
    assert result.score_auc == pytest.approx(OFFICIAL_MAX_SCORE_AUC)
    assert result.efficiency == pytest.approx(1.0)
    assert len(result.history) == OFFICIAL_LEVEL_COUNT
    assert [level for _, _, level in factory.calls] == list(
        range(1, OFFICIAL_LEVEL_COUNT + 1)
    )


def test_zero_train_fail_enters_train_phase():
    gym, _ = _gym([0.0])
    gym.interact()
    report = gym.submit([])

    assert report.passed is False
    assert report.entry_eval is True
    assert report.eval_submissions == 0
    assert gym.active
    assert gym.state.level == 1
    assert gym.state.phase == "train"
    assert gym.state.train_calls == 0
    assert gym.state.n_exposed == 0
    assert gym.state.eval_submissions == 0


def test_train_interact_returns_full_split_then_same_level_test():
    gym, _ = _gym([0.0])
    gym.interact()
    gym.submit([])

    data, phase = gym.interact()
    assert phase == "train"
    assert data.level == 1
    assert data.split == "train"
    assert len(data) == OFFICIAL_TRAIN_SIZE
    assert gym.state.phase == "test"
    assert gym.state.train_calls == 1
    assert gym.state.n_exposed == OFFICIAL_TRAIN_SIZE
    assert gym.state.eval_submissions == 0

    data, phase = gym.interact()
    assert phase == "test"
    assert data.level == 1
    assert gym.state.entry_eval_pending is False


def test_train_interact_uses_available_local_train_size_when_smaller():
    gym, _ = _gym([0.0], train_size=100)
    gym.interact()
    gym.submit([])

    data, phase = gym.interact()

    assert phase == "train"
    assert data.level == 1
    assert data.split == "train"
    assert len(data) == 100
    assert gym.state.n_exposed == 100


def test_repeated_failures_expose_official_train_split_once_until_eval_limit():
    gym, _ = _gym([0.0] * (OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL + 1))

    seen = []
    tests = 0
    while gym.active:
        data, phase = gym.interact()
        if phase == "train":
            seen.append(len(data))
        else:
            tests += 1
            report = gym.submit([])

    assert seen == [OFFICIAL_TRAIN_SIZE]
    assert tests == OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL + 1
    assert report.eval_submissions == OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    assert report.entry_eval is False
    assert gym.result().failed_level == 1
    assert gym.result().highest_cleared_level == 0
    assert (
        gym.result().max_eval_submissions_per_level
        == OFFICIAL_MAX_EVAL_SUBMISSIONS_PER_LEVEL
    )
    assert gym.state.active is False
    assert gym.state.level is None
    assert gym.state.phase is None


def test_pass_after_training_records_exposure_and_advances():
    gym, _ = _gym([0.0, 1.0])
    gym.interact(); gym.submit([])  # zero-train fail
    train, phase = gym.interact()
    assert phase == "train" and len(train) == OFFICIAL_TRAIN_SIZE
    gym.interact()
    report = gym.submit([])         # 250k pass

    assert report.passed
    assert report.n_exposed == OFFICIAL_TRAIN_SIZE
    assert report.train_calls == 1
    assert report.entry_eval is False
    assert report.eval_submissions == 1
    assert report.charged_eval_trials == 1
    assert gym.state.level == 2
    assert gym.state.eval_submissions == 0
    assert gym.result().highest_cleared_level == 1


def test_failed_eval_after_training_retries_until_custom_limit():
    gym, _ = _gym([0.0, 0.0, 1.0], max_eval_submissions_per_level=3)

    gym.interact()
    gym.submit([])  # zero-train fail
    gym.interact()  # train once
    gym.interact()
    report = gym.submit([])

    assert report.passed is False
    assert report.entry_eval is False
    assert report.eval_submissions == 1
    assert gym.active
    assert gym.state.phase == "test"
    assert gym.state.train_calls == 1
    assert gym.state.n_exposed == OFFICIAL_TRAIN_SIZE

    data, phase = gym.interact()
    assert phase == "test"
    assert data.level == 1
    report = gym.submit([])

    assert report.passed is True
    assert report.eval_submissions == 2
    assert gym.state.level == 2


def test_interact_while_test_is_pending_raises():
    gym, _ = _gym([])
    gym.interact()
    with pytest.raises(RuntimeError, match="submit"):
        gym.interact()


def test_submit_without_pending_test_raises():
    gym, _ = _gym([])
    with pytest.raises(RuntimeError, match="pending test"):
        gym.submit([])


def test_submit_after_train_before_test_raises():
    gym, _ = _gym([0.0])
    gym.interact()
    gym.submit([])
    gym.interact()  # train, switches to test but no test payload served yet
    with pytest.raises(RuntimeError, match="pending test"):
        gym.submit([])


def test_eval_error_does_not_enter_pending_submit_state():
    class BadEvalEnv(StubEnv):
        def eval(self, level: int):
            raise FileNotFoundError("missing eval")

    def factory(data_dir: Path, mode: str, level: int):
        return BadEvalEnv(
            level, ScriptedFactory([]), OFFICIAL_TRAIN_SIZE, 100,
        )

    gym = BabyTheoristGym(mode="single-A", _env_factory=factory)
    with pytest.raises(FileNotFoundError, match="missing eval"):
        gym.interact()

    assert gym.state.awaiting_submit is False
    assert gym.state.phase == "test"


def test_train_error_does_not_consume_exposure():
    class BadTrainEnv(StubEnv):
        def train(self, level: int, n: int | None = None):
            raise ValueError("train too small")

    def factory(data_dir: Path, mode: str, level: int):
        return BadTrainEnv(
            level, ScriptedFactory([0.0]), OFFICIAL_TRAIN_SIZE, 100,
        )

    gym = BabyTheoristGym(mode="single-A", _env_factory=factory)
    gym.interact()
    gym.submit([])

    assert gym.state.phase == "train"
    assert gym.state.train_calls == 0
    assert gym.state.n_exposed == 0
    with pytest.raises(ValueError, match="train too small"):
        gym.interact()

    assert gym.state.phase == "train"
    assert gym.state.train_calls == 0
    assert gym.state.n_exposed == 0


def test_history_and_result_fields():
    gym, _ = _gym([0.0, 1.0], data_dir="/tmp/babytheorist-data")
    gym.interact(); r1 = gym.submit([])
    gym.interact()
    gym.interact(); r2 = gym.submit([])

    result = gym.result()
    assert result.mode == "single_a"
    assert result.data_dir == Path("/tmp/babytheorist-data")
    assert result.active is True
    assert result.failed_level is None
    assert result.highest_cleared_level == 1
    assert result.history == (r1, r2)
    assert result.score == 100.0
    assert result.charged_eval_trials == 1
    assert result.level_scores == (LevelScore(level=1, score=100.0),)
    assert result.score_curve == (
        GymScorePoint(level=1, score=100.0),
        GymScorePoint(level=2, score=100.0),
        GymScorePoint(level=3, score=100.0),
        GymScorePoint(level=4, score=100.0),
        GymScorePoint(level=5, score=100.0),
        GymScorePoint(level=6, score=100.0),
        GymScorePoint(level=7, score=100.0),
        GymScorePoint(level=8, score=100.0),
    )
    assert len(result.score_curve) == OFFICIAL_LEVEL_COUNT
    assert result.score_auc == pytest.approx(800.0)
    assert result.efficiency == pytest.approx(800.0 / OFFICIAL_MAX_SCORE_AUC)


def test_benchmark_score_uses_best_score_per_level_and_free_entry_eval():
    gym, _ = _gym(
        [0.70, 0.75, 0.96, 0.40, 0.60, 0.70, 0.70, 0.72, 0.75],
        max_eval_submissions_per_level=5,
    )

    while gym.active:
        _, phase = gym.interact()
        if phase == "test":
            gym.submit([])

    result = gym.result()
    assert result.highest_cleared_level == 1
    assert result.failed_level == 2
    assert result.score == pytest.approx(171.0)
    assert result.level_scores == (
        LevelScore(level=1, score=96.0),
        LevelScore(level=2, score=75.0),
    )
    assert result.charged_eval_trials == 7
    assert result.history[0].entry_eval is True
    assert result.history[0].eval_submissions == 0
    assert result.history[2].passed is True
    assert result.history[2].eval_submissions == 2
    assert result.history[3].entry_eval is True
    assert result.history[3].eval_submissions == 0
    assert result.history[-1].eval_submissions == 5
    assert result.score_curve == (
        GymScorePoint(level=1, score=96.0),
        GymScorePoint(level=2, score=171.0),
        GymScorePoint(level=3, score=171.0),
        GymScorePoint(level=4, score=171.0),
        GymScorePoint(level=5, score=171.0),
        GymScorePoint(level=6, score=171.0),
        GymScorePoint(level=7, score=171.0),
        GymScorePoint(level=8, score=171.0),
    )
    assert result.score_auc == pytest.approx(1293.0)
    assert result.efficiency == pytest.approx(1293.0 / OFFICIAL_MAX_SCORE_AUC)


def test_reset_restores_initial_state_but_keeps_env_cache():
    gym, factory = _gym([1.0])
    gym.interact()
    gym.submit([])
    assert gym.state.level == 2

    gym.reset()
    assert gym.active
    assert gym.state.level == 1
    assert gym.state.phase == "test"
    assert gym.state.n_exposed == 0
    assert gym.result().history == ()
    assert gym.result().score == 0.0
    assert gym.result().charged_eval_trials == 0
    assert factory.calls == [(Path("bench-root"), "single_a", 1)]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="mode"):
        BabyTheoristGym(mode="bogus")


def test_mode_alias_normalized_before_factory():
    factory = ScriptedFactory([1.0])
    gym = BabyTheoristGym(mode="single-A", data_dir="bench-root", _env_factory=factory)

    assert gym.mode == "single_a"
    gym.interact()
    assert factory.calls == [(Path("bench-root"), "single_a", 1)]


def test_mode_alias_dir_label_passed_to_default_env_factory(monkeypatch):
    import env.curriculum as curriculum

    calls = []

    class FakeEnv:
        def __init__(self, data_dir, levels, mode, generation):
            calls.append((Path(data_dir), tuple(levels), mode, generation))

        def eval(self, level: int):
            return StubDataset(level, 1, "test")

    monkeypatch.setattr(curriculum, "BabyTheoristEnv", FakeEnv)

    gym = BabyTheoristGym(mode="single-A", data_dir="bench-root")
    data, phase = gym.interact()

    assert gym.mode == "single_a"
    assert phase == "test"
    assert data.level == 1
    assert calls == [(Path("bench-root"), (1,), "single_a", None)]


def test_invalid_max_eval_submissions_rejected():
    with pytest.raises(ValueError, match="max_eval_submissions_per_level"):
        BabyTheoristGym(mode="single-A", max_eval_submissions_per_level=0)
