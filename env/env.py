"""Stateless curriculum benchmark kit for BabyTheorist.

See `env_plan.md` for design rationale. API surface:

    env = BabyTheoristEnv(data_dir="data/", levels=(1, 2, 3))
    env.train(level=k, n=?) -> Dataset[TrainExample]
    env.eval(level=k)       -> Dataset[EvalExample]
    env.score_predictions(preds, level=k) -> ScoreReport   # Sequence[(tid, grid)]
    env.score(model_fn, level=k)          -> ScoreReport   # thin wrapper
    env.metadata(split, level, task_id)   -> Metadata       # train only
    env.answers(level, task_id)           -> Answers        # eval only
"""

from __future__ import annotations

import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, NamedTuple, Sequence

import h5py
import numpy as np

from env import io_h5
from shape_profiles import (
    SHAPE_PROFILE_STANDARD as _SHAPE_PROFILE_STANDARD,
    SHAPE_PROFILES as _SHAPE_PROFILES,
    shape_profile_defaults as _shape_profile_defaults,
)


Grid = np.ndarray
GridBatch = np.ndarray
ShapeProfile = Literal["standard"]
BenchmarkMode = Literal[
    "multi_dev", "single_a", "multi_a", "single_b", "multi_b",
]


_BENCHMARK_MODES = ("multi_dev", "single_a", "multi_a", "single_b", "multi_b")
_BENCHMARK_MODE_ALIASES = {
    "singlea": "single_a",
    "single_easy": "single_a",
    "multia": "multi_a",
    "multi_easy": "multi_a",
    "singleb": "single_b",
    "single_hard": "single_b",
    "multib": "multi_b",
    "multi_hard": "multi_b",
}
_BENCHMARK_MODE_ALIAS_DIR_LABELS = {
    "single_easy": "single_a",
    "single_a": "single_a",
    "singlea": "single_a",
    "multi_easy": "multi_a",
    "multi_a": "multi_a",
    "multia": "multi_a",
    "single_hard": "single_b",
    "single_b": "single_b",
    "singleb": "single_b",
    "multi_hard": "multi_b",
    "multi_b": "multi_b",
    "multib": "multi_b",
}
_ATTEMPT_SCHEDULERS = ("ordered", "unordered")
_SPLITS = ("train", "eval")
_DEPTH_PRIOR_PIECEWISE = "piecewise_geometric"
_GENERATION_LOCK_STALE_GRACE_SEC = 5.0


def normalize_benchmark_mode(mode: str) -> BenchmarkMode:
    """Return the public canonical benchmark mode for aliases.

    The A/B labels are the public canonical names. Descriptive names such as
    ``single_easy`` remain legacy aliases for compatibility.
    """
    key = mode.strip().replace("-", "_").lower()
    canonical = {m: m for m in _BENCHMARK_MODES}
    canonical.update(_BENCHMARK_MODE_ALIASES)
    if key not in canonical:
        aliases = ", ".join(sorted(_BENCHMARK_MODE_ALIASES))
        raise ValueError(
            f"mode must be one of {_BENCHMARK_MODES} "
            f"(aliases: {aliases}), got {mode!r}"
        )
    return canonical[key]


def benchmark_mode_dir_label(mode: str) -> str:
    """Return the on-disk suite directory label for a mode or alias."""
    key = mode.strip().replace("-", "_").lower()
    if key in _BENCHMARK_MODES:
        return key
    if key in _BENCHMARK_MODE_ALIAS_DIR_LABELS:
        return _BENCHMARK_MODE_ALIAS_DIR_LABELS[key]
    normalize_benchmark_mode(mode)
    raise AssertionError("unreachable")


def _babytheorist_code_version() -> str:
    """Read babytheorist/version.py without importing babytheorist and its scipy dependency."""
    version_path = Path(__file__).resolve().parents[1] / "babytheorist" / "version.py"
    ns: dict[str, object] = {}
    exec(version_path.read_text(), ns)
    return str(ns["__version__"])


def _default_mode_dir_name(mode: BenchmarkMode) -> str:
    return f"{mode}-v{_babytheorist_code_version()}"


def _versioned_mode_dir_name(label: str) -> str:
    return f"{label}-v{_babytheorist_code_version()}"


def _dataset_release_version(mode: BenchmarkMode) -> str:
    return f"BabyTheorist-{mode}-v{_babytheorist_code_version()}"


def _default_dominant_pool_dir(data_dir: Path) -> Path:
    return data_dir.parent / f"dominant_pools-v{_babytheorist_code_version()}"


OFFICIAL_TRAIN_SIZE = 250_000
OFFICIAL_TEST_SIZE = 10_000


@dataclass(frozen=True)
class GenerationConfig:
    """Settings used when BabyTheoristEnv auto-generates missing benchmark files."""
    train_n: int = OFFICIAL_TRAIN_SIZE
    eval_n: int = OFFICIAL_TEST_SIZE
    world_size: int = 15
    n_shapes: int | None = None
    shape_profile: ShapeProfile = _SHAPE_PROFILE_STANDARD
    shape_min: int | None = None
    shape_max: int | None = None
    shape_compactness: float | None = None
    shape_border_min: int | None = None
    shape_max_bbox_span: int | None = None
    min_support: int = 2
    max_support: int = 5
    backend: str = "filter"
    workers: int = 1
    attempt_scheduler: str = "ordered"
    attempt_timeout_sec: float | None = None
    seed: int = 42
    pool_seed: int = 42
    dominant_pool_dir: str | Path | None = None
    initial_hollow_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.n_shapes is not None and self.n_shapes < 1:
            raise ValueError(f"n_shapes must be >= 1 when set, got {self.n_shapes}")
        if self.shape_profile not in _SHAPE_PROFILES:
            raise ValueError(
                f"shape_profile must be one of {_SHAPE_PROFILES}, "
                f"got {self.shape_profile!r}"
            )
        if self.shape_min is not None and self.shape_min < 1:
            raise ValueError(f"shape_min must be >= 1, got {self.shape_min}")
        if self.shape_max is not None and self.shape_max < 1:
            raise ValueError(f"shape_max must be >= 1, got {self.shape_max}")
        if (
            self.shape_min is not None
            and self.shape_max is not None
            and self.shape_max < self.shape_min
        ):
            raise ValueError(
                f"shape_max={self.shape_max} < shape_min={self.shape_min}"
            )
        if self.shape_compactness is not None and self.shape_compactness < 0:
            raise ValueError(
                "shape_compactness must be >= 0, "
                f"got {self.shape_compactness}"
            )
        if self.shape_border_min is not None and self.shape_border_min < 0:
            raise ValueError(
                "shape_border_min must be >= 0, "
                f"got {self.shape_border_min}"
            )
        if (
            self.shape_max_bbox_span is not None
            and self.shape_max_bbox_span < 0
        ):
            raise ValueError(
                "shape_max_bbox_span must be >= 0, "
                f"got {self.shape_max_bbox_span}"
            )
        if self.min_support < 1:
            raise ValueError(
                f"min_support must be >= 1, got {self.min_support}"
            )
        if self.max_support < 1:
            raise ValueError(
                f"max_support must be >= 1, got {self.max_support}"
            )
        if self.min_support > self.max_support:
            raise ValueError(
                f"min_support={self.min_support} > max_support={self.max_support}"
            )
        if self.attempt_scheduler not in _ATTEMPT_SCHEDULERS:
            raise ValueError(
                f"attempt_scheduler must be one of {_ATTEMPT_SCHEDULERS}, "
                f"got {self.attempt_scheduler!r}"
            )
        if self.attempt_timeout_sec is not None and self.attempt_timeout_sec <= 0.0:
            raise ValueError(
                "attempt_timeout_sec must be > 0 when set, "
                f"got {self.attempt_timeout_sec}"
            )


# ---- sample / report types ------------------------------------------------

@dataclass(frozen=True)
class TrainExample:
    x: Grid
    y: Grid
    level: int
    task_id: int  # split-local row index (0..N_train-1)


@dataclass(frozen=True)
class TrainBatch:
    x: GridBatch        # (B, H, W)
    y: GridBatch        # (B, H, W)
    task_id: np.ndarray  # (B,), split-local row indices
    level: int


@dataclass(frozen=True)
class EvalExample:
    support_x: Grid   # (K, H, W) — K support inputs, K ≥ 1
    support_y: Grid   # (K, H, W) — K support outputs
    query_x: Grid     # (H, W)
    level: int
    task_id: int


@dataclass(frozen=True)
class ScoreReport:
    level: int
    n: int
    correct: int
    accuracy: float


@dataclass(frozen=True)
class Answers:
    """Eval-only answer bundle. `Y_q[0] == y_q` (canonicalized in io_h5)."""
    y_q: Grid                          # (H, W)
    Y_q: tuple                         # tuple of (H, W)


@dataclass(frozen=True)
class Metadata:
    """Train-only metadata. Blocked for eval (privileged signal)."""
    program: list                      # list[Step dict]


class _Schema(NamedTuple):
    """File-invariant counts cached at BabyTheoristEnv.__init__."""
    N: int
    P: int


def _known_mode_dir_labels() -> tuple[str, ...]:
    labels = list(_BENCHMARK_MODES)
    for label in (
        *tuple(_BENCHMARK_MODE_ALIAS_DIR_LABELS.values()),
        *tuple(_BENCHMARK_MODE_ALIAS_DIR_LABELS.keys()),
    ):
        if label not in labels:
            labels.append(label)
    return tuple(labels)


def _resolve_data_dir(
    data_dir: str | Path,
    mode: BenchmarkMode | None,
    mode_dir_label: str | None = None,
) -> Path:
    root = Path(data_dir)
    if mode is None:
        return root
    # Accept root mode directories explicitly, but default
    # BabyTheoristEnv("data", mode="single-A") to data/single_a-vX.Y.Z.
    # Legacy aliases such as mode="single_easy" resolve to the same A/B path.
    label = mode_dir_label if mode_dir_label is not None else mode
    versioned_name = _versioned_mode_dir_name(label)
    canonical_versioned_name = _default_mode_dir_name(mode)
    if label == mode:
        accepted_roots = {mode, canonical_versioned_name}
    else:
        accepted_roots = {label, versioned_name}
    if root.name in accepted_roots:
        return root
    known_labels = _known_mode_dir_labels()
    versioned_mode_pattern = re.compile(
        rf"^({'|'.join(re.escape(item) for item in known_labels)})"
        r"-v\d+\.\d+\.\d+"
    )
    if root.name in known_labels or versioned_mode_pattern.match(root.name):
        raise ValueError(
            f"data_dir={str(root)!r} already appears to be a benchmark-mode "
            f"directory, which conflicts with requested mode={mode!r}"
        )
    return root / versioned_name


def _info_path(h5_path: Path) -> Path:
    return h5_path.with_suffix(".info.json")


def _generation_shape_metadata(cfg: GenerationConfig) -> dict[str, object]:
    """Expected `shape_generation` sidecar metadata for auto-generated files."""
    defaults = _shape_profile_defaults(cfg.shape_profile)
    default_min, default_max = defaults["shape_size_range"]
    expected: dict[str, object] = {
        "profile": cfg.shape_profile,
        "shape_size_range": [
            cfg.shape_min if cfg.shape_min is not None else default_min,
            cfg.shape_max if cfg.shape_max is not None else default_max,
        ],
        "shape_compactness": (
            cfg.shape_compactness
            if cfg.shape_compactness is not None
            else defaults["shape_compactness"]
        ),
        "shape_bordered_hollow_min_size": (
            defaults["shape_bordered_hollow_min_size"]
            if cfg.shape_border_min is None
            else (None if cfg.shape_border_min == 0 else cfg.shape_border_min)
        ),
        "shape_bordered_hollow_random_inner_size": (
            defaults["shape_bordered_hollow_random_inner_size"]
        ),
        "shape_max_bbox_span": (
            defaults["shape_max_bbox_span"]
            if cfg.shape_max_bbox_span is None
            else (None if cfg.shape_max_bbox_span == 0 else cfg.shape_max_bbox_span)
        ),
    }
    return expected


def _validate_shape_generation_metadata(
    info_path: Path,
    file_shape: object,
    expected: dict[str, object],
) -> None:
    if file_shape is None:
        raise ValueError(
            f"{info_path}: missing shape_generation metadata; cannot verify "
            f"requested shape_generation={expected!r}"
        )
    if not isinstance(file_shape, dict):
        raise ValueError(f"{info_path}: invalid shape_generation metadata")
    for key, expected_value in expected.items():
        file_value = file_shape.get(key)
        if file_value != expected_value:
            raise ValueError(
                f"{info_path}: shape_generation.{key}={file_value!r} "
                f"does not match requested {expected_value!r}"
            )


def _validate_mode_metadata(
    data_dir: Path,
    levels: Sequence[int],
    mode: BenchmarkMode,
    shape_generation: dict[str, object] | None = None,
) -> None:
    for lv in levels:
        for split in _SPLITS:
            info_path = _info_path(data_dir / f"lv{lv}_{split}.h5")
            if not info_path.exists():
                continue
            try:
                import json
                info = json.loads(info_path.read_text())
            except Exception as e:
                raise ValueError(f"{info_path}: failed to read metadata ({e})") from e
            file_mode = info.get("mode")
            if file_mode is not None:
                try:
                    normalized_file_mode = normalize_benchmark_mode(str(file_mode))
                except ValueError as e:
                    raise ValueError(
                        f"{info_path}: mode={file_mode!r} is not a known "
                        "benchmark mode"
                    ) from e
                if normalized_file_mode != mode:
                    raise ValueError(
                        f"{info_path}: mode={file_mode!r} does not match "
                        f"requested mode={mode!r}"
                    )
            file_depth_prior = info.get("depth_distribution")
            if (
                file_depth_prior is not None
                and file_depth_prior != _DEPTH_PRIOR_PIECEWISE
            ):
                raise ValueError(
                    f"{info_path}: depth_distribution={file_depth_prior!r} "
                    f"does not match official {_DEPTH_PRIOR_PIECEWISE!r}"
                )
            if shape_generation is not None:
                _validate_shape_generation_metadata(
                    info_path, info.get("shape_generation"), shape_generation,
                )


def _generate_missing_datasets(
    data_dir: Path,
    levels: Sequence[int],
    mode: BenchmarkMode,
    cfg: GenerationConfig,
) -> None:
    missing = [
        (lv, split)
        for lv in levels
        for split in _SPLITS
        if not (data_dir / f"lv{lv}_{split}.h5").exists()
    ]
    if not missing:
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    for lv, split in missing:
        target = data_dir / f"lv{lv}_{split}.h5"
        with _generation_lock(target) as should_generate:
            if not should_generate or target.exists():
                continue
            _generate_dataset_file(
                target,
                level=lv,
                split=split,
                mode=mode,
                cfg=cfg,
            )


@contextmanager
def _generation_lock(target: Path):
    lock_path = target.with_name(f"{target.name}.lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if target.exists():
                yield False
                return
            if _generation_lock_is_stale(lock_path):
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            time.sleep(0.1)
            continue
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        break
    try:
        yield True
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _generation_lock_is_stale(lock_path: Path) -> bool:
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    try:
        text = lock_path.read_text().strip()
    except FileNotFoundError:
        return False
    except OSError:
        return time.time() - stat.st_mtime > _GENERATION_LOCK_STALE_GRACE_SEC

    try:
        pid = int(text)
    except ValueError:
        return time.time() - stat.st_mtime > _GENERATION_LOCK_STALE_GRACE_SEC
    if pid < 1:
        return time.time() - stat.st_mtime > _GENERATION_LOCK_STALE_GRACE_SEC
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


def _generate_dataset_file(
    out_path: Path,
    level: int,
    split: str,
    mode: BenchmarkMode,
    cfg: GenerationConfig,
) -> None:
    from babytheorist.benchmark import benchmark_mode_default_n_shapes
    from babytheorist.generation.cli import main as generate_main

    n = cfg.train_n if split == "train" else cfg.eval_n
    seed = cfg.seed if split == "train" else cfg.seed + 1
    argv = [
        "--level", str(level),
        "--n", str(n),
        "--mode", split,
        "--world-size", str(cfg.world_size),
        "--shape-profile", cfg.shape_profile,
        "--min-support", str(cfg.min_support),
        "--max-support", str(cfg.max_support),
        "--backend", cfg.backend,
        "--workers", str(cfg.workers),
        "--attempt-scheduler", cfg.attempt_scheduler,
        "--seed", str(seed),
        "--pool-seed", str(cfg.pool_seed),
        "--initial-hollow-prob", str(cfg.initial_hollow_prob),
        "--dataset-version", _dataset_release_version(mode),
        "--out", str(out_path),
    ]
    if cfg.shape_min is not None:
        argv.extend(["--shape-min", str(cfg.shape_min)])
    if cfg.shape_max is not None:
        argv.extend(["--shape-max", str(cfg.shape_max)])
    if cfg.shape_compactness is not None:
        argv.extend(["--shape-compactness", str(cfg.shape_compactness)])
    if cfg.shape_border_min is not None:
        argv.extend(["--shape-border-min", str(cfg.shape_border_min)])
    if cfg.shape_max_bbox_span is not None:
        argv.extend(["--shape-max-bbox-span", str(cfg.shape_max_bbox_span)])
    if cfg.attempt_timeout_sec is not None:
        argv.extend(["--attempt-timeout-sec", str(cfg.attempt_timeout_sec)])
    dominant_pool_dir = (
        Path(cfg.dominant_pool_dir)
        if cfg.dominant_pool_dir is not None
        else _default_dominant_pool_dir(out_path.parent)
    )
    argv.extend(["--dominant-pool-dir", str(dominant_pool_dir)])
    n_shapes = cfg.n_shapes
    if n_shapes is None:
        n_shapes = benchmark_mode_default_n_shapes(mode)
    argv.extend([
        "--benchmark-mode", mode,
        "--n-shapes", str(n_shapes),
    ])
    rc = generate_main(argv)
    if rc:
        raise RuntimeError(
            f"failed to generate {out_path} "
            f"(level={level}, split={split}, mode={mode}, rc={rc})"
        )


# ---- h5-backed dataset ----------------------------------------------------

class _H5Dataset:
    """Map-style dataset over one (level, split). PID-guarded so fork-based
    multiprocessing (DataLoader, Pool) reopens the file in each worker."""

    def __init__(self, path: Path, level: int, n: int):
        self._path = Path(path)
        self._level = level
        self._n = n
        self._h: h5py.File | None = None
        self._opener_pid: int | None = None
        self._open_count = 0  # test oracle: proves fork reopen actually ran

    def _ensure_open(self):
        pid = os.getpid()
        if self._h is None or self._opener_pid != pid:
            if self._h is not None:
                try:
                    self._h.close()
                except Exception:
                    pass
            self._h = h5py.File(self._path, "r")
            self._opener_pid = pid
            self._open_count += 1

    def __getstate__(self):
        # Drop the open handle; child will reopen on first access.
        return {**self.__dict__,
                "_h": None, "_opener_pid": None, "_open_count": 0}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self) -> int:
        return self._n

    @property
    def path(self) -> Path:
        return self._path

    @property
    def level(self) -> int:
        return self._level

    def __getitem__(self, i: int):
        if not 0 <= i < self._n:
            raise IndexError(f"index {i} out of range [0,{self._n})")
        self._ensure_open()
        return self._make(i)

    def _make(self, i: int):
        raise NotImplementedError


class _TrainDataset(_H5Dataset):
    def _make(self, i: int) -> TrainExample:
        x, y = io_h5.read_train_row(self._h, i)
        return TrainExample(x=x, y=y, level=self._level, task_id=i)

    def _resolve_bounds(
        self, start: int | None = None, stop: int | None = None,
    ) -> tuple[int, int]:
        lo = 0 if start is None else int(start)
        hi = self._n if stop is None else int(stop)
        if lo < 0 or hi < lo or hi > self._n:
            raise ValueError(
                f"invalid slice [{lo}, {hi}) for dataset length {self._n}"
            )
        return lo, hi

    def as_arrays(
        self, start: int | None = None, stop: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return train observations as dense NumPy arrays.

        This is the preferred path when a model wants to materialize the
        current train exposure. It performs two contiguous HDF5 slab reads
        instead of one row read per task.
        """
        lo, hi = self._resolve_bounds(start, stop)
        self._ensure_open()
        H, W = self._h["inputs"].shape[2:]
        shape = (hi - lo, H, W)
        x = io_h5.assert_grid(
            self._h["inputs"][lo:hi, 0],
            f"train rows [{lo}, {hi}): x",
            shape,
        )
        y = io_h5.assert_grid(
            self._h["outputs"][lo:hi, 0],
            f"train rows [{lo}, {hi}): y",
            shape,
        )
        return x, y

    def iter_batches(
        self,
        batch_size: int,
        start: int | None = None,
        stop: int | None = None,
    ):
        """Yield train observations in HDF5 slab-read batches."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        lo, hi = self._resolve_bounds(start, stop)
        self._ensure_open()
        inputs = self._h["inputs"]
        outputs = self._h["outputs"]
        H, W = inputs.shape[2:]
        for batch_start in range(lo, hi, batch_size):
            batch_stop = min(batch_start + batch_size, hi)
            shape = (batch_stop - batch_start, H, W)
            x = io_h5.assert_grid(
                inputs[batch_start:batch_stop, 0],
                f"train rows [{batch_start}, {batch_stop}): x",
                shape,
            )
            y = io_h5.assert_grid(
                outputs[batch_start:batch_stop, 0],
                f"train rows [{batch_start}, {batch_stop}): y",
                shape,
            )
            yield TrainBatch(
                x=x,
                y=y,
                task_id=np.arange(batch_start, batch_stop, dtype=np.int64),
                level=self._level,
            )


class _EvalDataset(_H5Dataset):
    def __init__(self, path: Path, level: int, n: int, P: int):
        super().__init__(path, level, n)
        self._P = P

    def _make(self, i: int) -> EvalExample:
        row = io_h5.read_eval_row(self._h, i, self._P)
        return EvalExample(support_x=row["support_x"], support_y=row["support_y"],
                           query_x=row["query_x"], level=self._level, task_id=i)


# ---- main env -------------------------------------------------------------

class BabyTheoristEnv:
    def __init__(
        self,
        data_dir: str | Path,
        levels: Sequence[int],
        mode: BenchmarkMode | None = None,
        generate_missing: bool | None = None,
        generation: GenerationConfig | None = None,
    ):
        mode_dir_label = None
        if mode is not None:
            mode_dir_label = benchmark_mode_dir_label(mode)
            mode = normalize_benchmark_mode(mode)
        self._mode = mode
        self._data_dir = _resolve_data_dir(data_dir, mode, mode_dir_label)
        self._levels = tuple(levels)
        if generate_missing is None:
            generate_missing = mode is not None
        expected_shape_generation: dict[str, object] | None = None
        if generate_missing:
            if mode is None:
                raise ValueError(
                    "generate_missing=True requires mode to be set (not None)"
            )
            generation_cfg = generation or GenerationConfig()
            expected_shape_generation = _generation_shape_metadata(generation_cfg)
            _generate_missing_datasets(
                self._data_dir, self._levels, mode,
                generation_cfg,
            )
        elif generation is not None:
            raise ValueError("generation config is only used when generate_missing=True")
        if mode is not None:
            _validate_mode_metadata(
                self._data_dir, self._levels, mode, expected_shape_generation,
            )
        # One walk over all files: verify existence, check invariants, cache (N, P).
        self._schema: dict[tuple[int, str], _Schema] = {}
        for lv in self._levels:
            for split in _SPLITS:
                p = self._path(lv, split)
                if not p.exists():
                    raise FileNotFoundError(f"missing dataset: {p}")
                with h5py.File(p, "r") as h:
                    in_shape = tuple(h["inputs"].shape)
                    out_shape = tuple(h["outputs"].shape)
                    if in_shape != out_shape:
                        raise ValueError(
                            f"{p}: inputs.shape={in_shape} != outputs.shape={out_shape}"
                        )
                    N, P = int(in_shape[0]), int(in_shape[1])
                    if split == "train" and P != 1:
                        raise ValueError(f"{p}: train P={P} != 1 (train rows are single (x, y) pairs)")
                    if split == "eval" and P < 2:
                        raise ValueError(f"{p}: eval P={P} < 2 (need K≥1 support + query)")
                    if split == "eval" and "n_support" in h:
                        _validate_n_support(h["n_support"], N, P, p)
                    self._schema[(lv, split)] = _Schema(N=N, P=P)

    # --- path / open helpers ----------------------------------------------

    def _path(self, level: int, split: str) -> Path:
        return self._data_dir / f"lv{level}_{split}.h5"

    @property
    def data_dir(self) -> Path:
        """Resolved directory that this env reads from."""
        return self._data_dir

    @property
    def mode(self) -> BenchmarkMode | None:
        return self._mode

    def _check_level(self, level: int):
        if level not in self._levels:
            raise ValueError(f"level {level} not in env levels={self._levels}")

    def _check_tid(self, level: int, split: str, task_id: int):
        N = self._schema[(level, split)].N
        if not 0 <= task_id < N:
            raise IndexError(f"task_id={task_id} out of [0,{N})")

    @contextmanager
    def _open(self, level: int, split: str):
        with h5py.File(self._path(level, split), "r") as h:
            yield h

    # --- datasets ----------------------------------------------------------

    def train(self, level: int, n: int | None = None) -> _TrainDataset:
        self._check_level(level)
        N = self._schema[(level, "train")].N
        if n is None:
            n = N
        elif n < 0 or n > N:
            raise ValueError(f"n={n} out of [0, {N}] for train level={level}")
        return _TrainDataset(self._path(level, "train"), level, n)

    def train_size(self, level: int) -> int:
        """Total number of train rows at `level`. No file open."""
        self._check_level(level)
        return self._schema[(level, "train")].N

    def eval(self, level: int) -> _EvalDataset:
        self._check_level(level)
        s = self._schema[(level, "eval")]
        return _EvalDataset(self._path(level, "eval"), level, s.N, s.P)

    # --- annotation / answers ---------------------------------------------

    def metadata(self, split: str, level: int, task_id: int) -> Metadata:
        """Return `Metadata(program=...)` for a train row. `split="eval"` is blocked:
        eval program is a privileged signal (can be re-executed to synthesize y_q)."""
        self._check_level(level)
        if split != "train":
            raise ValueError(f"metadata: split must be 'train' (got {split!r}); "
                             "eval program is not exposed, use env.answers() for y_q")
        self._check_tid(level, "train", task_id)
        with self._open(level, "train") as h:
            return Metadata(program=io_h5.read_program(h, task_id))

    def answers(self, level: int, task_id: int) -> Answers:
        """Return `Answers(y_q, Y_q)` for an eval row (eval-only, explicit)."""
        self._check_level(level)
        self._check_tid(level, "eval", task_id)
        P = self._schema[(level, "eval")].P
        with self._open(level, "eval") as h:
            row = io_h5.read_eval_row(h, task_id, P)
        return Answers(y_q=row["y_q"], Y_q=row["Y_q"])

    # --- scoring -----------------------------------------------------------

    def score_predictions(
        self, preds: Sequence[tuple[int, Grid]], level: int,
    ) -> ScoreReport:
        self._check_level(level)
        preds = list(preds)
        N, P = self._schema[(level, "eval")]

        if len(preds) != N:
            raise ValueError(f"preds length={len(preds)}, expected {N}")

        seen: dict[int, int] = {}
        for i, (tid, _) in enumerate(preds):
            if not 0 <= tid < N:
                raise ValueError(f"task_id={tid} out-of-range at preds[{i}]")
            if tid in seen:
                raise ValueError(
                    f"duplicate task_id={tid} at preds[{i}]; first seen at preds[{seen[tid]}]"
                )
            seen[tid] = i

        correct = 0
        with self._open(level, "eval") as h:
            for tid, pred in preds:
                # Fast-path: read only Y_q, not support_x/support_y.
                Y_q = io_h5.read_Y_q(h, tid, P)
                arr = _validate_pred(pred, tid, expected_shape=Y_q[0].shape)
                if any(np.array_equal(arr, g) for g in Y_q):
                    correct += 1

        return ScoreReport(level=level, n=N, correct=correct,
                           accuracy=correct / N if N else 0.0)

    def score(
        self, model_fn: Callable[[Grid, Grid, Grid], Grid], level: int,
    ) -> ScoreReport:
        """Thin wrapper: calls `model_fn(support_x, support_y, query_x)` per row.
        `support_x`, `support_y` have shape (K, H, W); `query_x` is (H, W)."""
        preds = [
            (ex.task_id, model_fn(ex.support_x, ex.support_y, ex.query_x))
            for ex in self.eval(level=level)
        ]
        return self.score_predictions(preds, level=level)


# ---- schema helpers -------------------------------------------------------

def _validate_n_support(ds: "h5py.Dataset", N: int, P: int, path: Path) -> None:
    """File-level check for the optional `n_support` field (eval only).
    Validates shape (N,) and integer dtype; per-row range stays in io_h5."""
    if ds.shape != (N,):
        raise ValueError(f"{path}: n_support.shape={ds.shape}, expected ({N},)")
    if not np.issubdtype(ds.dtype, np.integer):
        raise ValueError(f"{path}: n_support.dtype={ds.dtype}, expected integer")


# ---- prediction validation ------------------------------------------------

def _validate_pred(pred, tid: int, expected_shape: tuple) -> np.ndarray:
    """Coerce to array + validate via io_h5.assert_grid (same rules as h5 reads)."""
    try:
        arr = np.asarray(pred)
    except Exception as e:
        raise TypeError(f"task_id={tid}: cannot coerce to array ({e})") from e
    return io_h5.assert_grid(arr, f"task_id={tid}", expected_shape=expected_shape)
