"""Benchmark BabyTheorist HDF5 loading paths.

This script compares the path a model sees through ``BabyTheoristEnv`` with direct
HDF5 slab reads. It is intentionally dependency-light so it can be run in the
same environment used for data generation.

Examples:
    python scripts/bench_data_loader.py --data-dir data --mode single-A --level 1
    python scripts/bench_data_loader.py --h5 data/single_a-v0.5.5/lv1_train.h5 --n 50000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from babytheorist.version import default_dataset_dir_name
from env import BabyTheoristEnv, benchmark_mode_dir_label


def _time(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    print(f"{label:32s} {dt:8.3f}s", flush=True)
    return result, dt


def _resolve_h5(args) -> Path:
    if args.h5 is not None:
        return Path(args.h5)
    root = Path(args.data_dir)
    mode_dir_label = benchmark_mode_dir_label(args.mode)
    versioned_name = default_dataset_dir_name(mode_dir_label)
    if root.name not in {mode_dir_label, versioned_name}:
        root = root / versioned_name
    return root / f"lv{args.level}_train.h5"


def _bench_env_iteration(
    data_dir: Path,
    mode: str | None,
    level: int,
    n: int,
    batch_size: int,
) -> None:
    env = BabyTheoristEnv(data_dir=data_dir, levels=(level,), mode=mode, generate_missing=False)
    ds = env.train(level=level, n=n)
    print(f"train_data.path={ds.path}", flush=True)

    def iterate():
        checksum = 0
        for ex in ds:
            checksum += int(ex.x[0, 0]) + int(ex.y[0, 0])
        return checksum

    _, dt = _time("BabyTheoristEnv row iteration", iterate)
    print(f"{'':32s} {dt / n * 1e6:8.1f} us/row", flush=True)

    def materialize():
        xs = []
        ys = []
        for ex in ds:
            xs.append(ex.x)
            ys.append(ex.y)
        return np.stack(xs), np.stack(ys)

    (x, y), dt = _time("BabyTheoristEnv np.stack materialize", materialize)
    print(f"{'':32s} x={x.shape} y={y.shape} {dt / n * 1e6:.1f} us/row", flush=True)

    def official_bulk():
        return ds.as_arrays()

    (x, y), dt = _time("BabyTheoristEnv as_arrays", official_bulk)
    print(f"{'':32s} x={x.shape} y={y.shape} {dt / n * 1e6:.1f} us/row", flush=True)

    def official_batches():
        checksum = 0
        n_seen = 0
        for batch in ds.iter_batches(batch_size):
            checksum += int(batch.x[:, 0, 0].sum()) + int(batch.y[:, 0, 0].sum())
            n_seen += len(batch.task_id)
        return checksum, n_seen

    (_, n_seen), dt = _time(f"BabyTheoristEnv iter_batches b={batch_size}", official_batches)
    print(f"{'':32s} n={n_seen} {dt / n_seen * 1e6:.1f} us/row", flush=True)


def _bench_direct_h5(path: Path, n: int, batch_size: int, random_rows: int) -> None:
    print(f"path={path}", flush=True)
    with h5py.File(path, "r") as h:
        n = min(n, int(h["inputs"].shape[0]))
        for name in ("inputs", "outputs"):
            ds = h[name]
            print(
                f"{name}: shape={ds.shape} dtype={ds.dtype} "
                f"chunks={ds.chunks} compression={ds.compression}",
                flush=True,
            )

    def bulk():
        with h5py.File(path, "r") as h:
            return h["inputs"][:n, 0], h["outputs"][:n, 0]

    (x, y), dt = _time("direct HDF5 bulk read", bulk)
    print(f"{'':32s} x={x.shape} y={y.shape} {dt / n * 1e6:.1f} us/row", flush=True)

    def batches():
        checksum = 0
        n_seen = 0
        with h5py.File(path, "r") as h:
            for start in range(0, n, batch_size):
                stop = min(start + batch_size, n)
                x = h["inputs"][start:stop, 0]
                y = h["outputs"][start:stop, 0]
                checksum += int(x[:, 0, 0].sum()) + int(y[:, 0, 0].sum())
                n_seen += stop - start
        return checksum, n_seen

    (_, n_seen), dt = _time(f"direct HDF5 batches b={batch_size}", batches)
    print(f"{'':32s} n={n_seen} {dt / n_seen * 1e6:.1f} us/row", flush=True)

    def direct_row_sequential():
        checksum = 0
        with h5py.File(path, "r") as h:
            inputs = h["inputs"]
            outputs = h["outputs"]
            for i in range(n):
                x = inputs[i, 0]
                y = outputs[i, 0]
                checksum += int(x[0, 0]) + int(y[0, 0])
        return checksum

    _, dt = _time("direct HDF5 row sequential", direct_row_sequential)
    print(f"{'':32s} {dt / n * 1e6:.1f} us/row", flush=True)

    m = min(random_rows, n)
    idxs = list(range(n))
    random.Random(0).shuffle(idxs)
    idxs = idxs[:m]

    def direct_row_random():
        checksum = 0
        with h5py.File(path, "r") as h:
            inputs = h["inputs"]
            outputs = h["outputs"]
            for i in idxs:
                x = inputs[i, 0]
                y = outputs[i, 0]
                checksum += int(x[0, 0]) + int(y[0, 0])
        return checksum

    _, dt = _time(f"direct HDF5 row random n={m}", direct_row_random)
    print(f"{'':32s} {dt / m * 1e6:.1f} us/row", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--mode", default="single-A")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--h5", type=Path, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--random-rows", type=int, default=5000)
    parser.add_argument(
        "--skip-env",
        action="store_true",
        help="Only benchmark direct HDF5 reads.",
    )
    args = parser.parse_args()

    h5_path = _resolve_h5(args)
    with h5py.File(h5_path, "r") as h:
        total = int(h["inputs"].shape[0])
    n = total if args.n is None else min(args.n, total)

    print(f"benchmarking n={n}", flush=True)
    if not args.skip_env:
        data_dir = Path(args.data_dir)
        _bench_env_iteration(data_dir, args.mode, args.level, n, args.batch_size)
    _bench_direct_h5(h5_path, n, args.batch_size, args.random_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
