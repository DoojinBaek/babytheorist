"""Minimal BabyTheoristGym interaction example.

Replace `Model` with your own learner, then run:

    python quick_start.py --mode single_easy --data data
"""
from __future__ import annotations

import argparse

import numpy as np

from babytheorist.benchmark import BENCHMARK_MODES
from env import BabyTheoristGym


class Model:
    def fit(self, train_data) -> None:
        """Consume the current level's independent (x, y) observations."""
        # TODO: update model state from train_data.

    def predict(self, eval_data):
        """Return (task_id, predicted_grid) pairs for eval examples."""
        preds = []
        for ex in eval_data:
            preds.append((ex.task_id, self.predict_one(ex)))
        return preds

    def predict_one(self, ex) -> np.ndarray:
        # TODO: infer a theory from ex.support_x/ex.support_y and apply it to
        # ex.query_x. Identity is only a placeholder.
        return ex.query_x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=BENCHMARK_MODES,
        default="single_easy",
    )
    ap.add_argument("--data", default="data",
                    help="benchmark data root; BabyTheoristGym reads/creates <data>/<mode>")
    args = ap.parse_args()

    model = Model()
    gym = BabyTheoristGym(mode=args.mode, data_dir=args.data)
    while gym.active:
        data, phase = gym.interact()
        if phase == "train":
            model.fit(data)
        elif phase == "test":
            report = gym.submit(model.predict(data))
            print(
                f"lv={report.level} submit={report.eval_submissions} "
                f"n={report.n_exposed} "
                f"acc={report.accuracy:.3f} "
                f"score={report.cumulative_score:.1f} "
                f"entry={report.entry_eval} passed={report.passed}"
            )

    result = gym.result()
    print(
        f"final score={result.score:.1f} "
        f"efficiency={result.efficiency:.3f} "
        f"charged_trials={result.charged_eval_trials}"
    )


if __name__ == "__main__":
    main()
