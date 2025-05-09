# baselines/epsilon/baseline.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from baselines.core.baseline import BaseBaseline
from baselines.epsilon.replayer import EpsilonGreedyReplayer


class EpsilonGreedyBaseline(BaseBaseline):
    """ε‑greedy baseline (no parametric model, just the replayer)."""

    def _build_model(self):
        return None

    def offline_fit(self, data):
        self.df = data
        Path(self.cfg.out_dir).mkdir(parents=True, exist_ok=True)
        return self

    def online_simulate(self,
                        n_visits: int,
                        *,
                        return_raw: bool = False):
        sim = EpsilonGreedyReplayer(
            n_visits=n_visits,
            reward_history=self.df,
            item_col_name="productId",
            visitor_col_name="userId",
            reward_col_name="rating",
            epsilon=getattr(self.cfg, "epsilon", 0.1),
            like_threshold=getattr(self.cfg, "ts_like_threshold", 4.0),
            n_iterations=1,
        )
        raw = sim.simulator()

        last_ctr = [r["fraction_relevant"]
                    for r in raw if r["visit"] == n_visits - 1]
        metrics = {"Reward Mean": float(np.mean(last_ctr))}

        if return_raw:
            return metrics, raw
        return metrics