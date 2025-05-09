from __future__ import annotations
import pandas as pd
import numpy as np

from baselines.core.baseline import BaseBaseline
from baselines.ucb.replayer import _UCBReplayer



class UCBBaseline(BaseBaseline):
    """UCB‑1 replay baseline."""

    def _build_model(self):         
        return None

    def offline_fit(self, ratings_df: pd.DataFrame):
        self.df = ratings_df
        return self

    def online_simulate(self, n_iters: int):
        rep = _UCBReplayer(
            self.df,
            item_col   = "productId",
            reward_col = "reward",
            n_visits   = n_iters,
            c          = getattr(self.cfg, "ucb_c", 2.0),
            n_iter     = 100,
            seed       = self.cfg.seed,
        )
        raw = rep.run()

        final_ctr = np.mean([r["fraction_relevant"]
                             for r in raw if r["visit"] == n_iters - 1])
        return {"Reward Mean": float(final_ctr)}, raw