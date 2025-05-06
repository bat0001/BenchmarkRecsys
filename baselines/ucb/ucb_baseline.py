import numpy as np
from baselines.core.baseline import BaseBaseline
from baselines.ucb.replayer import UCBSamplingReplayer

class UCBBaseline(BaseBaseline):
    """UCB : no neede for offline phasee."""

    def _build_model(self):
        return None

    def offline_fit(self, data):
        self.df = data
        return self

    def online_simulate(self, *_):
        sim = UCBSamplingReplayer(
            ucb_c=self.cfg.ucb_c,
            n_visits=self.cfg.num_iterations,
            reward_history=self.df,
            item_col_name="productId",
            visitor_col_name="userId",
            reward_col_name="rating",
            n_iterations=1
        )
        res = sim.simulator()
        rewards = [r["reward"] for r in res if r["visit"] == self.cfg.num_iterations - 1]
        return {"Reward Mean": float(np.mean(rewards))}
