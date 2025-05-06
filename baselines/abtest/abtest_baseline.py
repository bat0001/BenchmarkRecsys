import numpy as np
from baselines.core.baseline import BaseBaseline
from baselines.abtest.replayer import ABTestReplayer

class ABTestBaseline(BaseBaseline):
    """A/B‑test : phase offline empty, phase online = simulateur."""

    def _build_model(self):          
        return None

    def offline_fit(self, data):

        self.replayer = ABTestReplayer(
            n_visits=self.cfg.num_iterations,
            n_test_visits=self.cfg.abtest_n_test,
            reward_history=data,
            item_col_name="productId",
            visitor_col_name="userId",
            reward_col_name="rating",
            n_iterations=1
        )
        return self

    def online_simulate(self, *_):
        res = self.replayer.simulator()
        rewards = [r["reward"] for r in res if r["visit"] == self.cfg.num_iterations - 1]
        return {"Reward Mean": float(np.mean(rewards))}