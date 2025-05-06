import numpy as np
from baselines.core.baseline import BaseBaseline
from baselines.abtest.replayer import ABTestReplayer

class ABTestBaseline(BaseBaseline):
    """Baseline wrapper around :class:`ABTestReplayer`. Offline phase is just
    instantiation; evaluation is done via the simulator.
    """

    def _build_model(self):
        return None

    def offline_fit(self, data):
        """`data` is expected to be the full ratings DataFrame."""
        self.replayer = ABTestReplayer(
            n_visits=self.cfg.num_iterations,
            n_test_visits=self.cfg.abtest_n_test,
            reward_history=data,
            item_col_name="productId",
            visitor_col_name="userId",
            reward_col_name="rating",
            n_iterations=1,
        )
        return self

    def online_simulate(self, n_visits: int, *, return_raw: bool = False):
        raw = self.replayer.simulator() 
        if return_raw:
            return raw

        final_vals = [
            r.get("fraction_relevant", r.get("fraction"))
            for r in raw if r["visit"] == n_visits - 1
        ]
        metrics = {"Reward Mean": float(np.mean(final_vals))}
        return metrics, raw