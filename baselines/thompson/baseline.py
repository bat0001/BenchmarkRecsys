import numpy as np
from baselines.core.baseline import BaseBaseline
from baselines.thompson.replayer import ThompsonSamplingReplayer


class ThompsonBaseline(BaseBaseline):
    """Wrapper baseline for Thompson‑Sampling replayer."""

    def _build_model(self):
        return None

    def offline_fit(self, data):
        self.df = data
        return self

    def online_simulate(self, n_visits: int, *, return_raw: bool = False):
        sim = ThompsonSamplingReplayer(
            n_visits=n_visits,
            reward_history=self.df,
            item_col_name="productId",
            visitor_col_name="userId",
            reward_col_name="rating",
            n_iterations=1,
        )
        raw = sim.simulator()
        if return_raw:
            return raw

        final_vals = [
            r.get("fraction_relevant", r.get("fraction"))
            for r in raw
            if r["visit"] == n_visits - 1
        ]
        metrics = {"Reward Mean": float(np.mean(final_vals))}
        return metrics, raw