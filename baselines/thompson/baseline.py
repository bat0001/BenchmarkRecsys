import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple

from baselines.core.baseline import BaseBaseline


class ThompsonBaseline(BaseBaseline):
    """
    Pure‑numpy Thompson‑Sampling replay (offline).
    Keeps everything in CPU numpy → ultra‑light & portable.
    """

    def _build_model(self):
        return None

    def offline_fit(
        self,
        df: pd.DataFrame,
        *,
        item_col: str = "productId",
        reward_col: str = "reward",
        min_pos: int = 3,
        max_items: int = 1000,
    ):
        """
        Pre‑compute the ‘bank’ of historical rewards for each item.
        Filtering happens here (≥ min_pos clicks & ≤ max_items items).
        """
        pos = df[df[reward_col] == 1][item_col].value_counts()
        keep = pos[pos >= min_pos].index[:max_items]
        df   = df[df[item_col].isin(keep)].copy()

        self.item_col   = item_col
        self.reward_col = reward_col

        self.items      = df[item_col].unique()
        self.n_items    = len(self.items)

        self.reward_bank: list[np.ndarray] = [
            df.loc[df[item_col] == it, reward_col].to_numpy(np.int8)
            for it in self.items
        ]
        return self  

    def online_simulate(
        self,
        n_visits: int = 100,
        *,
        n_iterations: int | None = None,
        # return_raw: bool = False,
    ) -> Tuple[Dict[str, float], List[dict]] :
        """
        Run the replay loop and return either:
          • a metrics dict   (default)
          • (metrics, raw_records)   if return_raw=True
        """
        n_iter = n_iterations or self.cfg.num_iterations
        
        rng    = np.random.default_rng(self.cfg.seed)

        results: List[Dict] = []
        final_ctrs: list[float] = []

        for it in tqdm(range(n_iter), desc="[TS] run"):
            alpha = np.ones(self.n_items, dtype=np.float64)
            beta  = np.ones_like(alpha)

            cum = 0.0
            for v in range(n_visits):
                theta = rng.beta(alpha, beta)
                idx   = int(np.argmax(theta))

                rwd   = int(rng.choice(self.reward_bank[idx]))
                if rwd == 1:
                    alpha[idx] += 1.0
                else:
                    beta[idx]  += 1.0

                cum += rwd
                results.append({
                    "iteration": it,
                    "visit":     v,
                    "item_idx":  idx,
                    "reward":    rwd,
                    "fraction_relevant": cum / (v + 1),
                })

            final_ctrs.append(cum / n_visits)

        metrics = {"Reward Mean": float(np.mean(final_ctrs))}
        return metrics, results