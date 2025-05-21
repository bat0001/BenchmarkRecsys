from __future__ import annotations
import numpy as np, pandas as pd, torch
from typing import Dict, List
from baselines.core.baseline import BaseBaseline
from utils.device import DEVICE           

class LinUCBBaseline(BaseBaseline):
    """
    Contextual‑bandit baseline (Li et al. 2010).

    • context  x_i ∈ ℝ^d   (here : item features)
    • per‑arm A_i = d×d gram matrix,  b_i = d vector
    """

    def _build_model(self):
        return None

    def offline_fit(self,
                    df: pd.DataFrame,
                    ctx_cols: List[str] | None = None,
                    item_col: str = "productId",
                    reward_col: str = "reward"):
        """
        Pre‑compute one fixed context vector per item.

        Parameters
        ----------
        ctx_cols    : columns used as numerical context.
                      If None → one‑hot on item_id (⇒ reduces to classic UCB)
        """
        self.item_col   = item_col
        self.reward_col = reward_col

        if ctx_cols is None:                     
            items = df[item_col].unique()
            eye   = np.eye(len(items))
            self.context = {it: eye[k]
                            for k, it in enumerate(items)}
        else:
            ctx = (df
                   .drop_duplicates(item_col)
                   .set_index(item_col)[ctx_cols])

            ctx = (ctx - ctx.mean()) / ctx.std().replace(0, 1)
            self.context = {it: row.to_numpy(float)
                            for it, row in ctx.iterrows()}

        self.d = next(iter(self.context.values())).shape[0]
        self.items = list(self.context.keys())
        self.n_items = len(self.items)

        self.A  = {it: np.eye(self.d)           for it in self.items}
        self.b  = {it: np.zeros(self.d)         for it in self.items}

        self.bank = {it: df.loc[df[item_col] == it, reward_col]
                          .to_numpy(np.int8)
                     for it in self.items}
        return self

    def _ucb_score(self, it: str, alpha: float):
        A_inv = np.linalg.inv(self.A[it])
        theta = A_inv @ self.b[it]
        x     = self.context[it]
        return theta @ x + alpha * np.sqrt(x @ A_inv @ x)

    def online_simulate(self,
                        n_visits: int,
                        alpha: float | None = None,
                        return_raw: bool = True):
        """
        Simulate LinUCB for `n_visits` steps on historical log.
        """
        if alpha is None:
            alpha = getattr(self.cfg, "linucb_alpha", 1.0)

        rng, results, cum = np.random.default_rng(self.cfg.seed), [], 0.0

        for v in range(n_visits):
            scores = [self._ucb_score(it, alpha) for it in self.items]
            it     = self.items[int(np.argmax(scores))]

            r = int(rng.choice(self.bank[it]))

            x = self.context[it]
            self.A[it] += np.outer(x, x)
            self.b[it] += r * x

            cum += r
            results.append(dict(
                visit=v,
                item_id=it,
                reward=r,
                fraction_relevant=cum / (v + 1)
            ))

        metrics = {"CTR": cum / n_visits}
        return (metrics, results) if return_raw else metrics