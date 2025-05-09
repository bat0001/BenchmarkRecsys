from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm


class EpsilonGreedyReplayer:
    """
    Fast ε‑greedy replayer for very large implicit‑feedback datasets.

    • Pre‑builds a list of (visitor, reward_float) for every item.  
    • Each historic interaction is consumed at most once (`pop`) so the
      distribution gradually shifts from exploration to exploitation.
    """

    def __init__(
        self,
        n_visits: int,
        reward_history: pd.DataFrame,
        *,
        item_col_name: str,
        visitor_col_name: str,
        reward_col_name: str,
        epsilon: float = 0.1,
        like_threshold: float = 4.0,
        n_iterations: int = 1,
    ):
        self.n_visits = n_visits
        self.n_iterations = n_iterations
        self.epsilon = epsilon

        item = item_col_name
        user = visitor_col_name
        rwd  = reward_col_name

        # 1) binarise ratings
        df = reward_history[[item, user, rwd]].copy()
        df["bin"] = (df[rwd] >= like_threshold).astype(np.float32)

        # 2) group into per‑item lists
        g = (
            df.groupby(item)[[user, "bin"]]
              .agg(list)
              .reset_index()
              .sort_values(item)
        )
        self.items = g[item].to_numpy()
        self.n_items = len(self.items)

        self._pairs: list[list[tuple[str, float]]] = [
            [(v, float(r)) for v, r in zip(vis, rew)]
            for vis, rew in zip(g[user], g["bin"])
        ]

    def reset(self):
        self.n_seen   = np.zeros(self.n_items,  dtype=np.int32)
        self.n_reward = np.zeros(self.n_items,  dtype=np.float64)
        self.left     = [lst.copy() for lst in self._pairs]

    def _select_arm(self) -> int:
        if np.random.rand() < self.epsilon or not self.n_seen.any():
            # explore  – uniform random over arms
            return int(np.random.randint(self.n_items))
        # exploit – best empirical CTR
        ctr = self.n_reward / np.maximum(1, self.n_seen)
        return int(np.argmax(ctr))

    def simulator(self) -> list[dict]:
        out: list[dict] = []

        for it in range(self.n_iterations):
            self.reset()
            cum = 0.0

            pbar = tqdm(
                range(self.n_visits),
                desc=f"[ε‑greedy] run {it+1}/{self.n_iterations}",
                leave=False,
            )

            for v in pbar:
                arm  = self._select_arm()
                iid  = self.items[arm]

                if not self.left[arm]:
                    visitor, reward = "∅", 0.0
                else:
                    idx              = np.random.randint(len(self.left[arm]))
                    visitor, reward  = self.left[arm].pop(idx)

                self.n_seen[arm]   += 1
                self.n_reward[arm] += reward

                cum += reward
                frac = cum / (v + 1)
                pbar.set_postfix(ctr=f"{frac:.3f}")

                out.append({
                    "iteration": it,
                    "visit":     v,
                    "item_id":   iid,
                    "visitor_id":visitor,
                    "reward":    reward,
                    "total_reward": cum,
                    "fraction_relevant": frac,
                })

        return out