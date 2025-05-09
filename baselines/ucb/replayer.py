import pandas as pd
import numpy as np
from tqdm import tqdm

class _UCBReplayer:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        item_col: str,
        reward_col: str,
        n_visits: int,
        c: float = 2.0,
        n_iter: int = 1,
        seed: int | None = None,
    ):
        self.df         = df
        self.item_col   = item_col
        self.reward_col = reward_col
        self.n_visits   = n_visits
        self.c          = c
        self.n_iter     = n_iter
        self.rng        = np.random.default_rng(seed)

        self.items   = df[item_col].unique()
        self.n_items = len(self.items)
        self.bank    = [df.loc[df[item_col] == it, reward_col].to_numpy(np.int8)
                        for it in self.items]

    def run(self) -> list[dict]:
        recs = []
        for it in tqdm(range(self.n_iter), desc="[UCB] run"):
            Q = np.zeros(self.n_items)
            N = np.zeros(self.n_items) + 1e-6 
            cum = 0.0

            for v in range(self.n_visits):
                conf = self.c * np.sqrt(np.log(v + 1) / N)
                idx  = int(np.argmax(Q + conf))

                rwd  = int(self.rng.choice(self.bank[idx]))

                N[idx] += 1
                Q[idx] += (rwd - Q[idx]) / N[idx]

                cum += rwd
                recs.append(
                    dict(iteration=it,
                         visit=v,
                         item_idx=idx,
                         reward=rwd,
                         fraction_relevant=cum / (v + 1))
                )
        return recs
