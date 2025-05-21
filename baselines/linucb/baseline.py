from tqdm import tqdm
import numpy as np
import pandas as pd
from baselines.core.baseline import BaseBaseline
from utils.features import build_feature_matrix


class LinUCBBaseline(BaseBaseline):
    """
    Contexte = features (num + cat + texte)  →  LinUCB (Li et al. 2010).
    """

    def _build_model(self):      
        return None

    def _row_dense(self, idx: int) -> np.ndarray:
        row = self.X[idx]
        return row.toarray().ravel() if hasattr(row, "toarray") else row

    def offline_fit(self, canon_df: pd.DataFrame,
                    item_col: str = "productId",
                    reward_col: str = "reward",
                    min_pos: int = 1,
                    max_items: int = 5_000):

        pos_cnt = canon_df[canon_df[reward_col] == 1][item_col].value_counts()
        keep_ids = pos_cnt[pos_cnt >= min_pos].index[:max_items]

        df = canon_df[canon_df[item_col].isin(keep_ids)].reset_index(drop=True)
        self.df = df                                    

        self.items       = list(range(len(df)))         
        self.item_keys   = df[item_col].tolist()        

        self.bank = {i: df.loc[df.index == i, reward_col].to_numpy(np.int8)
                     for i in self.items}

        self.X = build_feature_matrix(df).astype(np.float32)
        if hasattr(self.X, "toarray"):      
            self.X = self.X.toarray()      

        d = self.X.shape[1]
        self.A_inv = {i: np.eye(d) for i in self.items} 
        self.b      = {i: np.zeros(d) for i in self.items}

        return self

    def _ucb(self, idx, alpha):
        A_inv = self.A_inv[idx]
        theta = A_inv @ self.b[idx]
        # x     = self.X[idx]
        x = self._row_dense(idx)
        mu    = float(theta @ x)
        var   = float(np.sqrt(x @ A_inv @ x))
        return mu + alpha * var

    @staticmethod
    def _sm_update(A_inv, x):
        # Sherman‑Morrison one‑rank update
        Ax = A_inv @ x
        denom = 1.0 + x @ Ax
        A_inv_new = A_inv - np.outer(Ax, Ax) / denom
        return A_inv_new

    def online_simulate(self,
                        n_visits: int,
                        *,
                        alpha: float | None = None,
                        return_raw: bool = True):

        alpha = alpha or getattr(self.cfg, "linucb_alpha", 1.0)
        rng   = np.random.default_rng(self.cfg.seed)

        results, cum = [], 0.0
        for v in tqdm(range(n_visits), desc="[LinUCB] replay"):
            scores = [self._ucb(i, alpha) for i in self.items]
            idx    = int(np.argmax(scores))      
            key    = self.item_keys[idx]            # identifiant réel

            r      = int(rng.choice(self.bank[idx]))

            # x = self.X[idx]
            x = self._row_dense(idx)
            self.A_inv[idx] = self._sm_update(self.A_inv[idx], x)
            self.b[idx]    += r * x

            cum += r
            results.append(dict(
                visit=v,
                row_idx=idx,
                item_key=key,
                reward=r,
                fraction_relevant=cum / (v + 1)
            ))

        metrics = {"CTR": cum / n_visits}
        return (metrics, results) if return_raw else metrics