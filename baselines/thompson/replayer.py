import numpy as np
import pandas as pd
from tqdm import tqdm


class ThompsonSamplingReplayer:
    """
    Fast Thompson‑Sampling replay.

    * Keeps only the `max_items` most‑rated products.
    * Pre‑computes one NumPy array per item: [[visitor_id, reward] …].
    * Single inner loop is pure NumPy, so no Python‑level while.
    """

    def __init__(
        self,
        n_visits: int,
        reward_history: pd.DataFrame,
        *,
        item_col_name: str,
        visitor_col_name: str,
        reward_col_name: str,
        max_items: int = 3_000,
        n_iterations: int = 5000,
    ):
        self.n_visits = n_visits
        self.n_iterations = n_iterations

        top_items = (
            reward_history[item_col_name]
            .value_counts()
            .head(max_items)
            .index
        )
        df = reward_history[reward_history[item_col_name].isin(top_items)]

        self.items    = df[item_col_name].unique()
        self.n_items  = len(self.items)

        agg = (
            df.groupby(item_col_name)[[visitor_col_name, reward_col_name]]
            .agg(list)
            .reset_index()
        )

        self.lookup = [
            list(zip(v_list, r_list))
            for v_list, r_list in zip(agg[visitor_col_name], agg[reward_col_name])
        ]
    def reset(self):
        self.alphas = np.ones(self.n_items, dtype=np.float64)
        self.betas = np.ones(self.n_items, dtype=np.float64)

    def _sample_item_index(self) -> int:
        samples = np.random.beta(self.alphas, self.betas)
        return int(np.argmax(samples))

    def simulator(self) -> list[dict]:
        results = []

        for it in range(self.n_iterations):
            self.reset()
            total = 0.0

            for visit in tqdm(range(self.n_visits),
                            desc=f"[TS] run {it+1}/{self.n_iterations}",
                            leave=False):
                idx = self._sample_item_index()
                item_id = self.items[idx]

                visitor_id, reward = self.lookup[idx][
                    np.random.randint(len(self.lookup[idx]))
                ]
                reward = float(reward)

                if reward == 1:
                    self.alphas[idx] += 1
                else:
                    self.betas[idx] += 1

                total += reward
                frac = total / (visit + 1)

                results.append(
                    dict(
                        iteration=it,
                        visit=visit,
                        item_id=item_id,
                        visitor_id=visitor_id,
                        reward=reward,
                        total_reward=total,
                        fraction_relevant=frac,
                    )
                )
            return results