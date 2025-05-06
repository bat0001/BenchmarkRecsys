import numpy as np
from tqdm import tqdm

class ABTestReplayer:
    """
    Simulate an A/B testing replayer on historical user-item interactions.
    """
    def __init__(
        self,
        n_visits: int,
        n_test_visits: int,
        reward_history, 
        item_col_name: str,
        visitor_col_name: str,
        reward_col_name: str,
        n_iterations: int = 1
    ):
        self.n_visits      = n_visits
        self.n_test_visits = n_test_visits
        self.n_iterations  = n_iterations

        self.df        = reward_history
        self.item_col  = item_col_name
        self.vis_col   = visitor_col_name
        self.rwd_col   = reward_col_name

        grp = self.df.groupby(self.item_col)[[self.vis_col, self.rwd_col]].agg(list)
        self.items     = grp.index.to_numpy()
        self.n_items   = len(self.items)
        self.item2vis  = {it: np.array(vis) for it, vis in zip(grp.index, grp[self.vis_col])}
        self.item2rwd  = {it: np.array(rw, dtype=float) for it, rw in zip(grp.index, grp[self.rwd_col])}

    def reset(self):
        self.counts  = np.zeros(self.n_items)
        self.estims  = np.zeros(self.n_items)
        self.testing = True
        self.best    = None

    def select_item(self) -> int:
        return np.random.randint(self.n_items) if self.testing else self.best

    def record_result(self, visit: int, idx: int, reward: float):
        self.counts[idx] += 1
        alpha = 1.0 / self.counts[idx]
        self.estims[idx] += alpha * (reward - self.estims[idx])
        if visit == self.n_test_visits - 1:
            self.testing = False
            self.best = int(np.argmax(self.estims))

    def simulator(self) -> list[dict]:
        results = []
        for run in range(self.n_iterations):
            tqdm.write(f"[ABTest] Run {run+1}/{self.n_iterations}")
            self.reset()
            total = 0.0

            for visit in tqdm(range(self.n_visits),
                              desc=f"[ABTest] Visits run {run+1}",
                              leave=True):
                idx = self.select_item()
                item_id = self.items[idx]

                vis_arr = self.item2vis[item_id]
                rwd_arr = self.item2rwd[item_id]
                pick = np.random.randint(len(vis_arr))
                visitor_id, reward_val = vis_arr[pick], float(rwd_arr[pick])

                self.record_result(visit, idx, reward_val)
                total += reward_val
                frac = total / (visit+1)

                results.append({
                    'run': run,
                    'visit': visit,
                    'item_id': item_id,
                    'visitor_id': visitor_id,
                    'reward': reward_val,
                    'cum_reward': total,
                    'fraction': frac
                })
            tqdm.write(f"[ABTest] Completed run {run+1}, final fraction={frac:.4f}")
        return results