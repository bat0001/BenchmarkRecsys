import numpy as np
from tqdm import tqdm

class UCBSamplingReplayer:
    """
    Simulate UCB bandit on historical user-item interactions.
    """
    def __init__(
        self,
        ucb_c: float,
        n_visits: int,
        reward_history,
        item_col_name: str,
        visitor_col_name: str,
        reward_col_name: str,
        n_iterations: int = 1
    ):
        self.ucb_c       = ucb_c
        self.n_visits    = n_visits
        self.n_iterations= n_iterations

        self.df       = reward_history
        self.item_col = item_col_name
        self.vis_col  = visitor_col_name
        self.rwd_col  = reward_col_name

        grp = self.df.groupby(self.item_col)[[self.vis_col, self.rwd_col]].agg(list)
        self.items   = grp.index.to_numpy()
        self.n_items = len(self.items)
        self.item2vis= {it: np.array(vis) for it, vis in zip(grp.index, grp[self.vis_col])}
        self.item2rwd= {it: np.array(rw, dtype=float) for it, rw in zip(grp.index, grp[self.rwd_col])}

    def reset(self):
        self.Q        = np.zeros(self.n_items)
        self.N        = np.zeros(self.n_items) + 1e-4
        self.timestep= 1

    def select_item(self) -> int:
        ln_t = np.log(self.timestep)
        ucb  = self.ucb_c * np.sqrt(ln_t/self.N)
        action = int(np.argmax(self.Q + ucb))
        self.timestep += 1
        return action

    def record_result(self, idx: int, reward: float):
        self.N[idx] += 1
        self.Q[idx] += (reward - self.Q[idx]) / self.N[idx]

    def simulator(self) -> list[dict]:
        results = []
        for run in range(self.n_iterations):
            tqdm.write(f"[UCB] Run {run+1}/{self.n_iterations}")
            self.reset()
            total = 0.0

            for visit in tqdm(range(self.n_visits),
                              desc=f"[UCB] Visits run {run+1}",
                              leave=True):
                idx = self.select_item()
                item_id = self.items[idx]

                vis_arr = self.item2vis[item_id]
                rwd_arr = self.item2rwd[item_id]
                pick = np.random.randint(len(vis_arr))
                visitor_id, reward_val = vis_arr[pick], float(rwd_arr[pick])

                self.record_result(idx, reward_val)
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
            tqdm.write(f"[UCB] Completed run {run+1}, final fraction={frac:.4f}")
        return results