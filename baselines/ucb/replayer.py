import numpy as np
from tqdm import tqdm

class UCBSamplingReplayer:
    """
    Simulate UCB (Upper‐Confidence Bound) bandit on historical interactions.
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
        self.ucb_c = ucb_c
        self.n_visits = n_visits
        self.n_iterations = n_iterations

        self.df = reward_history
        self.item_col = item_col_name
        self.visitor_col = visitor_col_name
        self.reward_col = reward_col_name

        self.items = self.df[self.item_col].unique()
        self.n_items = len(self.items)

        groups = reward_history.groupby(item_col_name)[[visitor_col_name,
                                                reward_col_name]].agg(list)


        self.item2rows = {
            item: np.column_stack(groups.loc[item].values)   
            for item in groups.index
        }

    def reset(self):
        self.Q = np.zeros(self.n_items)
        self.N = np.zeros(self.n_items) + 1e-4
        self.timestep = 1

    def select_item(self) -> int:
        ln_t = np.log(self.timestep)
        ucb = self.ucb_c * np.sqrt(ln_t / self.N)
        action = int(np.argmax(self.Q + ucb))
        self.timestep += 1
        return action

    def record_result(self, item_idx: int, reward: float):
        self.N[item_idx] += 1
        self.Q[item_idx] += (reward - self.Q[item_idx]) / self.N[item_idx]

    def simulator(self) -> list[dict]:
        results = []
        for it in range(self.n_iterations):
            print(f"[UCB] Run {it+1}/{self.n_iterations}")
            self.reset()
            total_rewards = 0.0

            for visit in range(self.n_visits):
                if visit % max(1, self.n_visits // 10) == 0:
                    print(f"  Visit {visit+1}/{self.n_visits}")

                item_idx = self.select_item()
                item_id = self.items[item_idx]

                rows = self.item2rows[item_id]
                visitor_id, reward_val = rows[np.random.randint(len(rows))]

                self.record_result(item_idx, float(reward_val))
                total_rewards += float(reward_val)
                frac = total_rewards / (visit + 1)

                results.append({
                    'iteration': it,
                    'visit': visit,
                    'item_id': item_id,
                    'visitor_id': visitor_id,
                    'reward': float(reward_val),
                    'total_reward': total_rewards,
                    'fraction_relevant': frac
                })
            print(f"[UCB] Completed run {it+1}/{self.n_iterations}, final frac={frac:.4f}")
        return results