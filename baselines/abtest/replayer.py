import numpy as np
from tqdm import tqdm

class ABTestReplayer:
    """
    Simulate an A/B testing replayer on historical user-item interactions,
    optimized with a single pass pre-indexing (no inner while loops).
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
        self.n_visits = n_visits
        self.n_test_visits = n_test_visits
        self.n_iterations = n_iterations

        self.df = reward_history
        self.item_col = item_col_name
        self.visitor_col = visitor_col_name
        self.reward_col = reward_col_name

        groups = self.df.groupby(self.item_col)[[self.visitor_col, self.reward_col]].agg(list)
        self.items = groups.index.to_numpy()
        self.n_items = len(self.items)

        self.item2visitors = {
            item: np.array(visitors)
            for item, visitors in zip(groups.index, groups[self.visitor_col])
        }
        self.item2rewards = {
            item: np.array(rewards, dtype=float)
            for item, rewards in zip(groups.index, groups[self.reward_col])
        }

    def reset(self):
        self.n_samples = np.zeros(self.n_items)
        self.n_rewards = np.zeros(self.n_items)
        self.is_testing = True
        self.best_idx = None

    def select_item(self) -> int:
        if self.is_testing:
            return np.random.randint(self.n_items)
        return self.best_idx

    def record_result(self, visit: int, item_idx: int, reward: float):
        self.n_samples[item_idx] += 1
        alpha = 1.0 / self.n_samples[item_idx]
        self.n_rewards[item_idx] += alpha * (reward - self.n_rewards[item_idx])
        if visit == self.n_test_visits - 1:
            self.is_testing = False
            self.best_idx = int(np.argmax(self.n_rewards))

    def simulator(self) -> list[dict]:
        results = []
        for it in range(self.n_iterations):
            tqdm.write(f"[ABTest] Run {it+1}/{self.n_iterations}")
            self.reset()
            total = 0.0
            desc = f"ABTest run {it+1}"
            for visit in tqdm(range(self.n_visits), desc=desc, leave=False):
                idx = self.select_item()
                item_id = self.items[idx]

                visitors = self.item2visitors[item_id]
                rewards  = self.item2rewards[item_id]
                pos = np.random.randint(len(visitors))
                visitor_id = visitors[pos]
                reward_val = rewards[pos]

                self.record_result(visit, idx, reward_val)
                total += reward_val
                frac = total / (visit + 1)

                results.append({
                    'iteration': it,
                    'visit': visit,
                    'item_id': item_id,
                    'visitor_id': visitor_id,
                    'reward': float(reward_val),
                    'total_reward': float(total),
                    'fraction_relevant': float(frac)
                })
            tqdm.write(f"[ABTest] Completed run {it+1}, final fraction={frac:.4f}")
        return results
