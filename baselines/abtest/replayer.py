import numpy as np
from tqdm import tqdm

class ABTestReplayer:
    """
    Simulate an A/B testing replayer on historical user-item interactions.
    """
    def __init__(self,
                 n_visits: int,
                 n_test_visits: int,
                 reward_history,
                 item_col_name: str,
                 visitor_col_name: str,
                 reward_col_name: str,
                 n_iterations: int = 1):
        self.reward_history = reward_history
        self.item_col_name = item_col_name
        self.visitor_col_name = visitor_col_name
        self.reward_col_name = reward_col_name
        self.n_iterations = n_iterations

        self.n_visits = n_visits
        self.n_test_visits = n_test_visits

        self.items = self.reward_history[self.item_col_name].unique()
        self.n_items = len(self.items)

        self.visitors = self.reward_history[self.visitor_col_name].unique()
        self.n_visitors = len(self.visitors)

    def reset(self):
        self.n_item_samples = np.zeros(self.n_items)
        self.n_item_rewards = np.zeros(self.n_items)
        self.is_testing = True
        self.best_item_idx = None

    def select_item(self) -> int:
        if self.is_testing:
            return np.random.randint(self.n_items)
        return self.best_item_idx

    def record_result(self, visit: int, item_idx: int, reward: float):
        self.n_item_samples[item_idx] += 1
        alpha = 1.0 / self.n_item_samples[item_idx]
        self.n_item_rewards[item_idx] += alpha * (reward - self.n_item_rewards[item_idx])

        if visit == self.n_test_visits - 1:
            self.is_testing = False
            self.best_item_idx = int(np.argmax(self.n_item_rewards))

    def simulator(self) -> list[dict]:
        results = []
        for it in tqdm(range(self.n_iterations), desc="ABTestReplayer"):
            self.reset()
            total_rewards = 0.0
            fraction_relevant = np.zeros(self.n_visits)

            for visit in range(self.n_visits):
                found = False
                while not found:
                    vidx = np.random.randint(self.n_visitors)
                    visitor_id = self.visitors[vidx]
                    item_idx = self.select_item()
                    item_id = self.items[item_idx]
                    df = self.reward_history
                    reward_series = df[(df[self.item_col_name] == item_id) &
                                       (df[self.visitor_col_name] == visitor_id)][self.reward_col_name]
                    found = len(reward_series) > 0

                reward_val = float(reward_series.iloc[0])
                self.record_result(visit, item_idx, reward_val)

                total_rewards += reward_val
                fraction_relevant[visit] = total_rewards / (visit + 1)

                results.append({
                    'iteration': it,
                    'visit': visit,
                    'item_id': item_id,
                    'visitor_id': visitor_id,
                    'reward': reward_val,
                    'total_reward': total_rewards,
                    'fraction_relevant': fraction_relevant[visit]
                })
        return results
