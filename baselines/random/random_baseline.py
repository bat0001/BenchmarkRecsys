import numpy as np
import torch
from baselines.core.baseline import BaseBaseline

class RandomBaseline(BaseBaseline):

    def offline_fit(self, data):
        if isinstance(data, tuple):
            self.embeddings, self.meta = data
        else:
            self.df = data
        return self

    def online_simulate(self, n_iters: int) -> dict:
        if hasattr(self, 'embeddings'):
            rewards = []
            for _ in range(n_iters):
                idx = torch.randint(
                    0, len(self.embeddings),
                    (self.cfg.subset_size,),
                    device=self.device
                )
                r = self.cfg.reward_fn(
                    idx.unsqueeze(0),
                    self.meta,
                    self.cfg.objectives,
                    self.cfg.class_indices
                )
                rewards.append(r.item())
            return {"Reward Mean": float(np.mean(rewards))}
        else:
            ratings = self.df['rating'].sample(
                n_iters, replace=True
            ).to_numpy().astype(float)
            return {"Reward Mean": float(np.mean(ratings))}
