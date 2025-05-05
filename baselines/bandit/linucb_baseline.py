import torch

from baselines.core.baseline import BaseBaseline
from baselines.bandit.linucb import LinUCB

class LinUCBBaseline(BaseBaseline):

    def __init__(self, cfg):
        super().__init__(cfg)             

        self.embeddings = None
        self.meta        = None
        self.objectives  = None
        self.class_idx   = None
        self.reward_fn   = None

    def _build_model(self):
        return LinUCB(self.cfg.latent_dim,
                      alpha=self.cfg.bandit_alpha,
                      device=self.device)

    def _update_step(self):
        idx = self.model.select(self.embeddings, self.cfg.subset_size)
        r   = self.reward_fn(idx.unsqueeze(0),
                             self.meta,
                             self.objectives,
                             self.class_idx)
        self.model.update(self.embeddings[idx], r)
        return float(r)

    def _select_indices(self):
        return self.model.select(self.embeddings, self.cfg.subset_size)