import torch
from abc import ABC, abstractmethod
from pathlib import Path

class BaseBaseline(ABC):
    """
    Base class for all baselines (MAB, GFlowNet, etc.)
    Provides generic train/evaluate pipeline based on embeddings + meta.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = self._build_model()
        if hasattr(model, 'to'):
            self.model = model.to(self.device)
        else:
            self.model = model

    @abstractmethod
    def _build_model(self):
        """Instantiate and return the algorithm-specific model/policy."""

    @abstractmethod
    def _update_step(self):
        """Perform one training update using self.embeddings + self.meta."""

    @abstractmethod
    def _select_indices(self):
        """Select a subset of indices (Tensor) for evaluation."""

    def train(self, embeddings, meta):
        """
        Generic training loop given precomputed embeddings and metadata.
        """
        self.embeddings = embeddings.to(self.device)
        if isinstance(meta, torch.Tensor):
            self.meta = meta.to(self.device)
        else:
            self.meta = meta

        for it in range(self.cfg.num_iterations):
            outcome = self._update_step()
            if (it + 1) % 500 == 0:
                print(f"[it={it+1}] outcome={outcome:.3f}")

        ckpt_path = self.out_dir / f"{self.cfg.baseline}.pt"
        if hasattr(self.model, 'state_dict'):
            torch.save(self.model.state_dict(), ckpt_path)
        else:
            torch.save(self.model, ckpt_path)
        print(f"✔ Saved checkpoint to {ckpt_path}")

    def evaluate(self, n_trials: int = 1000):
        """
        Generic evaluation: run n_trials selections and average rewards.
        """
        total_reward = 0.0
        for _ in range(n_trials):
            idx = self._select_indices()
            r = self.cfg.reward_fn(
                idx.unsqueeze(0),
                self.meta,
                self.cfg.objectives,
                self.cfg.class_idx
            )
            total_reward += float(r)
        mean_reward = total_reward / n_trials
        return {"Reward Mean": mean_reward}
