import torch 

from baselines.core.baseline import BaseBaseline
from models.gflownet_classical import GFlowNetMulticlass
from trainers.gflownet_trainer import gflownet_loss_multiclass

class ClassicalGFNBaseline(BaseBaseline):
    def _build_model(self):
        return GFlowNetMulticlass(self.cfg.latent_dim,
                                  self.cfg.hidden_dim,
                                  self.cfg.pool_size,
                                  self.cfg.subset_size).to(self.cfg.device)

    def _get_batch(self, dataset):
        return dataset.sample_batch(self.cfg.batch_size_train)  #

    def _update_step(self, batch):
        emb, labels = batch
        logits = self.model(emb)
        loss = gflownet_loss_multiclass(..., logits, self.cfg.entropy_coeff)
        self.cfg.optimizer.zero_grad()
        loss.backward()
        self.cfg.optimizer.step()
        return loss.item()

    def _select_indices(self):
        with torch.no_grad():
            logits = self.model(self.cfg.embeddings)
            probs = torch.softmax(logits, dim=1)
        return torch.multinomial(probs, self.cfg.subset_size, replacement=False)