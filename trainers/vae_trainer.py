import torch
from trainers.base_trainer import BaseTrainer
from tqdm import tqdm

class VAETTrainer(BaseTrainer):
    def __init__(self, model, optimizer, device, criterion, logger):
        super(VAETTrainer, self).__init__(model, optimizer, device)
        self.criterion = criterion
        self.logger = logger
    
    def train_step(self, batch):
        batch = batch.to(self.device)
        recon, mu, logvar = self.model(batch)
        recon_loss = self.criterion(recon, batch)
        # Divergence KL
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + KLD
        return loss