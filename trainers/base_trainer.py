import torch
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_step(self, batch):
        raise NotImplementedError("train_step has to be inplemented.")
    
    def train(self, dataloader, num_epochs, logger):
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            for batch, _ in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
                loss = self.train_step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            logger.log({"Loss": avg_loss})