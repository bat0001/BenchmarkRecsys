import torch.nn as nn

class BinaryPreferenceGFlowNet(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, pool_size=40000, subset_size=4):
        super(BinaryPreferenceGFlowNet, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.subset_size = subset_size

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pool_size)
        )
    
    def forward(self, z):
        # z : (batch_size, latent_dim)
        logits = self.fc(z)  # (batch_size, pool_size)
        return logits