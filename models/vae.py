import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim=128, image_size=32):
        super().__init__()
        if image_size == 32:
            # encodeur 32x32
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),  # 32->16
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), # 16->8
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*8*8, 512),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(512, latent_dim)
            self.fc_logvar = nn.Linear(512, latent_dim)
            
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 64*8*8),
                nn.ReLU(),
                nn.Unflatten(1, (64, 8, 8)),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),
                nn.Sigmoid()
            )
        elif image_size == 224:
            # encodeur 224x224
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),  # 224->112
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), # 112->56
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1),# 56->28
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128*28*28, 1024),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(1024, latent_dim)
            self.fc_logvar = nn.Linear(1024, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 512*7*7),
                nn.ReLU(),
                nn.Unflatten(1, (512, 7, 7)),
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), 
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError("Unsupported image_size")

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar