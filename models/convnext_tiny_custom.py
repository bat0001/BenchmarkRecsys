import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
from transformers import ConvNextImageProcessor, ConvNextModel


class ConvNextTiny(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.features_extractor = models.convnext_tiny(pretrained=True)
        self.features_extractor.classifier[2] = nn.Identity()
        
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        
        self.fc_mu = nn.Linear(768, latent_dim)
        self.fc_logvar = nn.Linear(768, latent_dim)

    def encode(self, x):
        with torch.no_grad():
            feats = self.features_extractor(x)  
        mu = self.fc_mu(feats)                 
        logvar = self.fc_logvar(feats)        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # never used
        batch_size = z.size(0)
        return torch.zeros((batch_size, 3, 224, 224), device=z.device)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class ConvNeXtTinyEncoder(nn.Module):
    def __init__(self, model_name="facebook/convnext-tiny-224", latent_dim=128):
        super().__init__()
        self.processor = ConvNextImageProcessor.from_pretrained(model_name)
        self.backbone = ConvNextModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.projection = nn.Linear(768, latent_dim)
    
    def forward(self, pil_images):
        inputs = self.processor(images=pil_images, return_tensors="pt")
        device = next(self.backbone.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            hidden = outputs.last_hidden_state
            pooled = hidden.mean(dim=[2, 3])  
        
        projected = self.projection(pooled) 
        return projected