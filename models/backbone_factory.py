import torch

from models.vae import VAE
from models.convnext_tiny_custom import ConvNextTiny, ConvNeXtTinyEncoder
from utils.device import DEVICE

def make_backbone(cfg):
    if cfg.model_type == "vae":
        img_sz = 32 if cfg.dataset.startswith("CIFAR") else 224
        return VAE(cfg.latent_dim, img_sz).to(DEVICE).eval()
    if cfg.dataset == "COCO":
        return ConvNeXtTinyEncoder("facebook/convnext-tiny-224").to(DEVICE).eval()
    return ConvNextTiny(cfg.latent_dim).to(DEVICE).eval()