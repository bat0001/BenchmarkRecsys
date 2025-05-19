import wandb
import torch 
import numpy as np
from omegaconf import OmegaConf

def init_wandb_(config):
    wandb.init(
        project = config.wandb.project,
        name    = config.wandb.run_name,
        mode    = "offline" if config.wandb.get("offline", False) else "online",
        config  = OmegaConf.to_container(config, resolve=True)  
    )

def log_recommendations_coco(indices, dataset, key):
    images_wandb = []
    for i in indices:
        i_python = int(i)  
        pil_img, annots, real_idx = dataset[i_python] 

        if isinstance(pil_img, torch.Tensor):
            np_img = pil_img.permute(1, 2, 0).cpu().numpy()
        else:
            np_img = np.array(pil_img)

        caption = f"COCO idx={i_python}, #objs={len(annots)}"
        images_wandb.append(wandb.Image(np_img, caption=caption))

    wandb.log({key: images_wandb})

def log_recommendations_to_wandb(indices, dataset, key):
    images = [
        wandb.Image(dataset.data[idx], caption=f"Image {idx} - {dataset.classes[dataset.targets[idx]]}")
        for idx in indices
    ]
    wandb.log({key: images})
