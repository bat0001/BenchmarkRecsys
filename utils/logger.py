import wandb
import torch 
import numpy as np

def init_wandb_(config):
    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config={
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "subset_size": config.subset_size,
            "batch_size_train": config.batch_size_train,
            "learning_rate": config.learning_rate,
            "num_iterations": config.num_iterations,
            "entropy_coeff": config.entropy_coeff,
            "dataset": config.dataset,
            "target_classes": config.target_classes,
            "num_epochs_vae": config.num_epochs_vae,
            "seed": config.seed
        }
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
