from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

from datasets.datasets import prepare_dataset
from models.vae import VAE
from models.convnext_tiny_custom import ConvNextTiny, ConvNeXtTinyEncoder
from models.gflownet_classical import GFlowNetMulticlass
from models.gflownet_binary_preference import BinaryPreferenceGFlowNet
from models.gflownet_dpo_preference import PreferenceGFlowNet
from trainers.gflownet_trainer import GFlowNetTrainer
from utils.config import get_config
from utils.seed import set_seed
from utils.logger import init_wandb_, log_recommendations_coco, log_recommendations_to_wandb
from utils.filter import filter_dataset_all_classes, filter_coco_inplace
from utils.utils_coco import (
    build_cat_name_to_id_map, parse_objectives_coco,
    coco_collate_fn_keep_target_with_index,
)
from utils.rewards import reward_function_multiclass, reward_function_coco
from utils.visualization import (
    compute_image_selection_probabilities,
    plot_image_probability_by_objective,
    plot_image_probability_by_targets_cifar,
    plot_comparison,
    plot_comparison_iterations,
)
from utils.umap_ import plot_umap_for_sequences
from utils.metrics import (
    evaluate_model,
    evaluate_model_coco,
    measure_sequence_diversity_from_list,
    measure_image_diversity_from_list,
)
from utils.sampling import sample_many_sequences
from tabulate import tabulate
from wandb import Table

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BanditPolicy:
    def select(self, contexts: torch.Tensor, k: int) -> torch.Tensor:
        """Renvoie les indices des k images à proposer."""
    def update(self, chosen: torch.Tensor, reward: torch.Tensor): ...

import torch

class LinUCB(BanditPolicy):
    def __init__(self, dim: int, alpha: float = 1.0, device="cpu"):
        self.A = torch.eye(dim, device=device)        # d×d
        self.b = torch.zeros(dim, device=device)      # d
        self.alpha = alpha
        self.device = device

    def _ucb(self, X):  # X: n×d
        A_inv = torch.linalg.inv(self.A)
        theta = A_inv @ self.b                       # d
        means = X @ theta                            # n
        vars = torch.sum(X @ A_inv * X, dim=1)       # n
        return means + self.alpha * torch.sqrt(vars)

    @torch.no_grad()
    def select(self, contexts: torch.Tensor, k: int) -> torch.Tensor:
        """Greedy top‑k according to current UCB score."""
        scores = self._ucb(contexts)                 # n
        return torch.topk(scores, k=k).indices       # k

    def update(self, chosen_ctx: torch.Tensor, reward: torch.Tensor):
        """chosen_ctx: k×d  – reward: scalaire ou (k,) broadcasté."""
        if reward.ndim == 0:
            reward = reward.expand(len(chosen_ctx))
        for x, r in zip(chosen_ctx, reward):
            x = x.unsqueeze(1)                      # d×1
            self.A += x @ x.T
            self.b += r * x.squeeze()

def train_bandit(policy: BanditPolicy,
                 embeddings: torch.Tensor,
                 labels_or_anns,
                 reward_fn,            # même signature que GFN
                 objectives, class_indices,
                 subset_size=4,
                 iters=10_000, batch=1):
    rewards_vs_iter = []
    for it in range(iters):
        idx = policy.select(embeddings, subset_size)        # k idx
        rew = reward_fn(idx.unsqueeze(0),
                        labels_or_anns,
                        objectives,
                        class_indices)                      # tensor[1]
        policy.update(embeddings[idx], rew)                 # online update
        if (it+1) % 100 == 0:
            rewards_vs_iter.append(rew.item())
    return rewards_vs_iter
        
def _load_coco(cfg):
    ds, _ = prepare_dataset("COCO")
    objectives = parse_objectives_coco(cfg.target_classes_coco)
    ds = filter_coco_inplace(ds, objectives, retaining_ratio=0.01)
    cat_ids = sorted(ds.coco.getCatIds())
    class_names = [c["name"] for c in ds.coco.loadCats(cat_ids)]
    class_indices = {n: i for i, n in enumerate(class_names)}
    return ds, objectives, class_names, class_indices


def _load_cifar(cfg):
    ds, _ = prepare_dataset(cfg.dataset)
    num = getattr(cfg, 'num_per_class', 200)
    ds = filter_dataset_all_classes(ds, num_per_class=num)
    objectives = cfg.target_classes
    class_indices = {n: ds.classes.index(n) for n in objectives}
    return ds, objectives, ds.classes, class_indices

DATASET_FACTORY = {
    "COCO": _load_coco,
    "CIFAR-10": _load_cifar,
    "CIFAR-100": _load_cifar,
}

def make_backbone(cfg):
    if cfg.model_type == "vae":
        img_sz = 32 if cfg.dataset.startswith("CIFAR") else 224
        return VAE(cfg.latent_dim, img_sz).to(DEVICE).eval()
    # convnext
    if cfg.dataset == "COCO":
        return ConvNeXtTinyEncoder("facebook/convnext-tiny-224").to(DEVICE).eval()
    return ConvNextTiny(cfg.latent_dim).to(DEVICE).eval()



def encode_dataset(backbone: torch.nn.Module, ds, cfg) -> Tuple[torch.Tensor, Any]:
    """Encode dataset into latent embeddings + labels or annotations.

    Returns:
        embeddings: Tensor of shape [N, latent_dim]
        meta: Tensor of labels (for non-COCO) or list of annotations (for COCO)
    """
    kwargs = {}
    is_coco = (cfg.dataset == "COCO")
    if is_coco:
        kwargs["collate_fn"] = coco_collate_fn_keep_target_with_index
    dl = DataLoader(ds, batch_size=128, shuffle=False, **kwargs)

    embeddings_list, meta_list = [], []
    for batch in dl:
        if is_coco:
            imgs, anns, _ = batch
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                if hasattr(backbone, "encode"):
                    mu, _ = backbone.encode(imgs)
                    z = mu
                else:
                    z = backbone(imgs)
            embeddings_list.append(z.cpu())
            meta_list.extend(anns)
        else:
            imgs, lbls = batch
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                if hasattr(backbone, "encode"):
                    mu, _ = backbone.encode(imgs)
                    z = mu
                else:
                    z = backbone(imgs)
            embeddings_list.append(z.cpu())
            # lbls: Tensor of shape [batch_size]
            meta_list.append(lbls)
    embeddings = torch.cat(embeddings_list, dim=0)
    if is_coco:
        return embeddings, meta_list
    # non-COCO: flatten label tensors to one 1D tensor
    labels = torch.cat(meta_list, dim=0)
    return embeddings, labels

def build_reward_fn(cfg, objectives, cat_map, class_indices):
    if cfg.dataset == "COCO":
        def _fn(indices, all_annots, *_):
            return reward_function_coco(indices, all_annots, cat_map, objectives, device=DEVICE)
        return _fn
    return reward_function_multiclass

GFN_FACTORY = {
    "classical": GFlowNetMulticlass,
    "binary": BinaryPreferenceGFlowNet,
    "dpo": PreferenceGFlowNet,
}


def random_baseline(embeddings, labels_or_anns, reward_fn, objectives, class_indices, subset_size, device, iters, batch):
    if isinstance(labels_or_anns, torch.Tensor):
        labels_or_anns = labels_or_anns.to(device)
    rewards = []
    rewards = []
    for _ in range(iters):
        batch_rewards = []
        for _ in range(batch):
            idx = torch.randint(0, len(embeddings), (subset_size,), device=device)
            r = reward_fn(idx.unsqueeze(0), labels_or_anns, objectives, class_indices)
            batch_rewards.append(r.item())
        rewards.append(float(np.mean(batch_rewards)))
    return list(range(1, iters + 1)), rewards

def train_head(name: str, cfg, embeddings, meta, objectives, class_indices, reward_fn):
    ModelCls = GFN_FACTORY[name]
    model = ModelCls(cfg.latent_dim, cfg.hidden_dim, len(embeddings), cfg.subset_size).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    trainer = GFlowNetTrainer(model, opt, DEVICE, reward_fn, class_indices, cfg.subset_size,
                              cfg.entropy_coeff, 10.0, 1.0 if name == "dpo" else 6.0)
    if name == "classical":
        trainer.train_iterations_classical(embeddings, meta, objectives, cfg.num_iterations,
                                           cfg.batch_size_train, wandb)
    elif name == "binary":
        trainer.train_iterations_binary_preference(embeddings, meta, objectives, cfg.num_iterations,
                                                   cfg.batch_size_train, wandb)
    else:
        trainer.train_iterations_comparison(embeddings, meta, objectives, cfg.num_iterations,
                                            cfg.batch_size_train, wandb)
    return trainer, model


def evaluate_head(name, trainer, model, embeddings, meta, cfg, objectives, cat_map, class_indices, class_names, ds):
    if cfg.dataset == "COCO":
        metrics, idx = evaluate_model_coco(model, embeddings, meta, objectives, cat_map,
                                           cfg.subset_size, DEVICE, top_k=False)
        image_probs = compute_image_selection_probabilities(model, embeddings, DEVICE)
        fig = plot_image_probability_by_objective(image_probs, meta, objectives, cat_map,
                                                  title=f"{name}/Selection Prob.")
        wandb.log({f"{name}/Image_Prob": wandb.Image(fig)})
        log_recommendations_coco(idx.cpu().numpy().flatten(), ds, f"{name}/Recs")
    else:
        metrics, idx = evaluate_model(model, embeddings, meta, objectives, class_indices,
                                      subset_size=cfg.subset_size, device=DEVICE, top_k=False)
        image_probs = compute_image_selection_probabilities(model, embeddings, DEVICE)
        fig = plot_image_probability_by_targets_cifar(image_probs, meta, objectives,
                                                      class_names,
                                                      title=f"{name}/Selection Prob.")
        wandb.log({f"{name}/Image_Prob": wandb.Image(fig)})
        log_recommendations_to_wandb(idx.cpu().numpy().flatten(), ds, f"{name}/Recs")
    return metrics

def log_comparison(metrics_map: Dict[str, Dict[str, float]]):
    headers = [
        "Method", "Precision", "Recall", "F1", "ExactMatch",
        "SeqEntropy", "ImgEntropy", "RewardMean", "RewardMedian",
        "RewardMin", "RewardMax",
    ]
    table = Table(columns=headers)
    for name, m in metrics_map.items():
        row = [name] + [f"{m.get(k, 0):.3f}" for k in [
            "Precision", "Recall", "F1-Score", "Exact Match Ratio",
            "Sequence Entropy", "Image Entropy", "Reward Mean", "Reward Median",
            "Reward Min", "Reward Max",
        ]]
        table.add_data(*row)
    wandb.log({"Comparison": table})
    print(tabulate(table.data, headers=headers, tablefmt="github"))

# ---------------------------------------------------------------------------
# 8 ▸ MAIN -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--it", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    args, _ = parser.parse_known_args()

    cfg = get_config()
    set_seed(cfg.seed)
    init_wandb_(cfg)

    ds, objectives, class_names, class_indices = DATASET_FACTORY[cfg.dataset](cfg)
    cat_map = build_cat_name_to_id_map(ds.coco) if cfg.dataset == "COCO" else None

    backbone = make_backbone(cfg)
    embeddings, meta = encode_dataset(backbone, ds, cfg)
    # Move labels tensor to GPU for non-COCO (avoid index device mismatch)
    if isinstance(meta, torch.Tensor):
        meta = meta.to(DEVICE)  
    wandb.log({"Embeddings/pool_size": len(embeddings)})

    reward_fn = build_reward_fn(cfg, objectives, cat_map, class_indices)

    # Determine which methods to run: GFlowNet variants or MAB
    active_methods = {
        "classical": cfg.classical_gflownet or cfg.all,
        "binary": cfg.preference_binary_gflownet or cfg.all,
        "dpo": cfg.preference_dpo_gflownet or cfg.all,
        "bandit": args.MAB,
    }

    # Dispatch table: each method yields (trainer_or_policy, model_or_rewards)
    METHOD_DISPATCH = {
        "classical": (
            lambda: train_head("classical", cfg, embeddings, meta, objectives, class_indices, reward_fn),
            lambda t, m: evaluate_head("classical", t, m, embeddings, meta, cfg, objectives, cat_map, class_indices, class_names, ds)
        ),
        "binary": (
            lambda: train_head("binary", cfg, embeddings, meta, objectives, class_indices, reward_fn),
            lambda t, m: evaluate_head("binary", t, m, embeddings, meta, cfg, objectives, cat_map, class_indices, class_names, ds)
        ),
        "dpo": (
            lambda: train_head("dpo", cfg, embeddings, meta, objectives, class_indices, reward_fn),
            lambda t, m: evaluate_head("dpo", t, m, embeddings, meta, cfg, objectives, cat_map, class_indices, class_names, ds)
        ),
        "bandit": (
            # returns (policy, rewards_list)
            lambda: (
                LinUCB(embeddings.shape[1], alpha=cfg.bandit_alpha, device=DEVICE),
                train_bandit(
                    LinUCB(embeddings.shape[1], alpha=cfg.bandit_alpha, device=DEVICE),
                    embeddings, meta, reward_fn, objectives, class_indices,
                    subset_size=cfg.subset_size,
                    iters=cfg.num_iterations,
                    batch=cfg.batch_size_train
                )
            ),
            # evaluate returns simple metrics dict
            lambda policy, rewards: {"Reward Mean": rewards[-1] if rewards else 0.0}
        ),
    }

    # TRAIN all selected methods
    trained = {}
    trained_results = {}
    for name, (train_fn, eval_fn) in METHOD_DISPATCH.items():
        if active_methods[name]:
            trained[name], trained_results[name] = train_fn()

    # EVALUATE all selected methods
    metrics_map = {}
    for name, (train_fn, eval_fn) in METHOD_DISPATCH.items():
        if active_methods[name]:
            metrics_map[name] = eval_fn(trained[name], trained_results[name])

    # Log comparative table
    log_comparison(metrics_map)

    active_heads = {
        "classical": cfg.classical_gflownet or cfg.all,
        "binary": cfg.preference_binary_gflownet or cfg.all,
        "dpo": cfg.preference_dpo_gflownet or cfg.all,
    }
    trainers, models = {}, {}
    for name, flag in active_heads.items():
        if flag:
            trainers[name], models[name] = train_head(name, cfg, embeddings, meta, objectives,
                                                      class_indices, reward_fn)

    rand_x, rand_y = random_baseline(embeddings, meta, reward_fn, objectives, class_indices,
                                     cfg.subset_size, DEVICE, cfg.num_iterations,
                                     cfg.batch_size_train)

    metrics_map = {}
    for name, trainer in trainers.items():
        m = evaluate_head(name, trainer, models[name], embeddings, meta, cfg,
                          objectives, cat_map, class_indices, class_names, ds)
        seqs = sample_many_sequences(models[name], embeddings, 5, cfg.subset_size, DEVICE, top_k=False)
        m["Sequence Entropy"] = measure_sequence_diversity_from_list(seqs)
        m["Image Entropy"] = measure_image_diversity_from_list(seqs)
        metrics_map[name] = m

    log_comparison(metrics_map)

    if {"classical", "dpo"}.issubset(metrics_map):
        plot_comparison(trainers["classical"].iterations_classical,
                        trainers["classical"].reward_vs_images_classical,
                        trainers["dpo"].iterations_comparison,
                        trainers["dpo"].reward_vs_images_comparison,
                        rand_x, rand_y, max_reward=None,
                        title="Images seen vs Reward",
                        x_label="Images", y_label="Reward",
                        wandb_key="RewardVsImages")
        plot_comparison_iterations(trainers["classical"].iterations_classical,
                                   trainers["classical"].reward_vs_images_classical,
                                   trainers["dpo"].iterations_comparison,
                                   trainers["dpo"].reward_vs_images_comparison,
                                   rand_x, rand_y, max_reward=None,
                                   title="Iterations vs Reward",
                                   x_label="Iter", y_label="Reward",
                                   wandb_key="RewardVsIter")

    plot_umap_for_sequences(cfg, DEVICE, embeddings,
                            trainers.get("classical").model if "classical" in trainers else None,
                            trainers.get("dpo").model if "dpo" in trainers else None,
                            all_annotations=meta if cfg.dataset == "COCO" else None,
                            cat_map=cat_map if cfg.dataset == "COCO" else None,
                            all_labels=meta if cfg.dataset != "COCO" else None,
                            target_classes=objectives)

    if args.it and "dpo" in trainers:
        trainers["dpo"].iterative_human_feedback(embeddings, meta, objectives,
                                                 class_indices, class_names, cfg.subset_size,
                                                 iterations=args.iterations)

    wandb.finish()

if __name__ == "__main__":
    main()
