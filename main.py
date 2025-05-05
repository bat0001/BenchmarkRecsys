from __future__ import annotations

import torch
import wandb
import argparse
import numpy as np
import torch.optim as optim

from typing import Dict
from wandb import Table 
from tabulate import tabulate
import matplotlib.pyplot as plt

from models.gflownet_classical import GFlowNetMulticlass
from models.gflownet_binary_preference import BinaryPreferenceGFlowNet
from models.gflownet_dpo_preference import PreferenceGFlowNet
from trainers.gflownet_trainer import GFlowNetTrainer
from utils.config import get_config
from utils.seed import set_seed
from utils.logger import init_wandb_, log_recommendations_coco, log_recommendations_to_wandb
from utils.utils_coco import build_cat_name_to_id_map
from utils.rewards import build_reward_fn
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
from utils.device import DEVICE

from datasets.loader import DATASET_FACTORY
from models.backbone_factory import make_backbone
from utils.encoding import encode_dataset

from utils.plots import (
    plot_reward_curves, plot_selection_overlap, plot_entropy_vs_reward
)


GFN_FACTORY = {
    "classical": GFlowNetMulticlass,
    "binary": BinaryPreferenceGFlowNet,
    "dpo": PreferenceGFlowNet,
}

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

from baselines.bandit.utils import load_linucb

def evaluate_bandit(policy, embeddings, meta, reward_fn,
                    objectives, class_indices, subset_size, n_trials=1000):
    total = 0.0
    for _ in range(n_trials):
        idx = policy.select(embeddings, subset_size)
        r   = reward_fn(idx.unsqueeze(0), meta, objectives, class_indices)
        total += float(r)
    return {"Reward Mean": total / n_trials}

def random_baseline(embeddings, labels_or_anns, reward_fn,
                    objectives, class_indices,
                    subset_size, device, iters, batch):
    rewards = []
    for _ in range(iters):
        batch_rewards = []
        for _ in range(batch):
            idx = torch.randint(0, len(embeddings),
                                (subset_size,), device=device)
            r = reward_fn(idx.unsqueeze(0),
                          labels_or_anns,
                          objectives,
                          class_indices)
            batch_rewards.append(r.item())
        rewards.append(float(np.mean(batch_rewards)))
    return list(range(1, iters + 1)), rewards

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
    if isinstance(meta, torch.Tensor):
        meta = meta.to(DEVICE)  
    wandb.log({"Embeddings/pool_size": len(embeddings)})

    reward_fn = build_reward_fn(cfg, objectives, cat_map, class_indices)

    active_methods = {
        "classical": cfg.classical_gflownet or cfg.all,
        "binary": cfg.preference_binary_gflownet or cfg.all,
        "dpo": cfg.preference_dpo_gflownet or cfg.all,
        "bandit": cfg.MAB,
        "random": cfg.random_baseline
    }

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
            lambda: (
                load_linucb(
                    getattr(cfg, "bandit_ckpt", "baselines/bandit/linucb.pt"),
                    dim=embeddings.shape[1],
                    alpha=getattr(cfg, "bandit_alpha", 1.0)
                ),
                None
            ),
            lambda policy, _: evaluate_bandit(
                policy,
                embeddings,
                meta,
                reward_fn,
                objectives,
                class_indices,
                cfg.subset_size
            )
        ),
        "random": (
            lambda: random_baseline(
                embeddings, meta, reward_fn, objectives, class_indices,
                cfg.subset_size, DEVICE,
                cfg.num_iterations, cfg.batch_size_train
            ),
            lambda x, y: {"Reward Mean": float(np.mean(y)) if y else 0.0}
        ),
    }

    trained = {}
    trained_results = {}
    for name, (train_fn, eval_fn) in METHOD_DISPATCH.items():
        if active_methods[name]:
            cfg.out_dir = f"baselines/{name}"
            trained[name], trained_results[name] = train_fn()

    metrics_map = {}
    for name, (train_fn, eval_fn) in METHOD_DISPATCH.items():
        if active_methods[name]:
            metrics_map[name] = eval_fn(trained[name], trained_results[name])

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

    # if cfg.plots:
    #     histories = {
    #     name: res
    #     for name, res in trained_results.items()
    #     if isinstance(res, list) and len(res) > 0
    #     }
    #     if histories:
    #         fig = plot_reward_curves(histories, "Reward vs Iterations")
    #         wandb.log({"Plots/RewardCurves": wandb.Image(fig)})
    #         plt.close(fig)

    #     if {"bandit", "random"}.issubset(trained):
    #         idx_bandit = trained["bandit"].select(embeddings, cfg.subset_size)
    #         rand_idx   = torch.randperm(len(embeddings))[:cfg.subset_size].to(DEVICE)
    #         fig = plot_selection_overlap(idx_bandit.cpu(), rand_idx.cpu(), len(embeddings))
    #         wandb.log({"Plots/Overlap": wandb.Image(fig)})
    #         plt.close(fig)

    #     fig = plot_entropy_vs_reward(metrics_map)
    #     wandb.log({"Plots/DiversityVsReward": wandb.Image(fig)})
    #     plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    main()