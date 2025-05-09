from __future__ import annotations

import torch
import wandb
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Dict
from wandb import Table 
from tabulate import tabulate

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
    plot_reward_curves, plot_selection_overlap, plot_cumulative_regret, plot_fraction_relevant_curves
)
from baselines import BASELINE_REGISTRY
from baselines.abtest.abtest_baseline import ABTestBaseline
from baselines.ucb.ucb_baseline import UCBBaseline
from baselines.bandit.utils import load_linucb
from baselines.random.random_baseline import RandomBaseline
from baselines.thompson.baseline import ThompsonBaseline
from baselines.epsilon.baseline import EpsilonGreedyBaseline

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


def evaluate_bandit(policy, embeddings, meta, reward_fn,
                    objectives, class_indices, subset_size, n_trials=1000):
    total = 0.0
    for _ in range(n_trials):
        idx = policy.select(embeddings, subset_size)
        r   = reward_fn(idx.unsqueeze(0), meta, objectives, class_indices)
        total += float(r)
    return {"Reward Mean": total / n_trials}

# def random_baseline(embeddings, labels_or_anns, reward_fn,
#                     objectives, class_indices,
#                     subset_size, device, iters, batch):
#     rewards = []
#     for _ in range(iters):
#         batch_rewards = []
#         for _ in range(batch):
#             idx = torch.randint(0, len(embeddings),
#                                 (subset_size,), device=device)
#             r = reward_fn(idx.unsqueeze(0),
#                           labels_or_anns,
#                           objectives,
#                           class_indices)
#             batch_rewards.append(r.item())
#         rewards.append(float(np.mean(batch_rewards)))
#     return list(range(1, iters + 1)), rewards

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


def filter_df_for_bandit(df, item_col="productId",
                         reward_col="reward", min_pos=5, max_items=5_000):
    pos = df[df[reward_col] == 1][item_col].value_counts()
    keep = pos[pos >= min_pos].index[:max_items]      
    return df[df[item_col].isin(keep)].copy()

def run_thompson_sampling(
        df            : pd.DataFrame,
        item_col      : str = "productId",
        reward_col    : str = "reward",        
        n_visits      : int = 5_000,
        n_iterations  : int = 10,
        rng_seed      : int | None = None
) -> tuple[list[dict], float]:
    rng = np.random.default_rng(rng_seed)
    base_rng = np.random.default_rng(rng_seed)
    items = df[item_col].unique()
    n_items = len(items)

    lookup = [df[df[item_col] == it][reward_col].to_numpy(np.int8) for it in items]

    all_results = []
    final_ctrs = []

    for it in tqdm(range(n_iterations), desc="[TS] run"):
        rng = np.random.default_rng(base_rng.integers(1_000_000_000))
        alpha = np.ones(n_items, dtype=np.float64)
        beta = np.ones(n_items, dtype=np.float64)

        cum = 0.0
        for v in range(n_visits):
            sample = rng.beta(alpha, beta)
            idx = int(np.argmax(sample))
            rwd = int(rng.choice(lookup[idx]))

            if rwd == 1:
                alpha[idx] += 1.0
            else:
                beta[idx] += 1.0

            cum += rwd
            all_results.append(dict(
                iteration=it,
                visit=v,
                item_idx=idx,
                reward=rwd,
                fraction_relevant=cum / (v + 1)
            ))

        final_ctrs.append(cum / n_visits)

    mean_final_ctr = float(np.mean(final_ctrs))
    return all_results, mean_final_ctr


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--it", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    args, _ = parser.parse_known_args()

    cfg = get_config()
    # set_seed(cfg.seed)
    init_wandb_(cfg)
    metrics_map = {}
    raw_results = {}

    ds, objectives, class_names, class_indices = DATASET_FACTORY[cfg.dataset](cfg)
    cat_map = build_cat_name_to_id_map(ds.coco) if cfg.dataset == "COCO" else None

    if cfg.dataset == "AMAZON":
        embeddings = None
        # ds = filter_df_for_bandit(ds, min_pos=5, max_items=100)
        
        # meta = ds
    else:
        backbone = make_backbone(cfg)
        embeddings, meta = encode_dataset(backbone, ds, cfg)
        if isinstance(meta, torch.Tensor):
            meta = meta.to(DEVICE)
        wandb.log({"Embeddings/pool_size": len(embeddings)})
    
    if cfg.dataset == "AMAZON" and cfg.simple_ts:
        if "reward" not in ds.columns:
            ds["reward"] = (ds["rating"] > cfg.reward_threshold).astype(int)

        ds_filtered = filter_df_for_bandit(ds, item_col="productId",
                                       reward_col="reward",
                                       min_pos=5, max_items=50)
        assert set(ds_filtered["reward"].unique()).issubset({0, 1})
        ts_raw, ts_ctr = run_thompson_sampling(
            ds_filtered,
            item_col="productId",
            reward_col="reward",
            n_visits=5000,
            n_iterations=1000,
            rng_seed=cfg.seed,
        )

        metrics_map["simple_ts"] = {
            "Reward Mean": float(np.mean([x["reward"] for x in ts_raw])),
            "Reward Median": float(np.median([x["reward"] for x in ts_raw])),
            "Reward Max": float(np.max([x["reward"] for x in ts_raw])),
            "Reward Min": float(np.min([x["reward"] for x in ts_raw])),
        }
        raw_results["simple_ts"] = ts_raw

    baseline_data = (embeddings, meta) if embeddings is not None else ds
    reward_fn = build_reward_fn(cfg, objectives, cat_map, class_indices)

    active_methods = {
        "abtest":  cfg.abtest,
        "ucb":     cfg.ucb,
        "bandit":  (cfg.MAB and embeddings is not None),
        "thompson": cfg.thompson,
        "epsilon": cfg.epsilon_greedy,
        "random":  cfg.random_baseline,
    }

    METHOD_DISPATCH = {
        "random": (
            lambda: (RandomBaseline(cfg).offline_fit(baseline_data), None),
            lambda bl, _: bl.online_simulate(cfg.num_iterations)
        ),
        "abtest": (
            lambda: (ABTestBaseline(cfg).offline_fit(ds), None),
            lambda bl,_: bl.online_simulate(cfg.num_iterations)     
        ),
        "ucb": (
            lambda: (UCBBaseline(cfg).offline_fit(ds), None),
            lambda bl,_: bl.online_simulate(cfg.num_iterations)  
        ),
       "thompson": (
            lambda: (ThompsonBaseline(cfg).offline_fit(ds), None),
            lambda bl, _: bl.online_simulate(cfg.num_iterations)   # returns (metrics, raw)
        ),
        "epsilon": (
            lambda: (EpsilonGreedyBaseline(cfg).offline_fit(ds), None),
            lambda bl,_: bl.online_simulate(cfg.num_iterations, return_raw=True)
        ),
        "bandit": (
            lambda: (load_linucb(
                        getattr(cfg, "bandit_ckpt", "baselines/bandit/linucb.pt"),
                        dim=embeddings.shape[1],
                        alpha=getattr(cfg, "bandit_alpha", 1.0)
                     ), None),
            lambda pol,_: evaluate_bandit(                          
                        pol, embeddings, meta, reward_fn,
                        objectives, class_indices, cfg.subset_size)
        ),
    }


    for name, (train_fn, eval_fn) in METHOD_DISPATCH.items():
        if not active_methods.get(name):
            continue
        cfg.out_dir = f"baselines/{name}"   
        obj, _ = train_fn()                 
        res    = eval_fn(obj, None)
        
        if isinstance(res, tuple):
            metrics, raw = res
            metrics_map[name] = metrics
            raw_results[name] = raw
        elif isinstance(res, list):
            raw_results[name] = res
            final = np.mean([r['fraction_relevant'] for r in res
                              if r['visit'] == cfg.num_iterations-1])
            metrics_map[name] = {"Reward Mean": float(final)}
        else:
            metrics_map[name] = res

    log_comparison(metrics_map)

    ctr_curves = {}
    regret_curves = {}
    for nm, raw in raw_results.items():
        df = pd.DataFrame(raw)

        if "fraction_relevant" in df.columns:
            ctr_avg = df.groupby("visit", as_index=False)["fraction_relevant"].mean()
            ctr_curves[nm] = ctr_avg

        if "reward" in df.columns:
            # On garde les rewards bruts (par visit)
            df_sorted = df.sort_values(["iteration", "visit"])
            grouped = df_sorted.groupby("visit", as_index=False)
            # reward moyen par timestep (across iterations)
            avg_reward_per_step = grouped["reward"].mean()
            # reconstruction cumulative
            avg_reward_per_step["cumulative_reward"] = avg_reward_per_step["reward"].cumsum()
            regret_curves[nm] = avg_reward_per_step.rename(columns={"cumulative_reward": "cum_reward"})

    styles = {"ucb": "r-", "thompson": "g--", "epsilon": "b-",
            "abtest": "y--", "random": "k:"}
    labels = {"ucb": "UCB", "thompson": "Thompson Sampling",
            "epsilon": r"$\epsilon$‑Greedy", "abtest": "A/B Test",
            "random": "Random"}

    if ctr_curves:
        fig_ctr = plot_fraction_relevant_curves(ctr_curves, styles, labels)
        wandb.log({"Plots/CTR": wandb.Image(fig_ctr)})
        plt.close(fig_ctr)

    if regret_curves:
        item_ctr = ds_filtered.groupby("productId")["reward"].mean()
        print(item_ctr.sort_values(ascending=False).tail(10))
        best_ctr = item_ctr.max()
        fig_regret = plot_cumulative_regret(regret_curves, best_ctr, styles, labels)
        wandb.log({"Plots/Regret": wandb.Image(fig_regret)})
        plt.close(fig_regret)

    
    wandb.finish()

if __name__ == "__main__":
    main()