from __future__ import annotations
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch, wandb
from typing  import Dict, Tuple
from wandb   import Table
from tabulate import tabulate
from utils.device import DEVICE

from utils.formatters.registry import get as get_formatter
from utils.config       import get_config
from utils.seed         import set_seed
from utils.logger       import init_wandb_
from utils.rewards      import build_reward_fn
from utils.plots        import plot_fraction_relevant_curves, plot_cumulative_regret
from datasets.loader    import DATASET_FACTORY
from baselines          import BASELINE_REGISTRY        

import sys

def filter_df_for_bandit(df: pd.DataFrame,
                         *, item_col="productId",
                         reward_col="reward",
                         min_pos=5, max_items=5_000) -> pd.DataFrame:
    """Keep only items with ≥ min_pos positive rewards (binary `reward` col)."""
    keep = (df[df[reward_col] == 1][item_col]
            .value_counts()
            .loc[lambda s: s >= min_pos]
            .index[:max_items])
    return df[df[item_col].isin(keep)].copy()


def log_comparison(metrics: Dict[str, Dict[str, float]]):
    headers = ["Method", "RewardMean", "RewardMedian", "RewardMin", "RewardMax"]
    table   = Table(columns=headers)
    for k, v in metrics.items():
        row = [k] + [f"{v.get(x, 0):.3f}" for x in
                     ["Reward Mean", "Reward Median", "Reward Min", "Reward Max"]]
        table.add_data(*row)
    wandb.log({"Comparison": table})
    print(tabulate(table.data, headers=headers, tablefmt="github"))


def main() -> None:
    # parser = argparse.ArgumentParser()
   
    # args, _ = parser.parse_known_args()

    cfg = get_config()
    set_seed(cfg.seed)
    init_wandb_(cfg)

    ds, objectives, class_names, class_idx = DATASET_FACTORY[cfg.dataset](cfg)
    
    formatter  = get_formatter(cfg.dataset)
    canon_df   = formatter(ds, cfg) 

    sys.exit(0)
    if cfg.dataset == "AMAZON":
        if "reward" not in ds.columns:
            ds["reward"] = (ds["rating"] > cfg.reward_threshold).astype(int)
        ds = filter_df_for_bandit(ds, max_items=cfg.max_items)

    active: Dict[str, bool] = {
        "abtest":    cfg.abtest,
        "ucb":       cfg.ucb,
        "thompson":  cfg.thompson,
        "epsilon":   cfg.epsilon_greedy,
        "random":    cfg.random_baseline,
    }

    metrics_map: Dict[str, Dict[str, float]] = {}
    raw_results: Dict[str, list[dict]]      = {}

    for name, on in active.items():
        if not on:
            continue
        bl_cls    = BASELINE_REGISTRY[name]
        baseline  = bl_cls(cfg).offline_fit(ds)

    
        print(cfg.num_iterations)
        metrics, raw = baseline.online_simulate(cfg.num_iterations)

        metrics_map[name] = metrics
        raw_results[name] = raw  

    log_comparison(metrics_map)

    ctr_curves, reward_curves = {}, {}
    for nm, raw in raw_results.items():
        df = pd.DataFrame(raw)
        if "fraction_relevant" in df.columns:
            ctr_curves[nm] = df.groupby("visit", as_index=False)["fraction_relevant"].mean()
        if "reward" in df.columns:
            g = (df.sort_values(["iteration", "visit"])
                   .groupby("visit", as_index=False)["reward"].mean())
            g["cum_reward"] = g["reward"].cumsum()
            reward_curves[nm] = g

    styles = {"ucb": "r-", "thompson": "g--", "epsilon": "b-",
              "abtest": "y--", "random": "k:"}
    labels = {"ucb": "UCB", "thompson": "Thompson Samp.",
              "epsilon": r"$\epsilon$‑Greedy", "abtest": "A/B Test",
              "random": "Random"}

    if ctr_curves:
        wandb.log({"CTR": wandb.Image(plot_fraction_relevant_curves(
            ctr_curves, styles, labels))})

    if reward_curves:
        best_ctr = ds.groupby("productId")["reward"].mean().max()
        wandb.log({"Regret": wandb.Image(plot_cumulative_regret(
            reward_curves, best_ctr, styles, labels))})

    wandb.finish()


if __name__ == "__main__":
    main()