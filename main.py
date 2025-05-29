from __future__ import annotations
import argparse, pandas as pd, wandb, numpy as np
from typing     import Dict
from wandb      import Table
from tabulate   import tabulate

from utils.config                import get_config
from utils.seed                  import set_seed
from utils.logger                import init_wandb_
from utils.formatters.registry   import get as get_formatter
from utils.metrics.registry      import get_all as get_metric_classes
from utils.metrics.base          import SequenceView    
from datasets.loader             import DATASET_FACTORY
from baselines                   import BASELINE_REGISTRY
from utils.plots                 import plot_fraction_relevant_curves
from baselines.core.baseline     import OnlineMixin
from utils.data_split import train_test_split_by_user
from utils.user_profile import build_user_profiles

def log_comparison(metrics: Dict[str, Dict[str, float]]):
    all_keys = sorted({k for m in metrics.values() for k in m.keys()})
    headers  = ["Method"] + all_keys

    table = Table(columns=headers)
    wb = {}
    for name, m in metrics.items():
        row = [name]
        for k in all_keys:
            val = m.get(k, None)
            if isinstance(val, (int, float)):
                row.append(f"{val:.4f}")
                wb[f"{name}/{k}"] = val
            else:
                row.append(str(val))
        table.add_data(*row)

    wandb.log({"Comparison Table": table})
    wandb.log(wb)

    print(tabulate(table.data, headers=headers, tablefmt="github"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", "-n", type=int, default=None,
                   help="Nombre de visites online")
    args, _ = p.parse_known_args()

    cfg = get_config()
    if args.iterations is not None:
        cfg.num_iterations = args.iterations
    set_seed(cfg.seed)
    init_wandb_(cfg)

    raw_df, objectives, class_names, class_idx = DATASET_FACTORY[cfg.dataset](cfg)
    formatter = get_formatter(cfg.dataset)
    canon_df  = formatter(raw_df, cfg)  

    n_users = 1000
    user_col = "user_id"

    if user_col in canon_df.columns:
        sampled_users = canon_df[user_col].dropna().unique()
        sampled_users = np.random.default_rng(cfg.seed).choice(sampled_users, size=n_users, replace=False)
        canon_df = canon_df[canon_df[user_col].isin(sampled_users)].reset_index(drop=True)
        print(f"✅ canon_df réduit à {len(canon_df)} interactions sur {n_users} utilisateurs")
    else:
        raise ValueError(f"Colonne '{user_col}' introuvable dans canon_df.")

    # print(canon_df.user_id.value_counts())
    item_key  = formatter.item_key

    active = {
        "abtest":   cfg.bandit.baselines.abtest,
        "ucb":      cfg.bandit.baselines.ucb,
        "thompson": cfg.bandit.baselines.thompson,
        "epsilon":  cfg.bandit.baselines.epsilon,
        "random":   cfg.bandit.baselines.random,
        "linucb":   cfg.bandit.baselines.linucb,
        "bpr":      cfg.bandit.baselines.bpr,
        "neuMF":    cfg.bandit.baselines.neuMF
    }

    cfg.baselines = [k for k, v in active.items() if v]
    metrics_map : Dict[str, Dict[str, float]] = {}
    raw_results : Dict[str, list[dict]]       = {}
    
    train_df, test_df = train_test_split_by_user(
        canon_df, user_col="user_id",
        n_test=getattr(cfg, "n_test_per_user", 5),
        seed=cfg.seed)

    user_profiles = build_user_profiles(
            train_df,
            user_col  ="user_id",
            title_col ="title",
            reward_col="reward")


    for name, on in active.items():
        if not on:
            continue

        baseline = BASELINE_REGISTRY[name](cfg)

        # met_off, raw_off = baseline.fit(canon_df)
        met_off, raw_off = baseline.fit(train_df) 
        if isinstance(baseline, OnlineMixin):
            met_onl, raw_onl = baseline.online_simulate(cfg.num_iterations)
            met_off.update(met_onl)
            raw_off = raw_onl if raw_onl else raw_off

        metrics_map[name] = met_off
        if raw_off:
            raw_results[name] = raw_off

    metric_classes = get_metric_classes()

    local_metrics   = {n: c for n, c in metric_classes.items()
                    if not getattr(c, "global_metric", False)}
    global_metrics  = {n: c for n, c in metric_classes.items()
                    if getattr(c, "global_metric", False)}

    print(f'Local metrics -- {local_metrics}')
    print(f'Global metrics -- {global_metrics}')
    for bl_name, raw in raw_results.items():
        metrics_map.setdefault(bl_name, {})
        for metric_name, MetricCls in local_metrics.items():
            metric_obj = MetricCls(cfg)                       
            score      = metric_obj(canon_df, raw, cfg)          
            if isinstance(score, dict):
                metrics_map[bl_name].update(score)
            else:
                metrics_map[bl_name][metric_name] = score

    if global_metrics:
        # seq_view  = SequenceView(raw_results, canon_df, item_key=item_key)
        seq_view = SequenceView(
            raw_results,
            test_df,                 # dataframe de référence = TEST
            item_key=item_key,
            user_profiles=user_profiles
        )

        for metric_name, MetricCls in global_metrics.items():
            metric_obj = MetricCls(cfg)                       
            scores = metric_obj(seq_view, cfg)               
            for bl, val in scores.items():
                metrics_map.setdefault(bl, {})[metric_name] = val    

    log_comparison(metrics_map)

    ctr_curves = {}
    for name, raw in raw_results.items():
        df = pd.DataFrame(raw)
        if "fraction_relevant" in df:
            ctr_curves[name] = df.groupby("visit", as_index=False)["fraction_relevant"].mean()

    if ctr_curves:
        styles = {
            "abtest":   "y--", "ucb": "r-", "thompson": "g--",
            "epsilon":  "b-",  "random": "k:"
        }
        labels = {
            "abtest":   "A/B Test",
            "ucb":      "UCB",
            "thompson": "Thompson Sampling",
            "epsilon":  r"$\epsilon$-Greedy",
            "linucb": "linucb",
            "random":   "Random"
        }
        fig_ctr = plot_fraction_relevant_curves(ctr_curves, styles, labels)
        wandb.log({"CTR Curves": wandb.Image(fig_ctr)})

    wandb.finish()


if __name__ == "__main__":
    main()