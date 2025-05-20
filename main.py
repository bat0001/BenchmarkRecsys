from __future__ import annotations
import argparse, pandas as pd, wandb
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

    active = {
        "abtest":   cfg.bandit.baselines.abtest,
        "ucb":      cfg.bandit.baselines.ucb,
        "thompson": cfg.bandit.baselines.thompson,
        "epsilon":  cfg.bandit.baselines.epsilon,
        "random":   cfg.bandit.baselines.random,
    }

    cfg.baselines = [k for k, v in active.items() if v]
    metrics_map : Dict[str, Dict[str, float]] = {}
    raw_results : Dict[str, list[dict]]       = {}

    for name, on in active.items():
        if not on:
            continue
        bl_cls   = BASELINE_REGISTRY[name]

        

        baseline = bl_cls(cfg).offline_fit(canon_df)
        res = baseline.online_simulate(cfg.num_iterations)

        if isinstance(res, tuple):
            m, raw = res
        else:
            m, raw = res, []

        metrics_map[name] = m
        if raw:
            raw_results[name] = raw

    metric_classes = get_metric_classes()

    local_metrics   = {n: c for n, c in metric_classes.items()
                    if not getattr(c, "global_metric", False)}
    global_metrics  = {n: c for n, c in metric_classes.items()
                    if getattr(c, "global_metric", False)}

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
        seq_view = SequenceView(raw_results, canon_df)
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
            "random":   "Random"
        }
        fig_ctr = plot_fraction_relevant_curves(ctr_curves, styles, labels)
        wandb.log({"CTR Curves": wandb.Image(fig_ctr)})

    wandb.finish()


if __name__ == "__main__":
    main()