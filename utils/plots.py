import itertools
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt


def plot_reward_curves(histories: dict[str, list[float]], title: str):
    """
    histories = {"bandit": [r1, r2, ...], "random": [...]}
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in histories.items():
        ax.plot(range(1, len(ys)+1), ys, label=name)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Reward (mean batch)")
    ax.set_title(title)
    ax.legend()
    return fig

def plot_selection_overlap(idx_a: torch.Tensor, idx_b: torch.Tensor, pool_size: int):
    set_a, set_b = set(idx_a.tolist()), set(idx_b.tolist())
    inter = len(set_a & set_b)
    only_a = len(set_a) - inter
    only_b = len(set_b) - inter
    labels = ["Unique A", "Shared", "Unique B"]
    sizes  = [only_a, inter, only_b]

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, labels=labels, autopct="%.1f%%", startangle=90)
    ax.set_title("Overlap of selected images")
    return fig

def plot_entropy_vs_reward(metrics_map: dict[str, dict[str, float]]):
    """
    Scatter : Sequence Entropy (x) vs RewardMean (y) pour chaque méthode.
    Ignore methods without both metrics.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted = False
    for name, m in metrics_map.items():
        seq_ent = m.get("Sequence Entropy")
        reward = m.get("Reward Mean")
        # skip if missing
        if seq_ent is None or reward is None:
            continue
        ax.scatter(seq_ent, reward, label=name, s=60)
        ax.annotate(name, (seq_ent, reward),
                    textcoords="offset points", xytext=(4, 4))
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, 'No data for entropy vs reward',
                ha='center', va='center')
    ax.set_xlabel("Sequence entropy (bits)")
    ax.set_ylabel("Reward Mean")
    ax.set_title("Diversity vs Quality")
    ax.grid(True, alpha=.3)
    return fig


_DEFAULT_STYLES = ["r-", "g--", "b-.", "c:", "m-", "y--", "k-"]

def plot_fraction_relevant_curves(
    curves: dict[str, "pd.DataFrame"],
    *,
    labels: dict[str, str] | None = None,
    styles: dict[str, str] | None = None,
    title: str = "Percentage of Liked Recommendations",
    xlabel: str = "Recommendation #",
    ylabel: str = "% of Recs Clicked",
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Plot fraction_relevant vs visit for an arbitrary number of baselines.

    Parameters
    ----------
    curves  : mapping name -> DataFrame with columns ['visit', 'fraction_relevant']
    labels  : optional mapping name -> legend label (defaults to `name`)
    styles  : optional mapping name -> matplotlib linestyle (defaults to cyclic palette)
    title   : figure title
    xlabel  : x‑axis label
    ylabel  : y‑axis label (displayed as percentage)
    figsize : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    labels = labels or {}
    styles = styles or {}

    style_cycle = itertools.cycle(_DEFAULT_STYLES)

    fig, ax = plt.subplots(figsize=figsize)

    for name, df in curves.items():
        style = styles.get(name, next(style_cycle))
        lbl   = labels.get(name, name)
        ax.plot(df["visit"], df["fraction_relevant"],
                style, linewidth=3.0, label=lbl)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # convert y‑ticks to %
    ticks = ax.get_yticks()
    ax.set_yticklabels((ticks * 100).astype(int))

    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig