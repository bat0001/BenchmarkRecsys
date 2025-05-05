# utils/plots.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

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