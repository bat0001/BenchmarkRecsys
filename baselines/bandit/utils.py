from __future__ import annotations
import torch
from utils.device import DEVICE
from baselines.bandit.linucb import LinUCB


def load_linucb(ckpt_path: str, dim: int, alpha: float = 1.0) -> LinUCB:
    """Load a LinUCB policy from *either* a full object pickle or a dict.

    We support two checkpoint formats because earlier training scripts
    sometimes saved the **whole LinUCB instance**, while the newer code
    only serialises the matrices `A` and `b` for safety.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint.
    dim : int
        Embedding dimension (needed only for the dict format).
    alpha : float, default 1.0
        Exploration hyper‑parameter used when we need to recreate a policy.

    Returns
    -------
    LinUCB
        A policy ready to call `.select()` on the proper device.
    """

    obj = torch.load(ckpt_path, map_location=DEVICE)

    if isinstance(obj, LinUCB):
        pol = obj
        pol.A = pol.A.to(DEVICE)
        pol.b = pol.b.to(DEVICE)
        pol.device = DEVICE  
        return pol

    if isinstance(obj, dict) and {"A", "b"}.issubset(obj):
        pol = LinUCB(dim=dim, alpha=alpha, device=DEVICE)
        pol.A.copy_(obj["A"].to(DEVICE))
        pol.b.copy_(obj["b"].to(DEVICE))
        return pol

    raise ValueError(
        f"Unsupported LinUCB checkpoint format {type(obj)} at {ckpt_path}.\n"
        "Expected a LinUCB object or a dict with keys 'A' and 'b'."
    )
