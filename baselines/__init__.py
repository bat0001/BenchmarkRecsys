from baselines.bandit.linucb_baseline import LinUCBBaseline
from baselines.gfn.gfn_baseline import ClassicalGFNBaseline

BASELINE_REGISTRY = {
    "linucb": LinUCBBaseline,
    "gfn_classical": ClassicalGFNBaseline,
}