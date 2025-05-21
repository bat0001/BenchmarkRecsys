from baselines.abtest.abtest_baseline import ABTestBaseline
from baselines.ucb.ucb_baseline     import UCBBaseline
from baselines.bandit.linucb_baseline import LinUCBBaseline
from baselines.gfn.gfn_baseline import ClassicalGFNBaseline
from baselines.thompson.baseline import ThompsonBaseline
from baselines.epsilon.baseline import EpsilonGreedyBaseline
from baselines.linucb.baseline import LinUCBBaseline

BASELINE_REGISTRY = {
    # "gfn_classical": ClassicalGFNBaseline,
    # "abtest": ABTestBaseline,
    "thompson": ThompsonBaseline,
    # "epsilon":   EpsilonGreedyBaseline,
    "ucb":    UCBBaseline,
    "linucb":     LinUCBBaseline,
}