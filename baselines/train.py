import argparse
from baselines import BASELINE_REGISTRY

from utils.config import get_config, prepare_dataset_and_embeddings   

def get_local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit_alpha", type=float, default=1.0, help="Exploration parameter for LinUCB")
    return parser.parse_known_args()[0]

cfg = get_config()

cfg.baseline = "linucb"

local_args = get_local_args()
cfg.bandit_alpha = local_args.bandit_alpha

ds, cfg.embeddings, cfg.reward_fn = prepare_dataset_and_embeddings(cfg)

BaselineCls = BASELINE_REGISTRY[cfg.baseline]
baseline = BaselineCls(cfg)

baseline.train(ds)
metrics = baseline.evaluate()
print(metrics)