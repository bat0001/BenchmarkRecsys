import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import argparse
from baselines import BASELINE_REGISTRY
from utils.config import get_config
from utils.utils_coco import build_cat_name_to_id_map

from datasets.loader import DATASET_FACTORY
from models.backbone_factory import make_backbone
from utils.encoding import encode_dataset
from utils.rewards import build_reward_fn

def get_local_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bandit_alpha", type=float, default=1.0)
    return p.parse_known_args()[0]


def main():
    cfg = get_config()
    local = get_local_args()
    cfg.bandit_alpha = local.bandit_alpha
    cfg.baseline = "linucb"
    cfg.out_dir = "baselines/bandit"
    ds, objectives, _, class_idx = DATASET_FACTORY[cfg.dataset](cfg)
    backbone = make_backbone(cfg)
    embeddings, meta = encode_dataset(backbone, ds, cfg)
    cat_map = build_cat_name_to_id_map(ds.coco) if cfg.dataset == "COCO" else None
    cfg.reward_fn = build_reward_fn(cfg, objectives, cat_map, class_idx)
    cfg.objectives = objectives
    cfg.class_idx = class_idx

    BaselineCls = BASELINE_REGISTRY[cfg.baseline]
    baseline = BaselineCls(cfg)

    baseline.embeddings = embeddings.to(baseline.device)
    baseline.meta        = meta
    baseline.objectives  = objectives
    baseline.class_idx   = class_idx
    baseline.reward_fn   = cfg.reward_fn

    baseline.train(embeddings, meta)
    metrics = baseline.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
