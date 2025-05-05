#!/usr/bin/env python3
import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from baselines import BASELINE_REGISTRY
from utils.config        import get_config
from datasets.loader     import DATASET_FACTORY
from models.backbone_factory import make_backbone
from utils.encoding      import encode_dataset
from utils.rewards       import build_reward_fn
from utils.utils_coco    import build_cat_name_to_id_map

def parse_cli():
    cfg        = get_config()                    
    parser     = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--baseline", required=True,
                        choices=BASELINE_REGISTRY.keys(),
                        help="Nom de la baseline à entraîner")
    parser.add_argument("--bandit_alpha", type=float, default=1.0)
    local, _ = parser.parse_known_args()
    cfg.baseline     = local.baseline
    cfg.bandit_alpha = local.bandit_alpha
    cfg.out_dir      = f"baselines/{cfg.baseline}"
    return cfg

def prepare_data(cfg):
    ds, objectives, class_names, class_idx = DATASET_FACTORY[cfg.dataset](cfg)
    backbone    = make_backbone(cfg)
    embeddings, meta = encode_dataset(backbone, ds, cfg)
    cat_map     = build_cat_name_to_id_map(ds.coco) if cfg.dataset == "COCO" else None
    reward_fn   = build_reward_fn(cfg, objectives, cat_map, class_idx)
    return embeddings, meta, reward_fn, objectives, class_idx

def main():
    cfg = parse_cli()
    emb, meta, reward_fn, objectives, class_idx = prepare_data(cfg)

    BaselineCls = BASELINE_REGISTRY[cfg.baseline]
    baseline    = BaselineCls(cfg)

    baseline.set_data(emb, meta, reward_fn, objectives, class_idx)

    baseline.train()
    print(f"{cfg.baseline} metrics:", baseline.evaluate())

if __name__ == "__main__":
    main()