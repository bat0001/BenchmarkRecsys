
import pandas as pd

from pathlib import Path

from datasets.datasets import prepare_dataset
from utils.filter import filter_dataset_all_classes, filter_coco_inplace
from utils.utils_coco import parse_objectives_coco

def _load_coco(cfg):
    ds, _ = prepare_dataset("COCO")
    objectives = parse_objectives_coco(cfg.target_classes_coco)
    ds = filter_coco_inplace(ds, objectives, retaining_ratio=0.01)
    cat_ids = sorted(ds.coco.getCatIds())
    class_names = [c["name"] for c in ds.coco.loadCats(cat_ids)]
    class_indices = {n: i for i, n in enumerate(class_names)}
    return ds, objectives, class_names, class_indices

def _load_cifar(cfg):
    ds, _ = prepare_dataset(cfg.dataset)
    num = getattr(cfg, "num_per_class", 200)
    ds = filter_dataset_all_classes(ds, num_per_class=num)
    objectives = cfg.target_classes
    class_indices = {n: ds.classes.index(n) for n in objectives}
    return ds, objectives, ds.classes, class_indices

def _load_amazon(cfg):
    df = pd.read_csv(cfg.data.amazon_path,
                     names=["userId", "productId", "rating", "timestamp"])

    if cfg.data.amazon_subset:
        df = df.sample(n=cfg.data.amazon_subset, random_state=cfg.seed)

    reward_thr = getattr(cfg, "reward_threshold", 4)
    df["reward"] = (df["rating"] > reward_thr).astype(int)

    return df, None, None, None 

DATASET_FACTORY = {
    "COCO": _load_coco,
    "CIFAR-10": _load_cifar,
    "CIFAR-100": _load_cifar,
    "AMAZON": _load_amazon
}