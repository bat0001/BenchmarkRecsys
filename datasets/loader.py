
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
    csv_path = Path(cfg.amazon_path)
    df = pd.read_csv(csv_path, header=0)

    df = df.iloc[:, :4]
    df.columns = ["userId", "productId", "rating", "timestamp"]

    objectives = {}
    class_names  = df["productId"].unique().tolist()
    class_indices = {p: i for i, p in enumerate(class_names)}
    return df, objectives, class_names, class_indices

DATASET_FACTORY = {
    "COCO": _load_coco,
    "CIFAR-10": _load_cifar,
    "CIFAR-100": _load_cifar,
    "AMAZON": _load_amazon
}