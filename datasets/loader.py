
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

def _load_amazon_sales(cfg):
    csv_path = Path(cfg.data.amazon_sales_path).expanduser()
    df = pd.read_csv(csv_path)

    if cfg.data.amazon_sales_subset > 0:
        df = df.sample(cfg.data.amazon_sales_subset, random_state=cfg.core.seed)

    objectives, class_names, class_idx = {}, [], {}
    return df, objectives, class_names, class_idx

def _load_movielens20m(cfg):
    path = cfg.data.movielens_root        
    dfs  = {
        "ratings": pd.read_csv(f"{path}/rating.csv"),
        "movies" : pd.read_csv(f"{path}/movie.csv"),
        "tags": pd.read_csv(f'{path}/tag.csv')
    }
    return dfs, None, None, None          

def _load_movielens1m(cfg):
    """
    Reads MovieLens‑1M .dat files ( ‘::’‑separated ) and returns
    dict[str, pd.DataFrame] compatible with the formatter above.
    """

    path = cfg.data.movielens_root.rstrip("/")

    ratings = pd.read_csv(
        f"{path}/ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        f"{path}/movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    dfs = {
        "ratings": ratings,
        "movies":  movies,
    }

    return dfs, None, None, None

DATASET_FACTORY = {
    "COCO": _load_coco,
    "CIFAR-10": _load_cifar,
    "CIFAR-100": _load_cifar,
    "AMAZON": _load_amazon,
    "AMAZON_SALES": _load_amazon_sales,
    "MOVIELENS1M": _load_movielens1m,    
    "MOVIELENS20M": _load_movielens20m
}