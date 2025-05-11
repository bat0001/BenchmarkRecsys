import pandas as pd
from utils.formatters.base import BaseFormatter
from utils.formatters.registry import register

@register("AMAZON")
class AmazonFormatter(BaseFormatter):
    def __call__(self, raw_csv: pd.DataFrame, cfg):
        df = raw_csv.rename(columns={
            "productId": "productId",
            "userId":    "userId",
            "rating":    "rating"
        })
        df["reward"] = (df["rating"] > cfg.reward_threshold).astype(int)

        pos_counts = df[df["reward"] == 1]["productId"].value_counts()
        keep_items = pos_counts[pos_counts >= cfg.amazon_min_pos].index
        if cfg.amazon_subset is not None:
            keep_items = keep_items[: cfg.amazon_subset]
        df = df[df["productId"].isin(keep_items)]

        return df[["productId", "userId", "reward"]]