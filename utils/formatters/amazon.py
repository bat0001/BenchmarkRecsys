import pandas as pd
from utils.formatter.base import BaseFormatter
from utils.formatter.registry import register

@register("AMAZON")
class AmazonFormatter(BaseFormatter):
    def __call__(self, raw_csv: pd.DataFrame, cfg):
        df = raw_csv.rename(columns={
            "productId": "productId",
            "userId":    "userId",
            "rating":    "rating"
        })
        # binary reward
        df["reward"] = (df["rating"] > cfg.reward_threshold).astype(int)
        return df[["productId", "userId", "reward"]]