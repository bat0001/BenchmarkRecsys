import pandas as pd

from utils.formatters.base import BaseFormatter
from utils.formatters.registry import register

@register("AMAZON_SALES")
class AmazonSalesFormatter(BaseFormatter):
    def __call__(self, raw: pd.DataFrame, cfg) -> pd.DataFrame:
        df = raw.rename(columns={
            "product_id": "productId",
            "user_id":    "user_id",
            "rating":     "rating"
        })
        df["reward"] = (pd.to_numeric(df["rating"], errors="coerce")
                          > cfg.data.reward_threshold).astype(int)

        meta_cols = ["category", "discounted_price"]
        for c in meta_cols:
            if c in df.columns:
                df[f"meta_{c}"] = df[c]

        return df[["productId", "user_id", "reward"] +
                  [c for c in df.columns if c.startswith("meta_")]]