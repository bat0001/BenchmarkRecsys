import pandas as pd
from utils.formatters.base import BaseFormatter
from utils.formatters.registry import register

@register("AMAZON_SALES")
class AmazonSalesFormatter(BaseFormatter):
    def __call__(self, raw_csv: pd.DataFrame, cfg):
        df = raw_csv.rename(
            columns={
                "product_id": "productId",
                "user_id":    "userId",
                "rating":     "rating",
            }
        )

        df["rating"] = (
            pd.to_numeric(df["rating"], errors="coerce")
              .fillna(0.0)
              .astype(float)
        )

        thr = cfg.data.reward_threshold
        df["reward"] = (df["rating"] > thr).astype(int)

        base_cols = ["productId", "userId", "reward"]
        extra_cols = [
            "category", "discounted_price", "actual_price", "discount_percentage",
            "rating_count", "about_product", "img_link"
        ]
        return df[base_cols + extra_cols]