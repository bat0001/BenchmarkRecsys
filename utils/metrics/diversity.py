import numpy as np, pandas as pd
from utils.metrics.base      import BaseMetric
from utils.metrics.registry  import register

@register("ItemDiversity")
class DiversityMetric(BaseMetric):
    def requires_predictions(self): return False
    def __call__(self, canon_df: pd.DataFrame, y_pred, cfg):
        pass
        # uniq = canon_df["productId"].nunique()
        # return "Item Diversity", uniq / len(canon_df)