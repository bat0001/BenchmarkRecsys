import pandas as pd
from utils.metrics.base      import BaseMetric
from utils.metrics.registry  import register

@register("CTR")
class CTRMetric(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg) 

    def requires_predictions(self): return False

    def __call__(self, canon_df: pd.DataFrame, y_pred, cfg):
        ctr = canon_df["reward"].mean()
        return "CTR", ctr