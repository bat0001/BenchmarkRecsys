from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

class BaseMetric(ABC):
    @abstractmethod
    def requires_predictions(self) -> bool:
        """True si la métrique a besoin de y_pred (ex : précision)."""

    @abstractmethod
    def __call__(self,
                 canon_df: pd.DataFrame,
                 y_pred:   pd.Series | None,
                 cfg) -> Tuple[str, float]:
        """Retourne (nom, valeur)."""