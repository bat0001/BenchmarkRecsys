from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple


import pandas as pd

class BaseMetric(ABC):
    
    #TODO TMP PATCH 
    global_metric = False

    def __init__(self, cfg=None):
        self.cfg = cfg

    @abstractmethod
    def requires_predictions(self) -> bool:
        """True if métrique need y_pred (e.g. précision)."""

    @abstractmethod
    def __call__(self,
                 canon_df: pd.DataFrame,
                 y_pred:   pd.Series | None,
                 cfg) -> Tuple[str, float]:
        """Return (name, value)."""


@dataclass
class SequenceView:
    """
    Simple containeur given to global metrics:
        raw  : { baseline → list[dict] }  (online_simulate)
        meta : canon_df (optionnel)
    """
    def __init__(self, raw, canon_df, *, item_key, user_profiles=None):
        self.raw           = raw
        self.canon_df      = canon_df
        self.item_key      = item_key
        self.user_profiles = user_profiles or {}

class SequenceMetric(ABC):
    """Base for metrics which look final sqeuence / raw."""

    def __init__(self, cfg):
        self.cfg = cfg           

    @abstractmethod
    def __call__(self, seq_view: SequenceView, cfg) -> Dict[str, float] | float:
        ...