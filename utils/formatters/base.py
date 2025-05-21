from abc import ABC, abstractmethod
import pandas as pd

class BaseFormatter(ABC):
    """
    Converts an arbitrary raw dataset into a *canonical* DataFrame with
        • productId  (str | int)
        • userId     (str | int)
        • reward     (float)   0/1   or any scalar
        • Optional extra columns (embedding, timestamp, …)
    """
    item_key: str = NotImplemented
    
    @abstractmethod
    def __call__(self, raw: any, cfg) -> pd.DataFrame:
        ...