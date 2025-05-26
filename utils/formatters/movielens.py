import pandas as pd
from utils.formatters.base import BaseFormatter
from utils.formatters.registry import register

@register("MOVIELENS20M")
class MovieLensFormatter(BaseFormatter):
    """
    Canonical columns after formatting
    ----------------------------------
    item_id   : movieId (int)
    user_id   : userId  (int)
    reward    : 0 / 1   (binary click‑like)
    meta.*    : any extra useful columns (title, genres …)
    """

    def __call__(self, raw_dfs: dict[str, pd.DataFrame], cfg):
        ratings = raw_dfs["ratings"]                 
        movies  = raw_dfs["movies"]                  

        #TODO maybe change this binary reward (≥ threshold)
        thr = float(cfg.data.get("reward_threshold", 4.0))
        ratings["reward"] = (ratings["rating"] >= thr).astype(int)

        canon = ratings[["movieId", "userId", "reward"]].rename(
            columns={"movieId": "item_id", "userId": "user_id"}
        )

        canon = canon.merge(
            movies[["movieId", "title", "genres"]], how="left",
            left_on="item_id", right_on="movieId"
        ).drop(columns="movieId")                      

        return canon