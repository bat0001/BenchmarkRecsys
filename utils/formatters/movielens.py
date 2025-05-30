from __future__ import annotations
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
        tags    = raw_dfs.get("tags")     

        thr = float(cfg.data.get("reward_threshold", 4.0))
        ratings = ratings.copy()
        ratings["reward"] = (ratings["rating"] >= thr).astype(int)

        canon = ratings.rename(
            columns={"movieId": "item_id", "userId": "user_id"}
        )[["item_id", "user_id", "reward"]]

        canon = canon.merge(
            movies[["movieId", "title", "genres"]]
                  .rename(columns={"movieId": "item_id"}),
            on="item_id",
            how="left",
        )

        if tags is not None and len(tags):
            tag_col = "tag" if "tag" in tags.columns else "review_title"

            COL_MAP = {
                "movieId":   "item_id",
                "productId": "item_id",
                "product_id": "item_id",
            }
            tags_agg = (
                tags                                     
                .rename(columns={k: v for k, v in COL_MAP.items() if k in tags.columns})
                .groupby("item_id")[tag_col]
                .apply(lambda s: "|".join(s.astype(str).str.lower().unique()))
                .reset_index(name="tags")
            )

            canon = canon.merge(tags_agg, on="item_id", how="left")
        canon[["title", "genres", "tags"]] = (
            canon[["title", "genres", "tags"]].fillna("")
        )

        return canon
    
@register("MOVIELENS1M")
class MovieLens1MFormatter(BaseFormatter):
    """
    Canonical columns after formatting
    ----------------------------------
    item_id   : movieId  (int)
    user_id   : userId   (int)
    reward    : 0 / 1    (binary like‑click)
    meta.*    : any extra useful columns (title, genres …)
    """

    def __call__(self, raw_dfs: dict[str, pd.DataFrame], cfg):
        ratings = raw_dfs["ratings"]      
        movies  = raw_dfs["movies"]      

        thr = float(cfg.data.get("reward_threshold", 4.0))
        ratings = ratings.copy()
        # ratings["reward"] = (ratings["rating"] >= thr).astype(int)
        ratings["reward"] = ratings["rating"].astype(float)      # garde 1 … 5
        ratings["reward"] = ratings["rating"] / 5.0 
        canon = ratings.rename(
            columns={"movieId": "item_id", "userId": "user_id"}
        )[["item_id", "user_id", "reward"]]

        canon = canon.merge(
            movies.rename(columns={"movieId": "item_id"}),
            on="item_id",
            how="left",
        )

        for col in ["title", "genres"]:
            if col in canon.columns:
                canon[col] = canon[col].fillna("")

        return canon
