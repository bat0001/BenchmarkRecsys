import pandas as pd
from collections import defaultdict

def build_user_profiles(
        df:            pd.DataFrame,
        *,
        user_col:      str = "user_id",
        item_col:      str = "item_id",
        title_col:     str = "title",
        reward_col:     str = "reward",
        like_thr:      float = 4.0,
        dislike_thr:   float = 2.0,
        max_items:     int   = 10,
):
    profiles = defaultdict(lambda: dict(likes=[], dislikes=[]))

    print(df.columns)
    print(df.reward.value_counts())
    liked    = df[df[reward_col] >= like_thr]
    disliked = df[df[reward_col] <= dislike_thr]

    for u, sub in liked.groupby(user_col):
        profiles[u]["likes"] = (
            sub.sort_values(reward_col, ascending=False)[title_col]
               .head(max_items).tolist()
        )
    for u, sub in disliked.groupby(user_col):
        profiles[u]["dislikes"] = (
            sub.sort_values(reward_col)[title_col]
               .head(max_items).tolist()
        )
    for p in profiles.values():
        p.setdefault("likes",    [])
        p.setdefault("dislikes", [])
    return profiles