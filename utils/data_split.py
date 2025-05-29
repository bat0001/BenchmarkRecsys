import pandas as pd
import numpy as np

def train_test_split_by_user(
        df:            pd.DataFrame,
        *,
        user_col:      str = "user_id",
        n_test:        int = 5,
        shuffle:       bool = True,
        seed:          int  = 42,
):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []

    for _, user_df in df.groupby(user_col):
        idx = user_df.index.to_numpy()
        if shuffle:
            rng.shuffle(idx)
        test_idx  = idx[:n_test]
        train_idx = idx[n_test:]
        train_rows.extend(train_idx)
        test_rows.extend(test_idx)

    return df.loc[train_rows].reset_index(drop=True), \
           df.loc[test_rows ].reset_index(drop=True)