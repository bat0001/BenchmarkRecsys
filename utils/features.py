import pandas as pd, numpy as np, re, warnings
from sklearn.preprocessing   import MinMaxScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

_NUM_CUTOFF_CAT = 100          
_TXT_MIN_LEN    = 100          

def _is_floatish(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def build_feature_matrix(df: pd.DataFrame,
                         use_cols: list[str] | None = None,
                         dropna: bool = True) -> np.ndarray:
    if use_cols is None:
        use_cols = [c for c in df.columns if c.startswith("meta_")]

    mats = []
    for col in use_cols:
        s = df[col]
        if s.dtype.kind in "biufc" or all(_is_floatish(x) for x in s.head(20)):
            arr = s.astype(float).to_numpy().reshape(-1, 1)
            mats.append(MinMaxScaler().fit_transform(arr))
        else:
            uniq = s.fillna("NA").unique()
            if len(uniq) <= _NUM_CUTOFF_CAT and all(len(x) < _TXT_MIN_LEN for x in uniq):
                enc = OneHotEncoder(handle_unknown="ignore", sparse=True)
                mats.append(enc.fit_transform(s.fillna("NA").array.reshape(-1, 1)))
            else:
                vect = TfidfVectorizer(stop_words="english",
                                       max_features=2_000)
                mats.append(vect.fit_transform(s.fillna("").values))

    if not mats:
        warnings.warn("No meta cols exploitable -- return zeros")
        X = np.zeros((len(df), 1))
    else:
        X = sparse.hstack(mats).tocsr() if any(sparse.issparse(m) for m in mats) \
            else np.hstack(mats)

    if dropna:
        mask = np.array(X.sum(axis=1)).ravel() > 0
        if not mask.all():
            X = X[mask]
            warnings.warn(f"{(~mask).sum()} items without features have been ignored.")

    return X