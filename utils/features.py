import pandas as pd, numpy as np, re, warnings
from sklearn.preprocessing   import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse as sp

_NUM_CUTOFF_CAT = 100          
_TXT_MIN_LEN    = 100          

def _is_floatish(s):
    try:
        float(s)
        return True
    except Exception:
        return False

def build_feature_matrix(df: pd.DataFrame) -> sp.csr_matrix:
    """
    •  One‑hot‑encode genres               (max 20)
    •  Normalised log(#ratings)            (popularity)
    •  Mean user rating (z‑score)          (quality)
    Result → csr_matrix [N, d]
    """


    genre_lists = df["genres"].str.split("|")
    all_genres  = sorted({g for lst in genre_lists for g in lst})
    enc = OneHotEncoder(sparse=True, handle_unknown="ignore")
    G   = enc.fit_transform(genre_lists.apply(lambda l: [all_genres.index(g) for g in l]))

    pop  = np.log1p(df.groupby("item_id").size()).to_numpy().reshape(-1, 1)
    mean = df.groupby("item_id")["rating"].mean().fillna(0).to_numpy().reshape(-1, 1)
    scaler = StandardScaler(with_mean=False)
    X_num  = scaler.fit_transform(np.hstack([pop, mean]))    

    X = sp.hstack([G, sp.csr_matrix(X_num)], format="csr")
    return X