from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from baselines.core.baseline import BaseBaseline


class _PointwiseDataset(Dataset):
    def __init__(self,
                 user_pos: dict[int, list[int]],
                 n_items:  int,
                 n_neg:    int = 4,
                 n_steps:  int = 100_000):
        self.user_pos, self.n_items, self.n_neg, self.n_steps = \
            user_pos, n_items, n_neg, n_steps
        self.users = list(user_pos)

    def __len__(self): return self.n_steps

    def __getitem__(self, idx):
        rng  = np.random
        u    = rng.choice(self.users)
        pos  = rng.choice(self.user_pos[u])
        batch_u   = [u]
        batch_i   = [pos]
        batch_lab = [1]
        for _ in range(self.n_neg):
            while True:
                neg = rng.randint(self.n_items)
                if neg not in self.user_pos[u]:
                    break
            batch_u.append(u); batch_i.append(neg); batch_lab.append(0)
        return torch.tensor(batch_u), torch.tensor(batch_i), torch.tensor(batch_lab)


class _NeuMF(nn.Module):
    def __init__(self,
                 n_users: int,
                 n_items: int,
                 mf_dim:  int = 16,
                 mlp_dim: int = 32,
                 mlp_layers: list[int] = (64, 32, 16, 8),
                 dropout: float = 0.0):
        super().__init__()
        self.user_mf = nn.Embedding(n_users, mf_dim)
        self.item_mf = nn.Embedding(n_items, mf_dim)
        self.user_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_mlp = nn.Embedding(n_items, mlp_dim)

        dims = [2 * mlp_dim] + list(mlp_layers)
        mlp = []
        for i in range(len(dims) - 1):
            mlp += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ]
        self.mlp_layers = nn.Sequential(*mlp)
        self.mf_train  = True  
        self.mlp_train = True 
        self.predict = nn.Linear(mf_dim + dims[-1], 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u, i):
        mf_vec = self.user_mf(u) * self.item_mf(i)        
        mlp_in  = torch.cat([self.user_mlp(u), self.item_mlp(i)], dim=-1)
        mlp_vec = self.mlp_layers(mlp_in)                 
        if self.mf_train and self.mlp_train:
            vec = torch.cat([mf_vec, mlp_vec], dim=-1)    
        elif self.mf_train:
            vec = mf_vec
        else:
            vec = mlp_vec
        return self.predict(vec).squeeze(-1) 


class NeuMFBaseline(BaseBaseline):
    def fit(self, df: pd.DataFrame, *,
            user_col="user_id", item_col="item_id", reward_col="reward",
            min_inter=5, mf_dim=None, mlp_dim=None,
            epochs=None, batch=None, n_neg=4):

        cfg      = self.cfg
        mf_dim   = mf_dim  or getattr(cfg, "neumf_mf_dim", 16)
        mlp_dim  = mlp_dim or getattr(cfg, "neumf_mlp_dim", 32)
        epochs   = epochs  or getattr(cfg, "neumf_epochs",  10)
        batch    = batch   or getattr(cfg, "neumf_batch",  1024)
        lr       = getattr(cfg, "neumf_lr", 1e-3)
        top_k    = getattr(cfg, "eval_topk", 10)

        if {"title", "genres"}.issubset(df.columns):
            self.item_meta = (
                df.drop_duplicates(item_col)
                  .set_index(item_col)[["title", "genres"]]
                  .to_dict(orient="index")
            )
        else:
            self.item_meta = {}

        pos = df[df[reward_col] == 1].copy()
        keep_u = pos[user_col].value_counts()
        keep_u = keep_u[keep_u >= min_inter].index
        pos = pos[pos[user_col].isin(keep_u)]

        self.user_map     = {u: i for i, u in enumerate(sorted(pos[user_col].unique()))}
        self.item_map     = {v: i for i, v in enumerate(sorted(pos[item_col].unique()))}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        pos["u_idx"] = pos[user_col].map(self.user_map)
        pos["i_idx"] = pos[item_col].map(self.item_map)

        n_users, n_items = len(self.user_map), len(self.item_map)

        user_pos = pos.groupby("u_idx")["i_idx"].apply(list).to_dict()
        steps = max(len(pos) * (1 + n_neg), 50_000)

        loader = DataLoader(
            _PointwiseDataset(user_pos, n_items, n_neg, steps),
            batch_size=batch, shuffle=True, drop_last=True
        )

        self.model = _NeuMF(n_users, n_items, mf_dim, mlp_dim).to(self.device)
        opt  = torch.optim.Adam(self.model.parameters(), lr=lr)
        bce  = nn.BCEWithLogitsLoss()

        self.model.train()
        for ep in range(1, epochs + 1):
            losses = []
            for u, i, lab in tqdm(loader, leave=False,
                                  desc=f"[NeuMF] epoch {ep}/{epochs}"):
                u, i, lab = (t.view(-1).to(self.device) for t in (u, i, lab.float()))
                loss = bce(self.model(u, i), lab)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
            print(f"[NeuMF] epoch {ep:02d} | loss = {np.mean(losses):.4f}")

        # raw_logs = self._offline_log_sequences(
        #     top_k=top_k,
        #     visit_id=cfg.num_iterations - 1
        # )
        # return {}, raw_logs
        return {} 

    @torch.inference_mode()
    def rank(self, user_ids: torch.Tensor, k: int | None = None) -> torch.Tensor:
        u_idx = torch.tensor(
            [self.user_map[u.item()] for u in user_ids if u.item() in self.user_map],
            device=self.device
        )
        if len(u_idx) == 0:
            k = k or 0
            return torch.empty((0, k), dtype=torch.long)

        scores = self.model(
            u_idx.repeat_interleave(len(self.item_map)),
            torch.arange(len(self.item_map), device=self.device).repeat(len(u_idx))
        ).view(len(u_idx), -1)

        k = k or scores.size(1)
        _, topk_idx = torch.topk(scores, k, dim=1)
        return topk_idx.cpu().apply_(lambda x: self.inv_item_map[int(x)])

    def _offline_log_sequences(self, *, top_k:int = 10, visit_id:int = 0):
        """A record = (visit, user_id, item_key, title, genres)"""
        self.model.eval()
        logs = []

        for u_raw in self.user_map:                    
            rec_items = self.rank(torch.tensor([u_raw]), k=top_k)[0].tolist()

            for it in rec_items:
                meta = self.item_meta.get(it, {"title": "", "genres": ""})
                logs.append(
                    {
                        "visit":    visit_id,
                        "user_id":  int(u_raw),
                        "item_key": int(it), 
                        "title":    meta["title"],
                        "genres":   meta["genres"],
                    }
                )
        return logs
    
    @torch.inference_mode()
    def predict_sequences(self,
                          test_df:  pd.DataFrame,
                          *,
                          top_k:   int,
                          visit_id: int = 0) -> list[dict]:
        """
        Génèrate for each user in `test_df`
        the top‑k list, and return a list[dict] that works with the LLM.
        """
        self.model.eval()
        logs = []

        users_raw = sorted(test_df["user_id"].unique())
        for u_raw in users_raw:
            if u_raw not in self.user_map:       
                continue

            rec_items = self.rank(
                torch.tensor([u_raw]), k=top_k
            )[0].tolist()

            for it in rec_items:
                meta = self.item_meta.get(it, {"title": "", "genres": ""})
                logs.append({
                    "visit":    visit_id,
                    "user_id":  int(u_raw),
                    "item_key": int(it),
                    "title":    meta["title"],
                    "genres":   meta["genres"],
                })
        return logs