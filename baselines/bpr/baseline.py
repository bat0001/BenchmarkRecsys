from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from baselines.core.baseline import BaseBaseline
from utils.device import DEVICE

class _PairwiseDataset(Dataset):
    def __init__(self,
                 user_pos: dict[int, list[int]],
                 n_items:  int,
                 n_steps:  int):
        self.user_pos = user_pos
        self.n_items  = n_items
        self.users    = list(user_pos)
        self.n_steps  = n_steps                     

    def __len__(self): return self.n_steps

    def __getitem__(self, idx):
        rng   = np.random
        u     = rng.choice(self.users)
        pos_i = rng.choice(self.user_pos[u])
        while True:                               
            neg_i = rng.randint(self.n_items)
            if neg_i not in self.user_pos[u]:
                break
        return u, pos_i, neg_i


class _BPRModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int):
        super().__init__()
        self.user_e = nn.Embedding(n_users, emb_dim)
        self.item_e = nn.Embedding(n_items, emb_dim)
        
        nn.init.xavier_normal_(self.user_e.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_e.weight, gain=1.0)

    def forward(self, u, i):               
        return (self.user_e(u) * self.item_e(i)).sum(-1)

    def triplet_loss(self, u, pos_i, neg_i):
        pos_s = self.forward(u,  pos_i)
        neg_s = self.forward(u,  neg_i)
        return -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()

class BPRBaseline(BaseBaseline):
    def fit( self,
             train_df:  pd.DataFrame,
             *,
             user_col:   str = "user_id",
             item_col:   str = "item_id",
             reward_col: str = "reward",
             min_inter:  int = 5,
             emb_dim:    int | None = None,
             epochs:     int | None = None,
             batch_size: int | None = None,
           ) -> tuple[dict, list[dict]]:

        cfg        = self.cfg
        emb_dim    = emb_dim    or getattr(cfg, "bpr_emb_dim", 64)
        epochs     = epochs     or getattr(cfg, "bpr_epochs",   15)
        batch_size = batch_size or getattr(cfg, "bpr_batch", 2048)
        lr         = getattr(cfg, "bpr_lr", 1e-3)
        top_k      = getattr(cfg, "eval_topk", 10)

        pos_df = train_df[train_df[reward_col] == 1].copy()
        active = pos_df[user_col].value_counts()
        pos_df = pos_df[pos_df[user_col].isin(active[active >= min_inter].index)]

        self.item_meta = (
            pos_df.drop_duplicates(item_col)
                  .set_index(item_col)[["title", "genres"]]
                  .to_dict("index")
            if {"title", "genres"}.issubset(pos_df.columns) else {}
        )

        self.user_map = {u: i for i, u in enumerate(sorted(pos_df[user_col].unique()))}
        self.item_map = {v: i for i, v in enumerate(sorted(pos_df[item_col].unique()))}
        self.inv_item_map = {v: k for k, v in self.item_map.items()}

        pos_df["u_idx"] = pos_df[user_col].map(self.user_map)
        pos_df["i_idx"] = pos_df[item_col].map(self.item_map)

        n_users, n_items = len(self.user_map), len(self.item_map)
        user_pos = pos_df.groupby("u_idx")["i_idx"].agg(list).to_dict()

        steps = max(len(pos_df) * 5, 20_000)
        loader = DataLoader(
            _PairwiseDataset(user_pos, n_items, steps),
            batch_size=batch_size, shuffle=True, drop_last=True
        )

        self.model = _BPRModel(n_users, n_items, emb_dim).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self._loss_history: list[float] = []
        self.model.train()
        for ep in range(1, epochs + 1):
            epoch_losses = []
            for u, pos_i, neg_i in tqdm(loader, leave=False,
                                        desc=f"[BPR] epoch {ep}/{epochs}"):
                u, pos_i, neg_i = (t.to(self.device) for t in (u, pos_i, neg_i))
                loss = self.model.triplet_loss(u, pos_i, neg_i)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            self._loss_history.append(mean_loss)
            print(f"[BPR] epoch {ep:02d} | loss = {mean_loss:.4f}")

        metrics: dict[str, float] = {
            "final_train_loss": self._loss_history[-1]
        }


        raw_logs = self.predict_sequences(
            df        = train_df,                     
            top_k     = top_k,
            visit_id  = cfg.num_iterations - 1
        )

        return metrics, raw_logs

    @torch.inference_mode()
    def predict_sequences(self,
                          df:        pd.DataFrame,
                          *,
                          top_k:    int = 10,
                          visit_id: int = 0) -> list[dict]:
        """
        Génère les TOP‑k pour **tous** les users présents dans `df`
        et renvoie une liste de logs utilisables par la métrique LLM.
        """
        self.model.eval()
        logs = []
        unique_users = df["user_id"].unique()

        for u_raw in unique_users:
            if u_raw not in self.user_map:          
                continue

            rec_items = self.rank(
                torch.tensor([u_raw]), k=top_k
            )[0].tolist()

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
    def rank(self, user_ids: torch.Tensor, k: int | None = None) -> torch.Tensor:
        keep = [u.item() for u in user_ids if u.item() in self.user_map]
        if not keep:
            k = k or 0
            return torch.empty((0, k), dtype=torch.long)

        u_idx = torch.tensor([self.user_map[u] for u in keep], device=self.device)
        scores = self.model.user_e(u_idx) @ self.model.item_e.weight.T
        k = k or scores.size(1)
        _, topk_idx = torch.topk(scores, k, dim=1)
        return topk_idx.cpu().apply_(lambda x: self.inv_item_map[int(x)])