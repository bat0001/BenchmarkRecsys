import torch

class BanditPolicy:
    def select(self, contexts: torch.Tensor, k: int) -> torch.Tensor:
        """Renvoie les indices des k images à proposer."""
    def update(self, chosen: torch.Tensor, reward: torch.Tensor): ...

class LinUCB(BanditPolicy):
    def __init__(self, dim: int, alpha: float = 1.0, device="cpu"):
        self.A = torch.eye(dim, device=device)      
        self.b = torch.zeros(dim, device=device)    
        self.alpha = alpha
        self.device = device

    def _ucb(self, X): 
        X = X.to(self.device)
        A_inv = torch.linalg.inv(self.A)
        theta = A_inv @ self.b                     
        means = X @ theta                          
        vars = torch.sum(X @ A_inv * X, dim=1)      
        return means + self.alpha * torch.sqrt(vars)

    @torch.no_grad()
    def select(self, contexts: torch.Tensor, k: int) -> torch.Tensor:
        """Greedy top‑k according to current UCB score."""
        scores = self._ucb(contexts)          
        topk = torch.topk(scores, k=k).indices     
        return topk.to(contexts.device)     

    def update(self, chosen_ctx: torch.Tensor, reward: torch.Tensor):
        """chosen_ctx: k×d  – reward: scalaire ou (k,) broadcasté."""
        if reward.ndim == 0:
            reward = reward.expand(len(chosen_ctx))
        for x, r in zip(chosen_ctx, reward):
            x = x.unsqueeze(1)                    
            self.A += x @ x.T
            self.b += r * x.squeeze()