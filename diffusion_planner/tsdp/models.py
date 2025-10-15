from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.distributions as D


def _timestep_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AnisotropyNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        batch, steps, feat = stats.shape
        flat = stats.reshape(batch * steps, feat)
        r = torch.nn.functional.softplus(self.net(flat)) + 1.0
        return r.reshape(batch, steps)


class NoiseSchedulingPolicy(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int, embed_dim: int, min_log_std: float = -6.0, max_log_std: float = 1.5):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.time_proj = nn.Linear(embed_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        self.embed_dim = embed_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(
        self,
        context: torch.Tensor,
        time: torch.Tensor,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, D.Normal]:
        time_embed = _timestep_embed(time, self.embed_dim)
        h = self.context_proj(context) + self.time_proj(time_embed)
        h = self.mlp(h)
        mean = torch.tanh(self.mean_head(h)).squeeze(-1) + 1.0
        log_std = self.log_std_head(h).squeeze(-1).clamp(self.min_log_std, self.max_log_std)
        std = torch.nn.functional.softplus(log_std) + 1e-4
        dist = D.Normal(mean, std)
        if stochastic:
            sample = dist.rsample()
        else:
            sample = mean
        log_prob = dist.log_prob(sample)
        return sample, log_prob, dist


class ScoreCorrectionPolicy(nn.Module):
    def __init__(
        self,
        context_dim: int,
        trajectory_dim: int,
        hidden_dim: int,
        action_dim: int,
        min_log_std: float = -6.0,
        max_log_std: float = 1.0,
    ):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(trajectory_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(
        self,
        context: torch.Tensor,
        trajectory_state: torch.Tensor,
        stochastic: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, D.Normal]:
        state_flat = trajectory_state.reshape(trajectory_state.shape[0], -1)
        h = self.context_proj(context) + self.state_proj(state_flat)
        h = self.mlp(h)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.min_log_std, self.max_log_std)
        std = torch.nn.functional.softplus(log_std) + 1e-4
        dist = D.Normal(mean, std)
        if stochastic:
            sample = dist.rsample()
        else:
            sample = mean
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob, dist
