from typing import Tuple
import torch


def normalized_advantage(total_reward: torch.Tensor) -> torch.Tensor:
    mean = total_reward.mean()
    std = total_reward.std().clamp(min=1e-6)
    return (total_reward - mean) / std


def grpo_loss(
    log_probs_schedule: torch.Tensor,
    log_probs_score: torch.Tensor,
    advantages: torch.Tensor,
    kl_schedule: torch.Tensor,
    kl_score: torch.Tensor,
    kl_weight: float,
) -> torch.Tensor:
    policy_term = -(advantages.detach() * (log_probs_schedule + log_probs_score)).mean()
    kl_term = kl_weight * (kl_schedule.mean() + kl_score.mean())
    return policy_term + kl_term
