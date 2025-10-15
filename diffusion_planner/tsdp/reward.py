from typing import Dict, Tuple
import math
import torch


def compute_patch_rewards(
    trajectories: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    patch_size: int,
    gamma: float,
    weight_safety: float,
    weight_comfort: float,
    weight_progress: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, agents, horizon, _ = trajectories.shape
    num_patches = math.ceil(horizon / patch_size)

    static_objects = inputs["static_objects"][..., :2]
    static_mask = torch.isclose(static_objects.abs().sum(dim=-1), torch.tensor(0.0, device=trajectories.device))

    patch_rewards = []
    discount_factors = []

    ego_traj = trajectories[:, 0, :, :2]
    velocities = ego_traj[:, 1:, :] - ego_traj[:, :-1, :]
    accelerations = velocities[:, 1:, :] - velocities[:, :-1, :]
    jerks = accelerations[:, 1:, :] - accelerations[:, :-1, :]

    for idx in range(num_patches):
        start = idx * patch_size
        end = min((idx + 1) * patch_size, horizon)

        ego_segment = ego_traj[:, start:end, :]
        if end - start <= 1:
            jerk_cost = torch.zeros(batch, device=trajectories.device)
        else:
            jerk_slice = jerks[:, max(0, start - 3):max(0, end - 3), :]
            jerk_cost = jerk_slice.norm(dim=-1).mean(dim=-1)

        progress = torch.linalg.norm(ego_segment[:, -1, :] - ego_segment[:, 0, :], dim=-1)

        neighbor_segment = trajectories[:, 1:, start:end, :2]
        neighbor_mask = torch.isclose(neighbor_segment.abs().sum(dim=-1), torch.tensor(0.0, device=trajectories.device))
        neighbor_segment = torch.where(neighbor_mask[..., None], torch.zeros_like(neighbor_segment), neighbor_segment)

        if neighbor_segment.shape[1] == 0:
            agent_dist = torch.full((batch,), 1e3, device=trajectories.device)
        else:
            ego_points = ego_segment[:, None, :, :].unsqueeze(2)
            neighbor_points = neighbor_segment[:, :, :, :]
            diff_agents = ego_points - neighbor_points[:, None, :, :, :]
            agent_dist_map = diff_agents.norm(dim=-1)
            agent_mask_map = neighbor_mask[:, None, :, :].expand_as(agent_dist_map)
            agent_dist_map = torch.where(agent_mask_map, torch.full_like(agent_dist_map, 1e6), agent_dist_map)
            agent_dist = agent_dist_map.view(batch, -1).min(dim=-1).values

        static_points = static_objects.unsqueeze(-2)
        static_diff = ego_segment[:, None, :, :] - static_points[:, :, None, :]
        static_dist_map = static_diff.norm(dim=-1)
        static_mask_map = static_mask[:, :, None].expand_as(static_dist_map)
        static_dist_map = torch.where(static_mask_map, torch.full_like(static_dist_map, 1e6), static_dist_map)
        static_dist = static_dist_map.view(batch, -1).min(dim=-1).values

        safety = torch.minimum(agent_dist, static_dist)

        reward = (
            weight_progress * progress
            - weight_comfort * jerk_cost
            + weight_safety * safety
        )
        patch_rewards.append(reward)
        discount_factors.append(gamma ** idx)

    patch_tensor = torch.stack(patch_rewards, dim=1)
    discounts = torch.tensor(discount_factors, device=trajectories.device).view(1, -1)
    total_reward = (patch_tensor * discounts).sum(dim=1)

    return patch_tensor, total_reward
