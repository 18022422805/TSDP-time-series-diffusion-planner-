from typing import Dict
import torch


def _safe_mean(tensor: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    mask = torch.isfinite(tensor)
    tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1).to(tensor.dtype)
    return tensor.sum(dim=dim, keepdim=keepdim) / denom


def compute_time_series_stats(inputs: Dict[str, torch.Tensor], future_len: int) -> torch.Tensor:
    device = inputs["ego_current_state"].device
    batch = inputs["ego_current_state"].shape[0]

    neighbors = inputs["neighbor_agents_past"][..., :2]
    neighbor_mask = torch.isclose(neighbors.abs().sum(dim=-1), torch.tensor(0.0, device=device))
    velocity = neighbors[..., 1:, :] - neighbors[..., :-1, :]
    velocity_mask = neighbor_mask[..., 1:] | neighbor_mask[..., :-1]
    speed = velocity.norm(dim=-1)
    speed = torch.where(velocity_mask, torch.zeros_like(speed), speed)
    mean_speed = _safe_mean(speed, dim=(1, 2))

    acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]
    acc_mask = velocity_mask[..., 1:] | velocity_mask[..., :-1]
    acc_mag = acceleration.norm(dim=-1)
    acc_mag = torch.where(acc_mask, torch.zeros_like(acc_mag), acc_mag)
    acc_var = _safe_mean((acc_mag - _safe_mean(acc_mag, dim=2, keepdim=True)) ** 2, dim=(1, 2))

    route = inputs["route_lanes"][..., :2]
    route_mask = torch.isclose(route.abs().sum(dim=-1), torch.tensor(0.0, device=device))
    first_diff = route[..., 1:, :] - route[..., :-1, :]
    second_diff = first_diff[..., 1:, :] - first_diff[..., :-1, :]
    curvature = torch.linalg.norm(second_diff, dim=-1)
    curvature_mask = route_mask[..., 2:] | route_mask[..., 1:-1] | route_mask[..., :-2]
    curvature = torch.where(curvature_mask, torch.zeros_like(curvature), curvature)
    curvature = _safe_mean(curvature, dim=(1, 2))

    tau = torch.linspace(0.0, 1.0, future_len, device=device).unsqueeze(0).expand(batch, -1)
    stats = torch.stack(
        [
            tau,
            mean_speed.unsqueeze(-1).expand(-1, future_len),
            acc_var.unsqueeze(-1).expand(-1, future_len),
            curvature.unsqueeze(-1).expand(-1, future_len),
        ],
        dim=-1,
    )
    return stats


def compute_global_context(stats: torch.Tensor, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    batch = stats.shape[0]
    device = stats.device
    ego = inputs["ego_current_state"]
    ego_speed = torch.linalg.norm(ego[:, 4:6], dim=-1, keepdim=True) if ego.shape[-1] >= 6 else ego[:, :1]
    ego_heading = torch.stack(
        [torch.cos(ego[:, 2:3]), torch.sin(ego[:, 2:3])], dim=-1
    ).reshape(batch, -1)
    stats_mean = stats.mean(dim=1)
    stats_std = stats.std(dim=1).clamp(min=1e-6)

    neighbor_positions = inputs["neighbor_agents_past"][..., :2]
    neighbor_mask = torch.isclose(neighbor_positions.abs().sum(dim=-1), torch.tensor(0.0, device=device))
    neighbor_last = neighbor_positions[..., -1, :]
    neighbor_last = torch.where(neighbor_mask[..., -1, None], torch.zeros_like(neighbor_last), neighbor_last)
    neighbor_disp = torch.linalg.norm(neighbor_last - ego[:, None, :2], dim=-1)
    neighbor_disp = _safe_mean(neighbor_disp, dim=1, keepdim=True)

    global_context = torch.cat(
        [
            stats_mean,
            stats_std,
            ego_speed,
            ego_heading,
            neighbor_disp,
        ],
        dim=-1,
    )
    return global_context
