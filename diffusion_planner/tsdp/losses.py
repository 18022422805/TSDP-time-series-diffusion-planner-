from typing import Dict, Tuple
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def phase1_loss(
    backbone: nn.Module,
    anisotropy: nn.Module,
    inputs: Dict[str, torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    stats: torch.Tensor,
    state_normalizer: StateNormalizer,
    reg_mean_weight: float,
    reg_smooth_weight: float,
    reg_scale_weight: float,
    model_type: str,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask

    B, Pn, T, _ = neighbors_future.shape
    ego_current = inputs["ego_current_state"][:, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    r_tau = anisotropy(stats)
    r_tau = torch.clamp(r_tau, min=0.1, max=10.0)

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device)
    z = torch.randn_like(gt_future, device=gt_future.device)

    all_gt = torch.cat([current_states[:, :, None, :], state_normalizer(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = backbone.sde.marginal_prob(all_gt[..., 1:, :], t)
    std = std.view_as(mean)
    std = std * torch.sqrt(r_tau[:, None, :, None])

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = backbone(merged_inputs)
    score = decoder_output["score"][:, :, 1:, :]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z) ** 2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :]) ** 2, dim=-1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    loss_dict: Dict[str, torch.Tensor] = {}

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]
    if masked_prediction_loss.numel() > 0:
        loss_dict["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss_dict["neighbor_prediction_loss"] = torch.tensor(0.0, device=dpm_loss.device)

    loss_dict["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    mean_penalty = ((r_tau.mean(dim=-1) - 1.0) ** 2).mean()
    smooth_penalty = ((r_tau[:, 1:] - r_tau[:, :-1]) ** 2).mean()
    scale_penalty = (torch.clamp(r_tau - 1.5, min=0.0) ** 2 + torch.clamp(0.5 - r_tau, min=0.0) ** 2).mean()
    loss_dict["anisotropy_reg"] = (
        reg_mean_weight * mean_penalty + reg_smooth_weight * smooth_penalty + reg_scale_weight * scale_penalty
    )

    loss_dict["loss"] = (
        loss_dict["neighbor_prediction_loss"] + loss_dict["ego_planning_loss"] + loss_dict["anisotropy_reg"]
    )

    return loss_dict, decoder_output, r_tau
