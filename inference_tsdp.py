import argparse
import os
from typing import Dict

import torch
import yaml

from diffusion_planner.model.diffusion_planner import DiffusionBackbone
from diffusion_planner.tsdp.models import AnisotropyNetwork, NoiseSchedulingPolicy, ScoreCorrectionPolicy
from diffusion_planner.tsdp.features import compute_time_series_stats, compute_global_context
from diffusion_planner.tsdp.sampler import TSDPDiffusionEngine
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSDP inference script")
    parser.add_argument("--config", type=str, default="tsdp_config.yaml")
    parser.add_argument("--phase1_ckpt", type=str, required=True)
    parser.add_argument("--phase2_ckpt", type=str, required=True)
    parser.add_argument("--context_file", type=str, required=True, help="Path to serialized context tensor dictionary")
    parser.add_argument("--output", type=str, default="./tsdp_inference_output.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--predicted_neighbor_num", type=int, default=10)
    parser.add_argument("--diffusion_model_type", type=str, default="x_start", choices=["score", "x_start"])
    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--time_len", type=int, default=21)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_checkpoint(path: str) -> Dict:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return payload
    return {"backbone": payload}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    phase2_cfg = cfg.get("phase_2", {})
    infer_cfg = cfg.get("inference", {})

    device = torch.device(args.device)

    obs_normalizer = ObservationNormalizer.from_json(args)
    state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = obs_normalizer
    args.state_normalizer = state_normalizer

    backbone = DiffusionBackbone(args).to(device)
    anisotropy = AnisotropyNetwork(4).to(device)

    phase1_state = load_checkpoint(args.phase1_ckpt)
    if "backbone" not in phase1_state or "anisotropy" not in phase1_state:
        raise ValueError("Phase 1 checkpoint must contain backbone and anisotropy weights")
    backbone.load_state_dict(phase1_state["backbone"])
    anisotropy.load_state_dict(phase1_state["anisotropy"])
    backbone.eval()
    anisotropy.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in anisotropy.parameters():
        p.requires_grad_(False)

    context_dim = 12
    action_dim = (1 + args.predicted_neighbor_num) * args.future_len * 4

    schedule_policy = NoiseSchedulingPolicy(
        context_dim=context_dim,
        hidden_dim=phase2_cfg.get("schedule_hidden_dim", 256),
        embed_dim=phase2_cfg.get("schedule_embed_dim", 128),
    ).to(device)
    score_policy = ScoreCorrectionPolicy(
        context_dim=context_dim,
        trajectory_dim=action_dim,
        hidden_dim=phase2_cfg.get("score_hidden_dim", 384),
        action_dim=action_dim,
    ).to(device)

    phase2_state = load_checkpoint(args.phase2_ckpt)
    if "schedule_policy" not in phase2_state or "score_policy" not in phase2_state:
        raise ValueError("Phase 2 checkpoint missing policy weights")
    schedule_policy.load_state_dict(phase2_state["schedule_policy"])
    score_policy.load_state_dict(phase2_state["score_policy"])
    schedule_policy.eval()
    score_policy.eval()

    engine = TSDPDiffusionEngine(
        backbone,
        anisotropy,
        schedule_policy,
        score_policy,
        future_len=args.future_len,
        predicted_neighbor_num=args.predicted_neighbor_num,
        diffusion_steps=infer_cfg.get("diffusion_steps", phase2_cfg.get("diffusion_steps", 20)),
        beta_min=phase2_cfg.get("beta_min", 0.1),
        beta_max=phase2_cfg.get("beta_max", 20.0),
    )

    context = torch.load(args.context_file, map_location=device)
    if "ego_current_state" not in context:
        raise ValueError("Context file missing required keys")

    inputs = {}
    for k, v in context.items():
        tensor = v.to(device)
        if tensor.dim() == context["ego_current_state"].dim():
            tensor = tensor.unsqueeze(0)
        inputs[k] = tensor
    inputs = obs_normalizer(inputs)

    denorm_inputs = obs_normalizer.inverse(inputs)
    stats = compute_time_series_stats(denorm_inputs, args.future_len)
    global_context = compute_global_context(stats, denorm_inputs)

    rollout = engine.generate(inputs, stats, global_context, deterministic=infer_cfg.get("deterministic", True))
    trajectories = rollout["trajectories"]
    denorm_traj = state_normalizer.inverse(trajectories)

    schedule_samples = rollout["schedule_samples"]
    anisotropy_factor = rollout["anisotropy"]
    time_grid = engine.time_schedule.to(device)
    beta_min = engine.beta_min
    beta_max = engine.beta_max
    dynamic_schedule = []
    for idx in range(time_grid.shape[0]):
        base_beta = beta_min + (beta_max - beta_min) * time_grid[idx]
        step_scale = schedule_samples[idx].unsqueeze(-1)  # [B, 1]
        dynamic_schedule.append(base_beta * step_scale * anisotropy_factor)
    dynamic_schedule = torch.stack(dynamic_schedule, dim=0)

    output = {
        "trajectory": denorm_traj.squeeze(0).cpu(),
        "anisotropy": anisotropy_factor.squeeze(0).cpu(),
        "schedule": dynamic_schedule[:, 0].cpu(),
    }
    torch.save(output, args.output)
    print(f"Inference complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
