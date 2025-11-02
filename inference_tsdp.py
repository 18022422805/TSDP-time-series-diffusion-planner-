import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml

from diffusion_planner.tsdp.runtime import (
    build_engine,
    load_phase1,
    load_phase2,
    run_inference,
)
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSDP inference script")
    parser.add_argument("--config", type=str, default="tsdp_config.yaml")
    parser.add_argument("--phase1_ckpt", type=Path, required=True)
    parser.add_argument("--phase2_ckpt", type=Path, required=True)
    parser.add_argument("--context_file", type=Path, required=True, help="Path to serialized context tensor dictionary")
    parser.add_argument("--output", type=Path, default=Path("./tsdp_inference_output.pt"))
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    infer_cfg = cfg.get("inference", {})

    device = torch.device(args.device)

    obs_normalizer = ObservationNormalizer.from_json(args)
    state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = obs_normalizer
    args.state_normalizer = state_normalizer

    phase1 = load_phase1(args, args.phase1_ckpt, Path(args.config), device)

    context_dim = 12
    trajectory_dim = (1 + args.predicted_neighbor_num) * args.future_len * 4
    phase2 = load_phase2(Path(args.config), args.phase2_ckpt, device, context_dim, trajectory_dim)

    engine = build_engine(phase1, phase2)

    context = torch.load(args.context_file, map_location=device)
    if "ego_current_state" not in context:
        raise ValueError("Context file missing required keys")

    inputs = {}
    for k, v in context.items():
        tensor = v.to(device)
        if tensor.dim() == context["ego_current_state"].dim():
            tensor = tensor.unsqueeze(0)
        inputs[k] = tensor
    rollout = run_inference(
        engine=engine,
        inputs=inputs,
        obs_normalizer=obs_normalizer,
        state_normalizer=state_normalizer,
        future_len=args.future_len,
        deterministic=infer_cfg.get("deterministic", True),
    )

    schedule_samples = rollout["schedule_samples"]
    anisotropy_factor = rollout["anisotropy"]
    time_grid = engine.time_schedule.to(device)
    beta_min = engine.beta_min
    beta_max = engine.beta_max
    dynamic_schedule = []
    for idx in range(time_grid.shape[0]):
        base_beta = beta_min + (beta_max - beta_min) * time_grid[idx]
        step_scale = schedule_samples[idx].unsqueeze(-1)
        dynamic_schedule.append(base_beta * step_scale * anisotropy_factor)
    dynamic_schedule = torch.stack(dynamic_schedule, dim=0)

    output = {
        "trajectory": rollout["denorm_trajectories"].squeeze(0).cpu(),
        "anisotropy": anisotropy_factor.squeeze(0).cpu(),
        "schedule": dynamic_schedule[:, 0].cpu(),
    }
    torch.save(output, args.output)
    print(f"Inference complete. Saved to {args.output}")


if __name__ == "__main__":
    main()
