import argparse
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import DiffusionBackbone
from diffusion_planner.tsdp.models import AnisotropyNetwork, NoiseSchedulingPolicy, ScoreCorrectionPolicy
from diffusion_planner.tsdp.features import compute_time_series_stats, compute_global_context
from diffusion_planner.tsdp.sampler import TSDPDiffusionEngine
from diffusion_planner.tsdp.reward import compute_patch_rewards
from diffusion_planner.tsdp.grpo import normalized_advantage, grpo_loss
from diffusion_planner.utils.train_utils import set_seed
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 policy optimization for TSDP")
    parser.add_argument("--config", type=str, default="tsdp_config.yaml")
    parser.add_argument("--phase1_ckpt", type=str, required=True, help="Checkpoint from Phase 1 training")
    parser.add_argument("--train_set", type=str, default=None)
    parser.add_argument("--train_set_list", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="./artifacts/phase2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--augment_prob", type=float, default=0.0)
    parser.add_argument("--use_data_augment", type=bool, default=False)
    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--time_len", type=int, default=21)
    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--predicted_neighbor_num", type=int, default=10)
    parser.add_argument("--diffusion_model_type", type=str, default="x_start", choices=["score", "x_start"])
    return parser.parse_args()


def load_phase_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "phase_2" not in cfg:
        raise ValueError("Missing phase_2 section in config")
    return cfg["phase_2"], cfg.get("phase_1", {})


def build_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, ObservationNormalizer, StateNormalizer]:
    obs_normalizer = ObservationNormalizer.from_json(args)
    state_normalizer = StateNormalizer.from_json(args)

    dataset = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    args.observation_normalizer = obs_normalizer
    args.state_normalizer = state_normalizer
    return loader, obs_normalizer, state_normalizer


def prepare_inputs(
    batch,
    args: argparse.Namespace,
    device: torch.device,
    obs_normalizer: ObservationNormalizer,
    aug: StatePerturbation,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = {
        "ego_current_state": batch[0].to(device),
        "neighbor_agents_past": batch[2].to(device),
        "lanes": batch[4].to(device),
        "lanes_speed_limit": batch[5].to(device),
        "lanes_has_speed_limit": batch[6].to(device),
        "route_lanes": batch[7].to(device),
        "route_lanes_speed_limit": batch[8].to(device),
        "route_lanes_has_speed_limit": batch[9].to(device),
        "static_objects": batch[10].to(device),
    }

    ego_future = batch[1].to(device)
    neighbors_future = batch[3].to(device)

    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = torch.cat(
        [
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )

    neighbor_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )
    neighbors_future[neighbor_mask] = 0.0

    inputs = obs_normalizer(inputs)
    return inputs, ego_future, neighbors_future, neighbor_mask


def load_phase1_checkpoint(path: str, backbone: nn.Module, anisotropy: nn.Module, device: torch.device) -> None:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "backbone" in payload and "anisotropy" in payload:
        backbone.load_state_dict(payload["backbone"])
        anisotropy.load_state_dict(payload["anisotropy"])
    else:
        backbone.load_state_dict(payload)


def compute_kl(new_mean: torch.Tensor, new_std: torch.Tensor, ref_mean: torch.Tensor, ref_std: torch.Tensor) -> torch.Tensor:
    ratio = (new_std / ref_std).clamp(min=1e-6)
    term = torch.log(ratio) + (ref_std.pow(2) + (ref_mean - new_mean).pow(2)) / (2 * new_std.pow(2)) - 0.5
    return term.mean()


def main() -> None:
    args = parse_args()
    phase_cfg, phase1_cfg = load_phase_config(args.config)
    if args.batch_size is None:
        args.batch_size = phase_cfg.get("batch_size", 128)

    device = torch.device(args.device)
    set_seed(args.seed)

    loader, obs_normalizer, state_normalizer = build_dataloader(args)
    aug = StatePerturbation(augment_prob=args.augment_prob, device=device) if args.use_data_augment else None

    backbone = DiffusionBackbone(args).to(device)
    anisotropy = AnisotropyNetwork(4).to(device)
    load_phase1_checkpoint(args.phase1_ckpt, backbone, anisotropy, device)
    backbone.eval()
    anisotropy.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in anisotropy.parameters():
        p.requires_grad_(False)

    context_dim = 2 * 4 + 1 + 2 + 1
    action_dim = (1 + args.predicted_neighbor_num) * args.future_len * 4
    trajectory_dim = action_dim

    schedule_policy = NoiseSchedulingPolicy(
        context_dim=context_dim,
        hidden_dim=phase_cfg.get("schedule_hidden_dim", 256),
        embed_dim=phase_cfg.get("schedule_embed_dim", 128),
    ).to(device)
    score_policy = ScoreCorrectionPolicy(
        context_dim=context_dim,
        trajectory_dim=trajectory_dim,
        hidden_dim=phase_cfg.get("score_hidden_dim", 384),
        action_dim=action_dim,
    ).to(device)
    schedule_policy.train()
    score_policy.train()

    optimizer = torch.optim.AdamW(
        list(schedule_policy.parameters()) + list(score_policy.parameters()),
        lr=phase_cfg.get("learning_rate_policies", 1e-4),
        weight_decay=phase_cfg.get("weight_decay", 1e-2),
    )
    gradient_clip = phase_cfg.get("gradient_clip", 1.0)

    engine = TSDPDiffusionEngine(
        backbone,
        anisotropy,
        schedule_policy,
        score_policy,
        future_len=args.future_len,
        predicted_neighbor_num=args.predicted_neighbor_num,
        diffusion_steps=phase_cfg.get("diffusion_steps", 20),
        beta_min=phase_cfg.get("beta_min", phase1_cfg.get("beta_min", 0.1)),
        beta_max=phase_cfg.get("beta_max", phase1_cfg.get("beta_max", 20.0)),
    )

    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)

    prev_schedule: List[Tuple[torch.Tensor, torch.Tensor]] = []
    prev_score: List[Tuple[torch.Tensor, torch.Tensor]] = []
    kl_weight = phase_cfg.get("kl_weight", 0.05)

    for epoch in range(1, phase_cfg.get("epochs", 150) + 1):
        total_reward_epoch = []
        for batch in loader:
            inputs, _, _, _ = prepare_inputs(batch, args, device, obs_normalizer, aug)
            denorm_inputs = obs_normalizer.inverse(inputs)
            stats = compute_time_series_stats(denorm_inputs, args.future_len)
            global_context = compute_global_context(stats, denorm_inputs)

            rollout = engine.generate(inputs, stats, global_context, deterministic=False)
            trajectories = rollout["trajectories"]
            world_trajectories = state_normalizer.inverse(trajectories)
            reward_inputs = denorm_inputs

            patch_rewards, total_reward = compute_patch_rewards(
                world_trajectories,
                reward_inputs,
                patch_size=phase_cfg.get("patch_size", 8),
                gamma=phase_cfg.get("gamma", 0.95),
                weight_safety=phase_cfg.get("reward_weights", {}).get("safety", 0.1),
                weight_comfort=phase_cfg.get("reward_weights", {}).get("comfort", 0.02),
                weight_progress=phase_cfg.get("reward_weights", {}).get("progress", 1.0),
            )

            advantages = normalized_advantage(total_reward)
            schedule_log_prob = rollout["schedule_log_probs"].sum(dim=0)
            score_log_prob = rollout["score_log_probs"].sum(dim=0)

            if prev_schedule:
                schedule_kl_terms = []
                for idx, dist in enumerate(rollout["schedule_dists"]):
                    ref_mean, ref_std = prev_schedule[idx]
                    schedule_kl_terms.append(compute_kl(dist.loc, dist.scale, ref_mean, ref_std))
                schedule_kl = torch.stack(schedule_kl_terms).mean()
            else:
                schedule_kl = torch.tensor(0.0, device=device)

            if prev_score:
                score_kl_terms = []
                for idx, dist in enumerate(rollout["score_dists"]):
                    ref_mean, ref_std = prev_score[idx]
                    score_kl_terms.append(compute_kl(dist.loc, dist.scale, ref_mean, ref_std))
                score_kl = torch.stack(score_kl_terms).mean()
            else:
                score_kl = torch.tensor(0.0, device=device)

            loss = grpo_loss(
                schedule_log_prob,
                score_log_prob,
                advantages,
                schedule_kl,
                score_kl,
                kl_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(schedule_policy.parameters()) + list(score_policy.parameters()),
                gradient_clip,
            )
            optimizer.step()

            prev_schedule = [(dist.loc.detach(), dist.scale.detach()) for dist in rollout["schedule_dists"]]
            prev_score = [(dist.loc.detach(), dist.scale.detach()) for dist in rollout["score_dists"]]

            total_reward_epoch.append(total_reward.detach())

        mean_reward = torch.cat(total_reward_epoch).mean()
        print(f"Epoch {epoch:04d} | Reward: {mean_reward.item():.4f}")

        if epoch % 5 == 0:
            payload = {
                "epoch": epoch,
                "schedule_policy": schedule_policy.state_dict(),
                "score_policy": score_policy.state_dict(),
            }
            torch.save(payload, os.path.join(save_dir, f"epoch_{epoch:04d}.pth"))
            torch.save(payload, os.path.join(save_dir, "latest.pth"))

    payload = {
        "epoch": epoch,
        "schedule_policy": schedule_policy.state_dict(),
        "score_policy": score_policy.state_dict(),
    }
    torch.save(payload, os.path.join(save_dir, f"epoch_{epoch:04d}.pth"))
    torch.save(payload, os.path.join(save_dir, "latest.pth"))


if __name__ == "__main__":
    main()
