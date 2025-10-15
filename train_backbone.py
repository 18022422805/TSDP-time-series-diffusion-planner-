import argparse
import os
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from diffusion_planner.model.diffusion_planner import DiffusionBackbone
from diffusion_planner.tsdp.models import AnisotropyNetwork
from diffusion_planner.tsdp.features import compute_time_series_stats
from diffusion_planner.tsdp.losses import phase1_loss
from diffusion_planner.utils.train_utils import set_seed
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 training for TSDP")
    parser.add_argument("--config", type=str, default="tsdp_config.yaml", help="Config file with phase settings")
    parser.add_argument("--train_set", type=str, default=None, help="Path to serialized training data")
    parser.add_argument("--train_set_list", type=str, default=None, help="List file for training data")
    parser.add_argument("--save_dir", type=str, default="./artifacts/phase1", help="Checkpoint root")
    parser.add_argument("--device", type=str, default="cuda", help="Training device")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config if provided")
    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--use_data_augment", type=bool, default=True)
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
    if "phase_1" not in cfg:
        raise ValueError("Missing phase_1 section in config")
    return cfg["phase_1"]

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

def save_checkpoint(save_root: str, epoch: int, backbone: nn.Module, anisotropy: nn.Module) -> None:
    os.makedirs(save_root, exist_ok=True)
    payload = {
        "epoch": epoch,
        "backbone": backbone.state_dict(),
        "anisotropy": anisotropy.state_dict(),
    }
    ckpt_path = os.path.join(save_root, f"epoch_{epoch:04d}.pth")
    torch.save(payload, ckpt_path)
    torch.save(payload, os.path.join(save_root, "latest.pth"))


def main() -> None:
    args = parse_args()
    phase_cfg = load_phase_config(args.config)
    if args.batch_size is None:
        args.batch_size = phase_cfg.get("batch_size", 512)

    device = torch.device(args.device)
    set_seed(args.seed)

    loader, obs_normalizer, state_normalizer = build_dataloader(args)
    aug = StatePerturbation(augment_prob=args.augment_prob, device=device) if args.use_data_augment else None

    backbone = DiffusionBackbone(args).to(device)
    for p in backbone.parameters():
        p.requires_grad_(True)

    stats_dim = 4
    anisotropy = AnisotropyNetwork(stats_dim).to(device)

    params = [
        {"params": backbone.parameters(), "lr": phase_cfg.get("learning_rate_backbone", 5e-4)},
        {"params": anisotropy.parameters(), "lr": phase_cfg.get("learning_rate_anisotropy", 1e-4)},
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=phase_cfg.get("weight_decay", 1e-2))

    gradient_clip = phase_cfg.get("gradient_clip", 5.0)
    save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    backbone.train()
    anisotropy.train()

    for epoch in range(1, phase_cfg.get("epochs", 200) + 1):
        epoch_loss = []
        for batch in loader:
            inputs, ego_future, neighbors_future, neighbor_mask = prepare_inputs(batch, args, device, obs_normalizer, aug)
            denorm_inputs = obs_normalizer.inverse(inputs)
            stats = compute_time_series_stats(denorm_inputs, args.future_len)

            loss_dict, _, _ = phase1_loss(
                backbone,
                anisotropy,
                inputs,
                (ego_future, neighbors_future, neighbor_mask),
                stats,
                state_normalizer,
                phase_cfg.get("reg_mean_weight", 0.1),
                phase_cfg.get("reg_smooth_weight", 0.05),
                phase_cfg.get("reg_scale_weight", 1.0),
                args.diffusion_model_type,
            )

            optimizer.zero_grad()
            loss_dict["loss"].backward()
            nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(anisotropy.parameters()), gradient_clip)
            optimizer.step()

            epoch_loss.append(loss_dict["loss"].detach())

        mean_loss = torch.stack(epoch_loss).mean()
        print(f"Epoch {epoch:04d} | L_PhaseA: {mean_loss.item():.4f}")

        if epoch % 10 == 0:
            save_checkpoint(save_dir, epoch, backbone, anisotropy)

    save_checkpoint(save_dir, epoch, backbone, anisotropy)


if __name__ == "__main__":
    main()
