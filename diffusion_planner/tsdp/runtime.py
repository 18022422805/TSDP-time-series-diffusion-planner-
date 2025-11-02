from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml

from diffusion_planner.model.diffusion_planner import DiffusionBackbone
from diffusion_planner.tsdp.features import compute_time_series_stats, compute_global_context
from diffusion_planner.tsdp.models import (
    AnisotropyNetwork,
    NoiseSchedulingPolicy,
    ScoreCorrectionPolicy,
)
from diffusion_planner.tsdp.sampler import TSDPDiffusionEngine
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


@dataclass
class Phase1Artifacts:
    backbone: DiffusionBackbone
    anisotropy: AnisotropyNetwork
    future_len: int
    predicted_neighbor_num: int


@dataclass
class Phase2Artifacts:
    schedule_policy: NoiseSchedulingPolicy
    score_policy: ScoreCorrectionPolicy
    beta_min: float
    beta_max: float
    diffusion_steps: int


def _load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_phase1(
    args_namespace,
    phase1_ckpt: Path,
    config_path: Path,
    device: torch.device,
) -> Phase1Artifacts:
    cfg = _load_config(config_path)
    phase1_cfg = cfg.get("phase_1", {})

    args_namespace.normalization_file_path = getattr(args_namespace, "normalization_file_path", "normalization.json")
    args_namespace.predicted_neighbor_num = getattr(args_namespace, "predicted_neighbor_num", phase1_cfg.get("predicted_neighbor_num", 10))
    args_namespace.future_len = getattr(args_namespace, "future_len", phase1_cfg.get("future_len", 80))

    backbone = DiffusionBackbone(args_namespace).to(device)
    anisotropy = AnisotropyNetwork(feature_dim=4).to(device)

    checkpoint = torch.load(phase1_ckpt, map_location=device)
    if "backbone" not in checkpoint or "anisotropy" not in checkpoint:
        raise ValueError(f"Phase 1 checkpoint {phase1_ckpt} is missing required keys.")
    backbone.load_state_dict(checkpoint["backbone"])
    anisotropy.load_state_dict(checkpoint["anisotropy"])

    backbone.eval()
    anisotropy.eval()
    for module in (backbone, anisotropy):
        for param in module.parameters():
            param.requires_grad_(False)

    return Phase1Artifacts(
        backbone=backbone,
        anisotropy=anisotropy,
        future_len=args_namespace.future_len,
        predicted_neighbor_num=args_namespace.predicted_neighbor_num,
    )


def load_phase2(
    config_path: Path,
    phase2_ckpt: Path,
    device: torch.device,
    context_dim: int,
    trajectory_dim: int,
) -> Phase2Artifacts:
    cfg = _load_config(config_path)
    phase2_cfg = cfg.get("phase_2", {})

    schedule_policy = NoiseSchedulingPolicy(
        context_dim=context_dim,
        hidden_dim=phase2_cfg.get("schedule_hidden_dim", 256),
        embed_dim=phase2_cfg.get("schedule_embed_dim", 128),
    ).to(device)
    score_policy = ScoreCorrectionPolicy(
        context_dim=context_dim,
        trajectory_dim=trajectory_dim,
        hidden_dim=phase2_cfg.get("score_hidden_dim", 384),
        action_dim=trajectory_dim,
    ).to(device)

    checkpoint = torch.load(phase2_ckpt, map_location=device)
    if "schedule_policy" not in checkpoint or "score_policy" not in checkpoint:
        raise ValueError(f"Phase 2 checkpoint {phase2_ckpt} is missing required keys.")
    schedule_policy.load_state_dict(checkpoint["schedule_policy"])
    score_policy.load_state_dict(checkpoint["score_policy"])

    schedule_policy.eval()
    score_policy.eval()
    for module in (schedule_policy, score_policy):
        for param in module.parameters():
            param.requires_grad_(False)

    return Phase2Artifacts(
        schedule_policy=schedule_policy,
        score_policy=score_policy,
        beta_min=phase2_cfg.get("beta_min", 0.1),
        beta_max=phase2_cfg.get("beta_max", 20.0),
        diffusion_steps=phase2_cfg.get("diffusion_steps", 20),
    )


def build_engine(
    phase1: Phase1Artifacts,
    phase2: Phase2Artifacts,
) -> TSDPDiffusionEngine:
    return TSDPDiffusionEngine(
        backbone=phase1.backbone,
        anisotropy=phase1.anisotropy,
        schedule_policy=phase2.schedule_policy,
        score_policy=phase2.score_policy,
        future_len=phase1.future_len,
        predicted_neighbor_num=phase1.predicted_neighbor_num,
        diffusion_steps=phase2.diffusion_steps,
        beta_min=phase2.beta_min,
        beta_max=phase2.beta_max,
    )


def normalize_inputs(
    inputs: Dict[str, torch.Tensor],
    obs_normalizer: ObservationNormalizer,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    normalized_inputs = obs_normalizer(inputs)
    denorm_inputs = obs_normalizer.inverse(normalized_inputs)
    return normalized_inputs, denorm_inputs


def run_inference(
    engine: TSDPDiffusionEngine,
    inputs: Dict[str, torch.Tensor],
    obs_normalizer: ObservationNormalizer,
    state_normalizer: StateNormalizer,
    future_len: int,
    deterministic: bool = True,
) -> Dict[str, torch.Tensor]:
    normalized_inputs, denorm_inputs = normalize_inputs(inputs, obs_normalizer)
    stats = compute_time_series_stats(denorm_inputs, future_len)
    global_context = compute_global_context(stats, denorm_inputs)
    rollout = engine.generate(normalized_inputs, stats, global_context, deterministic=deterministic)
    trajectories = rollout["trajectories"]
    denorm_traj = state_normalizer.inverse(trajectories)
    rollout["denorm_trajectories"] = denorm_traj
    return rollout
