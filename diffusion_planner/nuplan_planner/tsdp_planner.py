from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import math
import torch

from diffusion_planner.nuplan_planner.context_builder import ContextSettings, build_tsdp_context
from diffusion_planner.tsdp.runtime import (
    build_engine,
    load_phase1,
    load_phase2,
    run_inference,
)
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer

try:
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.planning.simulation.observation.observation_type import Observation
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    from nuplan.planning.simulation.trajectory.trajectory import InterpolatedTrajectory
    from nuplan.planning.simulation.observation.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "nuPlan must be installed to use TSDPPlanner. "
        "Please ensure nuplan-devkit is available in the current environment."
    ) from exc


@dataclass
class TSDPPlannerConfig:
    phase1_ckpt: Path
    phase2_ckpt: Path
    normalization_file_path: Path
    config_file: Path
    device: str = "cuda"
    context_settings: ContextSettings = ContextSettings()
    deterministic: bool = True


class TSDPPlanner(AbstractPlanner):
    """
    Minimal nuPlan planner wrapper that leverages the two-stage TSDP pipeline.

    The planner currently provides a lightweight context builder that delivers tensors
    with the correct shape. For high-fidelity performance, the lane, route, and static
    object features should be populated with richer map extractions.
    """

    def __init__(self, cfg: TSDPPlannerConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._device = torch.device(cfg.device)
        self._context_settings = cfg.context_settings

        model_args = SimpleNamespace(
            device=cfg.device,
            normalization_file_path=str(cfg.normalization_file_path),
            predicted_neighbor_num=cfg.context_settings.predicted_neighbor_num,
            future_len=cfg.context_settings.future_len,
            time_len=cfg.context_settings.time_len,
            agent_state_dim=cfg.context_settings.agent_state_dim,
            agent_num=cfg.context_settings.agent_num,
            diffusion_model_type="x_start",
            encoder_depth=3,
            decoder_depth=3,
            hidden_dim=192,
            num_heads=6,
            encoder_drop_path_rate=0.1,
            decoder_drop_path_rate=0.1,
            static_objects_num=cfg.context_settings.static_objects_num,
            static_objects_state_dim=cfg.context_settings.static_objects_state_dim,
            lane_num=cfg.context_settings.lane_num,
            lane_len=cfg.context_settings.lane_len,
            lane_state_dim=cfg.context_settings.lane_state_dim,
            route_num=cfg.context_settings.route_num,
            route_len=cfg.context_settings.route_len,
            route_state_dim=cfg.context_settings.route_state_dim,
        )

        self._obs_normalizer = ObservationNormalizer.from_json(model_args)
        self._state_normalizer = StateNormalizer.from_json(model_args)
        model_args.observation_normalizer = self._obs_normalizer
        model_args.state_normalizer = self._state_normalizer

        self._phase1 = load_phase1(
            model_args,
            cfg.phase1_ckpt,
            cfg.config_file,
            self._device,
        )
        context_dim = 12
        trajectory_dim = (1 + cfg.context_settings.predicted_neighbor_num) * cfg.context_settings.future_len * 4
        self._phase2 = load_phase2(
            cfg.config_file,
            cfg.phase2_ckpt,
            self._device,
            context_dim,
            trajectory_dim,
        )
        self._engine = build_engine(self._phase1, self._phase2)

    def name(self) -> str:
        return "tsdp_planner"

    def initialize(self) -> None:
        # No persistent state required for open-loop evaluation.
        return

    def observation_to_ego_state(self, observation: Observation):
        return observation.ego_state

    def compute_trajectory(
        self,
        iteration: SimulationIteration,
        history: SimulationHistoryBuffer,
        observations: Observation,
    ) -> InterpolatedTrajectory:
        with torch.no_grad():
            context = build_tsdp_context(
                history=history,
                observation=observations,
                settings=self._context_settings,
                device=self._device,
            )
            batched_inputs = {}
            reference_dim = context["ego_current_state"].dim()
            for key, value in context.items():
                tensor = value
                if tensor.dim() == reference_dim:
                    tensor = tensor.unsqueeze(0)
                batched_inputs[key] = tensor
            rollout = run_inference(
                engine=self._engine,
                inputs=batched_inputs,
                obs_normalizer=self._obs_normalizer,
                state_normalizer=self._state_normalizer,
                future_len=self._context_settings.future_len,
                deterministic=self._cfg.deterministic,
            )

        denorm_traj = rollout["denorm_trajectories"][0, 0]
        step_delta = 0.1  # seconds per step (80 steps -> 8s horizon)

        states = []
        time_stamps = []
        base_time_us = iteration.time_point.time_us
        for idx in range(denorm_traj.shape[0]):
            x, y, cos_h, sin_h = denorm_traj[idx, :4].tolist()
            yaw = math.atan2(sin_h, cos_h)
            states.append(StateSE2(x, y, yaw))
            time_stamps.append(base_time_us + int((idx + 1) * step_delta * 1e6))

        return InterpolatedTrajectory(states, time_stamps)
