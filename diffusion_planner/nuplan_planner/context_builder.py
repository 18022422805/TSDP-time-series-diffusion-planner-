from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ContextSettings:
    agent_state_dim: int = 11
    agent_num: int = 32
    predicted_neighbor_num: int = 10
    time_len: int = 21
    future_len: int = 80
    lane_num: int = 70
    lane_len: int = 20
    lane_state_dim: int = 12
    route_num: int = 25
    route_len: int = 20
    route_state_dim: int = 12
    static_objects_num: int = 5
    static_objects_state_dim: int = 10


def _zero_tensor(device: torch.device, *shape: int) -> torch.Tensor:
    return torch.zeros(shape, device=device, dtype=torch.float32)


def _encode_ego_state(
    ego_state,
    settings: ContextSettings,
    device: torch.device,
) -> torch.Tensor:
    tensor = _zero_tensor(device, settings.agent_state_dim)
    rear_axle = ego_state.rear_axle
    heading = rear_axle.heading
    tensor[0] = rear_axle.x
    tensor[1] = rear_axle.y
    tensor[2] = math.cos(heading)
    tensor[3] = math.sin(heading)
    speed = ego_state.dynamic_car_state.speed
    tensor[4] = speed * tensor[2]
    tensor[5] = speed * tensor[3]
    tensor[6] = ego_state.dynamic_car_state.acceleration
    tensor[7] = ego_state.dynamic_car_state.angular_velocity
    return tensor


def _encode_neighbors(
    observation,
    settings: ContextSettings,
    device: torch.device,
) -> torch.Tensor:
    neighbors = _zero_tensor(device, settings.agent_num, settings.time_len, settings.agent_state_dim)
    agents = observation.tracked_objects.get_agents()
    limit = min(len(agents), settings.agent_num)
    for idx in range(limit):
        agent = agents[idx]
        heading = agent.center.heading
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        vx = agent.velocity * cos_h
        vy = agent.velocity * sin_h
        for t in range(settings.time_len):
            neighbors[idx, t, 0] = agent.center.x
            neighbors[idx, t, 1] = agent.center.y
            neighbors[idx, t, 2] = cos_h
            neighbors[idx, t, 3] = sin_h
            neighbors[idx, t, 4] = vx
            neighbors[idx, t, 5] = vy
            # occupancy flag
            neighbors[idx, t, 7] = 1.0
            # simple one-hot type encoding [neighbor, static, lane]
            neighbors[idx, t, -3:] = torch.tensor([1.0, 0.0, 0.0], device=device)
    return neighbors


def build_tsdp_context(
    history,
    observation,
    settings: ContextSettings,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Construct a model input dictionary from nuPlan simulation buffers.
    The current implementation focuses on delivering tensors with the correct shapes.
    Many features (lanes, static objects) are placeholders and should be replaced with
    richer map extraction for production use.
    """
    context: Dict[str, torch.Tensor] = {}

    context["ego_current_state"] = _encode_ego_state(observation.ego_state, settings, device)
    context["neighbor_agents_past"] = _encode_neighbors(observation, settings, device)

    lane_shape = (settings.lane_num, settings.lane_len, settings.lane_state_dim)
    context["lanes"] = _zero_tensor(device, *lane_shape)
    context["lanes_speed_limit"] = _zero_tensor(device, settings.lane_num, settings.lane_len)
    context["lanes_has_speed_limit"] = _zero_tensor(device, settings.lane_num, settings.lane_len)

    route_shape = (settings.route_num, settings.route_len, settings.route_state_dim)
    context["route_lanes"] = _zero_tensor(device, *route_shape)
    context["route_lanes_speed_limit"] = _zero_tensor(device, settings.route_num, settings.route_len)
    context["route_lanes_has_speed_limit"] = _zero_tensor(device, settings.route_num, settings.route_len)

    context["static_objects"] = _zero_tensor(device, settings.static_objects_num, settings.static_objects_state_dim)

    return context
