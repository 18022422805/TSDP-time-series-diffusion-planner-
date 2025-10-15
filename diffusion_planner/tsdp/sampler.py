from typing import Dict, List, Tuple
import torch

from diffusion_planner.tsdp.models import NoiseSchedulingPolicy, ScoreCorrectionPolicy, AnisotropyNetwork
from diffusion_planner.model.diffusion_planner import DiffusionBackbone


class TSDPDiffusionEngine:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        anisotropy: AnisotropyNetwork,
        schedule_policy: NoiseSchedulingPolicy,
        score_policy: ScoreCorrectionPolicy,
        future_len: int,
        predicted_neighbor_num: int,
        diffusion_steps: int,
        beta_min: float,
        beta_max: float,
        state_dim: int = 4,
        epsilon: float = 1e-3,
    ):
        self.backbone = backbone
        self.anisotropy = anisotropy
        self.schedule_policy = schedule_policy
        self.score_policy = score_policy
        self.future_len = future_len
        self.predicted_neighbor_num = predicted_neighbor_num
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.state_dim = state_dim
        self.epsilon = epsilon
        self.register_buffer = getattr(backbone, "register_buffer", None)

        steps = torch.linspace(epsilon, 1.0, diffusion_steps)
        self.time_schedule = steps

    def _prepare_current_states(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        ego_current = inputs["ego_current_state"][:, None, :self.state_dim]
        neighbors_current = inputs["neighbor_agents_past"][:, : self.predicted_neighbor_num, -1, :self.state_dim]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :self.state_dim], 0), dim=-1) == 0
        current_states = torch.cat([ego_current, neighbors_current], dim=1)
        return current_states, neighbor_current_mask

    def _backbone_score(
        self,
        inputs: Dict[str, torch.Tensor],
        x_t: torch.Tensor,
        t: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        model_inputs = {k: v for k, v in inputs.items()}
        model_inputs["neighbor_agents_past"] = model_inputs["neighbor_agents_past"].clone()
        model_inputs["sampled_trajectories"] = x_t
        model_inputs["diffusion_time"] = t
        model_inputs["neighbor_current_mask"] = neighbor_mask
        with torch.no_grad():
            _, decoder_outputs = self.backbone(model_inputs)
        score = decoder_outputs["score"][:, :, 1:, :]
        return score

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        stats: torch.Tensor,
        global_context: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        device = global_context.device
        batch = global_context.shape[0]

        current_states, neighbor_mask = self._prepare_current_states(inputs)
        r_tau = self.anisotropy(stats)
        r_tau = torch.clamp(r_tau, min=0.1, max=10.0)

        P = current_states.shape[1]
        x_t = torch.randn(batch, P, 1 + self.future_len, self.state_dim, device=device)
        x_t[:, :, 0, :] = current_states

        alpha_bar = torch.ones(batch, self.future_len, device=device)

        schedule_log_probs: List[torch.Tensor] = []
        score_log_probs: List[torch.Tensor] = []
        schedule_dists: List[torch.distributions.Distribution] = []
        score_dists: List[torch.distributions.Distribution] = []
        schedule_samples: List[torch.Tensor] = []

        for step in reversed(range(self.diffusion_steps)):
            time_value = self.time_schedule[step].to(device).expand(batch)
            base_beta = self.beta_min + (self.beta_max - self.beta_min) * time_value
            scale_sample, log_prob_scale, dist_scale = self.schedule_policy(global_context, time_value, stochastic=not deterministic)
            schedule_samples.append(scale_sample)
            schedule_log_probs.append(log_prob_scale)
            schedule_dists.append(dist_scale)

            beta_eff = base_beta.unsqueeze(-1) * scale_sample.unsqueeze(-1) * r_tau
            beta_eff = beta_eff.clamp(min=1e-4, max=0.999)
            alpha = (1.0 - beta_eff).clamp(min=1e-4, max=0.999)
            alpha_bar = alpha_bar * alpha

            t_tensor = time_value
            score = self._backbone_score(inputs, x_t, t_tensor, neighbor_mask)
            delta_flat, log_prob_delta, dist_delta = self.score_policy(global_context, x_t[:, :, 1:, :], stochastic=not deterministic)
            score_log_probs.append(log_prob_delta)
            score_dists.append(dist_delta)

            delta = delta_flat.reshape(batch, P, self.future_len, self.state_dim)
            guided_x0 = score + delta

            std = torch.sqrt((1.0 - alpha_bar).clamp(min=1e-6))
            noise = torch.zeros_like(guided_x0) if deterministic else torch.randn_like(guided_x0)
            x_future = torch.sqrt(alpha_bar[:, None, :, None]) * guided_x0 + std[:, None, :, None] * noise
            x_t = torch.cat([current_states[:, :, None, :], x_future], dim=2)

        return {
            "trajectories": x_t[:, :, 1:, :],
            "schedule_log_probs": torch.stack(schedule_log_probs, dim=0),
            "score_log_probs": torch.stack(score_log_probs, dim=0),
            "schedule_dists": schedule_dists,
            "score_dists": score_dists,
            "schedule_samples": torch.stack(schedule_samples, dim=0),
            "anisotropy": r_tau,
        }
