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
        time_grid = self.time_schedule.to(device)

        schedule_samples: List[torch.Tensor] = []
        schedule_log_probs: List[torch.Tensor] = []
        schedule_dists: List[torch.distributions.Distribution] = []
        alpha_list: List[torch.Tensor] = []
        alpha_bar_list: List[torch.Tensor] = []

        alpha_cum = torch.ones(batch, self.future_len, device=device)
        for step in range(self.diffusion_steps):
            time_value = time_grid[step].expand(batch)
            scale_sample, log_prob_scale, dist_scale = self.schedule_policy(
                global_context, time_value, stochastic=not deterministic
            )
            schedule_samples.append(scale_sample)
            schedule_log_probs.append(log_prob_scale)
            schedule_dists.append(dist_scale)

            beta_base = self.beta_min + (self.beta_max - self.beta_min) * time_value
            beta_eff = beta_base.unsqueeze(-1) * scale_sample.unsqueeze(-1) * r_tau
            beta_eff = beta_eff.clamp(min=1e-4, max=0.999)
            alpha = (1.0 - beta_eff).clamp(min=1e-4, max=0.999)

            alpha_cum = alpha_cum * alpha

            alpha_list.append(alpha)
            alpha_bar_list.append(alpha_cum.clone())

        x_future = torch.randn(batch, P, self.future_len, self.state_dim, device=device)
        score_log_prob_steps: List[torch.Tensor] = []
        score_dists: List[torch.distributions.Distribution] = []

        model_type = getattr(self.backbone.decoder.decoder.dit, "model_type", "x_start")
        if model_type != "x_start":
            raise ValueError(f"TSDP sampling currently supports only 'x_start' backbone models, got '{model_type}'.")

        eps_denom_eps = 1e-6

        for step in reversed(range(self.diffusion_steps)):
            time_value = time_grid[step].expand(batch)
            x_full = torch.cat([current_states[:, :, None, :], x_future], dim=2)
            score = self._backbone_score(inputs, x_full, time_value, neighbor_mask)

            delta_flat, log_prob_delta, dist_delta = self.score_policy(
                global_context, x_future, stochastic=not deterministic
            )
            score_log_prob_steps.append(log_prob_delta)
            score_dists.append(dist_delta)

            delta = delta_flat.reshape(batch, P, self.future_len, self.state_dim)
            guided_x0 = score + delta

            alpha = alpha_list[step]
            alpha_bar_t = alpha_bar_list[step]
            if step > 0:
                alpha_bar_prev = alpha_bar_list[step - 1]
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            alpha_expand = alpha.unsqueeze(1).unsqueeze(-1)
            alpha_bar_t_expand = alpha_bar_t.unsqueeze(1).unsqueeze(-1)
            alpha_bar_prev_expand = alpha_bar_prev.unsqueeze(1).unsqueeze(-1)

            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t_expand + eps_denom_eps)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t_expand + eps_denom_eps)
            eps = (x_future - sqrt_alpha_bar_t * guided_x0) / sqrt_one_minus_alpha_bar_t

            coef = (1 - alpha_expand) / sqrt_one_minus_alpha_bar_t
            mean = (x_future - coef * eps) / torch.sqrt(alpha_expand + eps_denom_eps)

            if deterministic:
                x_future = mean
            else:
                beta_tilde = ((1 - alpha_bar_prev_expand) / (1 - alpha_bar_t_expand + eps_denom_eps)) * (1 - alpha_expand)
                noise = torch.randn_like(x_future)
                x_future = mean + torch.sqrt(beta_tilde.clamp(min=eps_denom_eps)) * noise

        schedule_samples_tensor = torch.stack(schedule_samples, dim=0)
        schedule_log_probs_tensor = torch.stack(schedule_log_probs, dim=0)
        score_log_probs_tensor = torch.stack(list(reversed(score_log_prob_steps)), dim=0)
        score_dists = list(reversed(score_dists))

        return {
            "trajectories": x_future,
            "schedule_log_probs": schedule_log_probs_tensor,
            "score_log_probs": score_log_probs_tensor,
            "schedule_dists": schedule_dists,
            "score_dists": score_dists,
            "schedule_samples": schedule_samples_tensor,
            "anisotropy": r_tau,
        }
