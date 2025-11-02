# TSDP: Time-Series Diffusion Planner

Time-Series Diffusion Planner (TSDP) is a two-stage diffusion-based planning system for autonomous driving. The project couples a transformer diffusion backbone with adaptive noise scheduling and score correction policies to generate future ego and neighbor trajectories that remain consistent with observed map, route, and static object context.

## Highlights
- Diffusion backbone with transformer encoders/decoders that fuse agent history, map lanes, static objects, and route waypoints.
- Anisotropy modeling to adapt diffusion noise across the planning horizon.
- Policy optimization stage (GRPO) that learns noise schedules and score corrections using patch-wise rewards.
- Ready-to-use runtime that stitches together both phases for open-loop inference or integration with the nuPlan devkit.
- Utilities for data loading, normalization, augmentation, and time-series feature engineering.

## Repository Layout
- `diffusion_planner/model/`: Diffusion backbone, encoder/decoder modules, and DiT building blocks.
- `diffusion_planner/tsdp/`: Phase-specific networks, samplers, GRPO utilities, reward shaping, runtime glue code.
- `diffusion_planner/nuplan_planner/`: Minimal nuPlan planner wrapper and context construction helpers.
- `diffusion_planner/utils/`: Dataset wrappers, normalization helpers, augmentation routines, and training utilities.
- `train_backbone.py`: Phase 1 training entry point for the diffusion backbone and anisotropy network.
- `train_policies_grpo.py`: Phase 2 policy optimization script for schedule/score policies.
- `inference_tsdp.py`: Standalone inference CLI that loads checkpoints and produces rollouts.
- `configs/tsdp_open_loop.yaml`: Hydra-compatible planner config for nuPlan experiments.
- `tsdp_config.yaml`: Central configuration for both training phases and inference defaults.
- `normalization.json`, `nuplan_train.json`: Example normalization stats and dataset manifest.
- `Methnology.pdf`: Slides describing the method (informational reference).

## Prerequisites & Installation
1. **Python**: Tested with Python 3.10+. A GPU with CUDA support is recommended for training.
2. **Environment setup** (example using `venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. **Core dependencies**:
   ```bash
   pip install -r requirements_torch.txt
   ```
   The requirements file points to the Tsinghua PyPI mirror for PyTorch wheels (`torch==2.0.0+cu118`). Adjust the index if needed.
4. Additional packages such as `nuplan-devkit`, `hydra-core`, and visualization or logging tools should be installed when integrating with the full nuPlan simulation stack.

## Data Preparation
- **Serialized samples**: `DiffusionPlannerData` expects a directory of serialized tensors (e.g., `*.pt`) containing keys such as `ego_current_state`, `neighbor_agents_past`, `lanes`, `static_objects`, etc. The helper `normalization.json` illustrates the expected normalization statistics for each modality.
- **Dataset manifest**: Provide a JSON list of relative paths via `--train_set_list` (e.g., `nuplan_train.json`). The loader reads files from `--train_set`.
- **Neighbor limits**: Phase scripts truncate inputs to `agent_num`, `predicted_neighbor_num`, and other limits defined by CLI flags or configs.

## Normalization
`ObservationNormalizer` and `StateNormalizer` rely on `normalization.json`. The JSON structure should include mean/std arrays for each observation modality plus `ego` and `neighbor` trajectory stats. During preprocessing the normalizers zero out masked entries to avoid contaminating statistics.

## Training Workflow
### Phase 1: Diffusion Backbone + Anisotropy
Trains the encoder/decoder backbone jointly with the anisotropy network.
```bash
python train_backbone.py \
  --config tsdp_config.yaml \
  --train_set /path/to/serialized/data \
  --train_set_list nuplan_train.json \
  --save_dir ./artifacts/phase1 \
  --device cuda \
  --normalization_file_path normalization.json
```
- Outputs checkpoints every 10 epochs plus `latest.pth` containing both module state_dicts.
- Uses optional state perturbation augmentation (`--use_data_augment/--augment_prob`).

### Phase 2: Policy Optimization (GRPO)
Fine-tunes diffusion sampling using learned scheduling and score correction policies.
```bash
python train_policies_grpo.py \
  --config tsdp_config.yaml \
  --phase1_ckpt ./artifacts/phase1/<timestamp>/latest.pth \
  --train_set /path/to/serialized/data \
  --train_set_list nuplan_train.json \
  --save_dir ./artifacts/phase2 \
  --device cuda \
  --normalization_file_path normalization.json
```
- Minimizes GRPO loss with patch-level rewards (`safety`, `comfort`, `progress` weights in `tsdp_config.yaml`).
- Produces checkpoints holding `schedule_policy` and `score_policy`.

Both phases honor overrides supplied via CLI (e.g., `--batch_size`, `--future_len`, `--agent_num`) and fall back to values in `tsdp_config.yaml` when omitted.

## Inference
Use `inference_tsdp.py` to combine both phases and produce open-loop trajectories.
```bash
python inference_tsdp.py \
  --config tsdp_config.yaml \
  --phase1_ckpt ./artifacts/phase1/<timestamp>/latest.pth \
  --phase2_ckpt ./artifacts/phase2/<timestamp>/latest.pth \
  --context_file ./sample_context.pt \
  --normalization_file_path normalization.json \
  --output ./tsdp_inference_output.pt \
  --device cuda
```
The script:
1. Loads normalizers and checkpoints.
2. Builds the diffusion engine with phase-1/phase-2 components.
3. Consumes a serialized context dictionary (matching loader keys).
4. Returns denormalized trajectories, anisotropy factors, and dynamic noise schedule samples.

## nuPlan Integration
- `diffusion_planner/nuplan_planner/tsdp_planner.py` wraps TSDP as a `nuplan-devkit` planner. The planner bootstraps context tensors via `context_builder.py`.
- The Hydra config `configs/tsdp_open_loop.yaml` expects environment variables (`PHASE1_CKPT`, `PHASE2_CKPT`, `NORMALIZATION_JSON`, `TSDP_CONFIG`, `TSDP_DEVICE`). Example launch:
  ```bash
  PHASE1_CKPT=./artifacts/phase1/latest.pth \
  PHASE2_CKPT=./artifacts/phase2/latest.pth \
  nuplan challenge --planner=tsdp_open_loop ...
  ```
- Ensure the nuPlan devkit is installed and map extraction utilities are configured for high-fidelity features.

## Configuration
- `tsdp_config.yaml` contains separate sections for:
  - `phase_1`: optimizer, regularization, and gradient clipping settings.
  - `phase_2`: diffusion steps, beta bounds, policy model sizes, reward weights.
  - `inference`: deterministic vs. stochastic sampling switches.
- Adjust these values to suit new datasets or training budgets.

## Tips & Troubleshooting
- Validate the normalization JSON after collecting data; inconsistent scale values can destabilize training.
- Checkpoint directories include timestamps—update downstream scripts accordingly or symlink to `latest.pth`.
- When porting to new datasets, start by confirming `DiffusionPlannerData` receives tensors with shapes consistent with `ContextSettings`.
- For deterministic rollouts set `deterministic: true` in the `inference` config or pass `--diffusion_model_type` / `--device` via CLI overrides.

## License
No license information is provided in this repository. Add a license file if you plan to distribute the project.

