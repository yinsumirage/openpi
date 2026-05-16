# OpenPI ARX pi0.5 Agent Notes

This repository is being adapted to fine-tune pi0.5 for an ARX-5 bi-manual block stacking demo.

## Current Project State

- Branch: `arx-pi05`
- Target model path: `gs://openpi-assets/checkpoints/pi05_base/params`
- Target assets path: `gs://openpi-assets/checkpoints/pi05_base/assets`
- Target asset id: `arx`
- Training config added locally: `pi05_arx_debug`
- Fresh norm-stats comparison config: `pi05_arx_debug_fresh_stats`
- Dataset repo id expected by the config: `local/arx_block_stack_bimanual`
- Task prompt: `place the red block on the blue block`

## Environment Rules

- Do not commit local datasets, checkpoints, wandb logs, outputs, runs, or `.venv`.
- The remote training machine uses:
  - `conda activate openpi-jax`
  - `uv run ...`
- Do not manually activate `openpi/.venv`; use `uv run` inside the conda environment.
- This local Windows checkout does not have the training environment, so most runtime verification must happen remotely.

## ARX Data Assumptions

- First training path keeps the robot bi-manual: 14D state and 14D action.
- State/action ordering follows openpi ARX norm stats:
  - dims 0:6: left arm joints
  - dim 6: left gripper
  - dims 7:13: right arm joints
  - dim 13: right gripper
- Gripper values must be converted into openpi convention before training:
  - `0.0 = fully open`
  - `1.0 = fully closed`
- `scripts/make_arx_bimanual_lerobot.py` can do this with:
  - `--gripper-normalization physical`
  - `--gripper-open-value -2.8`
  - `--gripper-close-value 0.0`

## Files Added For ARX

- `src/openpi/policies/arx_policy.py`
  - Converts ARX images/state/action into openpi model inputs.
  - Does not apply ALOHA/Trossen joint flips or gripper transforms.
- `src/openpi/training/config.py`
  - Adds `LeRobotArxDataConfig`.
  - Adds `pi05_arx_debug`.
- `scripts/make_arx_bimanual_lerobot.py`
  - Copies a LeRobot v3 dataset.
  - Adds `observation.state = master_left_state[:7] + master_right_state[:7]`.
  - Adds `action = action.joint_actions[:14]`.
  - Optionally maps ARX physical gripper values into openpi `0=open, 1=closed`.
- `scripts/convert_lerobot_to_hdf5.py`
  - Stores the legacy LeRobot-to-HDF5 converter used by the ACT pipeline.
  - Supports the same optional physical gripper mapping.
- Tests:
  - `src/openpi/policies/arx_policy_test.py`
  - `src/openpi/training/arx_config_test.py`
  - `scripts/make_arx_bimanual_lerobot_test.py`

## Remote Validation Entry Point

Run this on the training machine after syncing this branch:

```bash
conda activate openpi-jax
cd ~/code/openpi

uv run pytest \
  src/openpi/policies/arx_policy_test.py \
  src/openpi/training/arx_config_test.py \
  scripts/make_arx_bimanual_lerobot_test.py \
  -q
```

Then follow `docs/arx_pi05_remote_validation.md`.
