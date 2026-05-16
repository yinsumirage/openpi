# ARX pi0.5 Fine-tuning Project Notes

## Goal

Fine-tune pi0.5 on a 100-episode ARX-5 block stacking dataset for:

```text
place the red block on the blue block
```

The first implementation keeps the robot bi-manual instead of reducing to right-arm-only. This matches the ARX pretraining action space and allows reuse of `asset_id="arx"` normalization stats.

## Current Design

- Model: `pi0_config.Pi0Config(pi05=True)`
- Weight loader: `gs://openpi-assets/checkpoints/pi05_base/params`
- Assets: `gs://openpi-assets/checkpoints/pi05_base/assets`
- Asset id: `arx`
- Config: `pi05_arx_debug`
- Fresh norm-stats comparison config: `pi05_arx_debug_fresh_stats`
- Dataset repo id: `local/arx_block_stack_bimanual`
- Default prompt: `place the red block on the blue block`
- Batch size: `8`
- Train steps: `5000`
- Save interval: `500`

## Data Contract

The converted dataset must expose:

```text
observation.state: float32[14]
action: float32[14]
observation.images.camera_h
observation.images.camera_l
observation.images.camera_r
```

The current config uses:

```text
cam_high        <- observation.images.camera_h
cam_right_wrist <- observation.images.camera_r
```

`cam_left_wrist` is optional in the transform and is masked out when absent from the repacked input.

## State And Action Ordering

```text
0:7   = left arm, 6 joints + gripper
7:14  = right arm, 6 joints + gripper
```

The converter builds:

```python
observation.state = observation.master_left_state[:7] + observation.master_right_state[:7]
action = action.joint_actions[:14]
```

## Gripper Requirement

Before training, re-export the dataset so gripper values follow the openpi ARX convention:

```text
0.0 = fully open
1.0 = fully closed
```

The current converter intentionally does not convert physical command values such as `-2.3` or `-0.3`. If old gripper command-space data is used, ARX norm stats are not valid.

## Why Not ALOHA Transforms

`AlohaInputs` and `AlohaOutputs` are tied to Trossen/ALOHA conventions. They apply a 14D joint flip mask and gripper conversion that is not appropriate for ARX. ARX uses its own transform to preserve the physical ARX action space.

## Next Experiments

1. Validate the converter and config on the remote environment.
2. Run a 5k-step `pi05_arx_debug` training job with reused ARX norm stats.
3. Inspect loss, checkpoint assets, and simple policy loading.
4. If the first run is stable, try 10k or 20k steps.
5. If reused ARX norm stats are unstable, compute dataset-specific norm stats with `pi05_arx_debug_fresh_stats`.
