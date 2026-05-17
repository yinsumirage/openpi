# Changelog

## Unreleased

### Added

- Added ARX-5 bi-manual policy transforms for pi0.5 fine-tuning.
- Added `LeRobotArxDataConfig`, full fine-tune ARX debug configs, and low-memory pi0.5 LoRA ARX debug configs.
- Set the ARX pi0.5 LoRA debug configs to 20k steps with checkpoint saves every 2k steps.
- Added a LeRobot v3 ARX bi-manual conversion script.
- Added optional physical gripper command to openpi `0=open, 1=closed` mapping.
- Added the legacy LeRobot-to-HDF5 converter with the same optional gripper mapping.
- Added an ACT HDF5-to-LeRobot converter for recovering OpenPI training datasets from `episode_*.hdf5`.
- Made the ARX LeRobot converter create `meta/tasks.jsonl` for offline local dataset loading.
- Made the ARX LeRobot converter emit v2.1-style per-episode parquet/video paths.
- This avoids pinned LeRobot loader failures on v3 `chunk_index/file_index` templates.
- Added a fast video move path when each source LeRobot video already corresponds to a single episode.
- Normalized converted per-episode timestamps to exact `frame_index / fps` values.
- This avoids LeRobot timestamp sync failures on small capture jitter.
- Added focused tests for ARX transforms, config registration, and dataset conversion.
- Added project memory and remote validation docs for continued ARX/pi0.5 work.
- Added ARX pi0.5 remote deployment notes for the policy server and controller client.
- Added `pi05_arx_lora_bad_debug` for comparing a lower-quality block-stacking dataset.

### Notes

- The first ARX training path keeps 14D bi-manual state/action so it can reuse `asset_id="arx"` norm stats.
- Gripper mapping is explicit and opt-in with `--gripper-normalization physical`.
