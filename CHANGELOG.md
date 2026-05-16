# Changelog

## Unreleased

### Added

- Added ARX-5 bi-manual policy transforms for pi0.5 fine-tuning.
- Added `LeRobotArxDataConfig`, `pi05_arx_debug`, and `pi05_arx_debug_fresh_stats` training configs.
- Added a LeRobot v3 ARX bi-manual conversion script.
- Added optional physical gripper command to openpi `0=open, 1=closed` mapping.
- Added the legacy LeRobot-to-HDF5 converter with the same optional gripper mapping.
- Made the ARX LeRobot converter create `meta/tasks.jsonl` for offline local dataset loading.
- Made the ARX LeRobot converter emit v2.1-style per-episode parquet/video paths so the pinned LeRobot loader does not fail on v3 `chunk_index/file_index` templates.
- Added a fast video move path when each source LeRobot video already corresponds to a single episode.
- Normalized converted per-episode timestamps to exact `frame_index / fps` values so LeRobot timestamp sync checks do not reject small capture jitter.
- Added focused tests for ARX transforms, config registration, and dataset conversion.
- Added project memory and remote validation docs for continued ARX/pi0.5 work.

### Notes

- The first ARX training path keeps 14D bi-manual state/action so it can reuse `asset_id="arx"` norm stats.
- Gripper mapping is explicit and opt-in with `--gripper-normalization physical`.
