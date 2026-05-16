# Changelog

## Unreleased

### Added

- Added ARX-5 bi-manual policy transforms for pi0.5 fine-tuning.
- Added `LeRobotArxDataConfig`, `pi05_arx_debug`, and `pi05_arx_debug_fresh_stats` training configs.
- Added a LeRobot v3 ARX bi-manual conversion script.
- Added focused tests for ARX transforms, config registration, and dataset conversion.
- Added project memory and remote validation docs for continued ARX/pi0.5 work.

### Notes

- The first ARX training path keeps 14D bi-manual state/action so it can reuse `asset_id="arx"` norm stats.
- The converter assumes gripper values are already in openpi ARX convention: `0.0=open`, `1.0=closed`.
