# ARX pi0.5 Remote Validation

Run these commands on the remote training machine, not in this local Windows checkout.

## 1. Environment

```bash
conda activate openpi-jax
cd ~/code/openpi
git status --short --branch
```

Expected:

```text
## arx-pi05...origin/arx-pi05
```

## 2. Unit Tests

```bash
uv run pytest \
  src/openpi/policies/arx_policy_test.py \
  src/openpi/training/arx_config_test.py \
  scripts/make_arx_bimanual_lerobot_test.py \
  scripts/make_arx_lerobot_from_hdf5_test.py \
  -q
```

Expected: all tests pass.

## 3. Config Smoke Test

```bash
uv run python - <<'PY'
from openpi.training import config

cfg = config.get_config("pi05_arx_lora_debug")
print(cfg.name)
print(cfg.model)
print(cfg.data)
print(cfg.weight_loader)
PY
```

Expected:

```text
pi05_arx_lora_debug
```

The printed data config should include `asset_id='arx'` and `repo_id='local/arx_block_stack_bimanual'`.
The printed model config should include `gemma_2b_lora` and `gemma_300m_lora`.

## 4. Convert Dataset

See `docs/arx_pi05_dataset_registry.md` for the current dataset repo ids, config names, and prompts.

Set these paths to the real remote dataset locations:

```bash
export ARX_RAW=/path/to/original/trainable
export ARX_REPO=local/arx_block_stack_bimanual
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
export ARX_OUT="${HF_LEROBOT_HOME}/${ARX_REPO}"

uv run scripts/make_arx_bimanual_lerobot.py \
  --raw-dir "${ARX_RAW}" \
  --output-dir "${ARX_OUT}" \
  --overwrite \
  --gripper-normalization physical \
  --gripper-open-value -2.8 \
  --gripper-close-value 0.0
```

Expected:

```text
${ARX_OUT}/data/chunk-000/episode_000000.parquet
${ARX_OUT}/meta/info.json
${ARX_OUT}/meta/tasks.jsonl
${ARX_OUT}/meta/episodes.jsonl
${ARX_OUT}/meta/episodes_stats.jsonl
${ARX_OUT}/videos/...
```

The converter rewrites the output copy into the per-episode path template expected by
the pinned LeRobot loader. `meta/info.json` should contain:

```text
"codebase_version": "v2.1"
"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
"video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
```

If the source videos are already one file per episode, conversion should mostly move
video files into the loader-compatible path and finish quickly. If conversion is
still slow, it is falling back to segment re-encoding because multiple episodes
share a source video file or the source video frame count does not match episode
length.

The converter also normalizes per-episode `timestamp` to `frame_index / fps`.
This avoids LeRobot rejecting small capture-time jitter such as 0.031s or 0.035s
between nominal 30 FPS frames.

### Convert ACT HDF5 Episodes Instead

If the source dataset only exists as ACT-style `episode_*.hdf5` files, convert it
directly into the same OpenPI-readable local LeRobot layout:

```bash
export ARX_HDF5=/path/to/hdf5_episodes
export ARX_REPO=local/arx_block_stack_bimanual_new
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
export ARX_OUT="${HF_LEROBOT_HOME}/${ARX_REPO}"

uv run scripts/make_arx_lerobot_from_hdf5.py \
  --hdf5-dir "${ARX_HDF5}" \
  --output-dir "${ARX_OUT}" \
  --overwrite \
  --gripper-normalization physical \
  --gripper-open-value -2.8 \
  --gripper-close-value 0.0
```

This reads `/observations/qpos`, `/action`, and JPEG-padded `head` /
`right_wrist` frames. It writes `observation.state`, `action`,
`observation.images.camera_h`, and `observation.images.camera_r`.

## 5. Dataset Shape Check

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import polars as pl

repo = Path(os.environ["HF_LEROBOT_HOME"]) / "local/arx_block_stack_bimanual"
parquet = sorted((repo / "data").glob("chunk-*/*.parquet"))[0]
df = pl.read_parquet(parquet, n_rows=3)
print(df.select(["observation.state", "action"]))
print("state len:", len(df["observation.state"][0]))
print("action len:", len(df["action"][0]))
PY
```

Expected:

```text
state len: 14
action len: 14
```

Check a few gripper values:

```bash
uv run python - <<'PY'
import os
from pathlib import Path
import polars as pl

repo = Path(os.environ["HF_LEROBOT_HOME"]) / "local/arx_block_stack_bimanual"
parquet = sorted((repo / "data").glob("chunk-*/*.parquet"))[0]
df = pl.read_parquet(parquet, n_rows=10)
states = df["observation.state"].to_list()
actions = df["action"].to_list()
print("state grippers:", [(row[6], row[13]) for row in states[:5]])
print("action grippers:", [(row[6], row[13]) for row in actions[:5]])
PY
```

Expected: gripper values are within `[0.0, 1.0]`.

Check task metadata:

```bash
cat "${ARX_OUT}/meta/tasks.jsonl"
```

Expected:

```text
{"task_index":0,"task":"place the red block on the blue block"}
```

## 6. Data Loader Smoke Test

This checks LeRobot loading, ARX transforms, ARX norm stats, tokenization, image resize, and action padding.

```bash
uv run python - <<'PY'
import dataclasses
from openpi.training import config
from openpi.training import data_loader

cfg = config.get_config("pi05_arx_lora_debug")
cfg = dataclasses.replace(cfg, batch_size=2, num_workers=0)
loader = data_loader.create_data_loader(cfg, num_batches=1, shuffle=False)
batch = next(iter(loader))
obs, actions = batch
print("state:", obs.state.shape)
print("actions:", actions.shape)
print("images:", {k: v.shape for k, v in obs.images.items()})
print("masks:", {k: v.shape for k, v in obs.image_masks.items()})
PY
```

Expected:

```text
state: (2, 32)
actions: (2, 50, 32)
```

## 7. First LoRA Training Run

Run this short smoke test first. It checks checkpoint writing without committing to a long run.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
uv run scripts/train.py \
  pi05_arx_lora_debug \
  --exp-name arx_lora_smoke \
  --batch-size 2 \
  --num-train-steps 2 \
  --save-interval 1 \
  --log-interval 1 \
  --overwrite
```

Then start the debug run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
uv run scripts/train.py \
  pi05_arx_lora_debug \
  --exp-name arx_block_stack_lora_debug \
  --overwrite
```

Watch for:

- No dataset key errors.
- No norm stats loading errors.
- Loss starts logging.
- Checkpoints save under `checkpoints/pi05_arx_lora_debug/arx_block_stack_lora_debug`.

`pi05_arx_lora_debug` runs 20k steps and saves every 5k steps.
The smoke test above still overrides those values from the CLI.

## 8. Optional Dataset-specific Norm Stats Comparison

Only run this after the reused ARX stats path works, or if training clearly fails because stats are mismatched.

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_arx_lora_debug_fresh_stats
```

This writes stats under `assets/pi05_arx_lora_debug_fresh_stats/local/arx_block_stack_bimanual`.

Then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
uv run scripts/train.py \
  pi05_arx_lora_debug_fresh_stats \
  --exp-name arx_block_stack_lora_fresh_stats_debug \
  --overwrite
```

## Failure Notes

- If gripper values are still in physical command space, rerun conversion with `--gripper-normalization physical`.
- If `meta/tasks.jsonl` is missing, rerun conversion after pulling the latest branch.
- If the loader raises `KeyError: 'chunk_index'`, the output still has the old LeRobot v3 `data_path`.
  Rerun conversion after pulling the latest branch and using `--overwrite`.
- If the loader prints `diff`, `episode_index`, and `timestamps` entries and exits, timestamps are still jittered.
  Rerun conversion after pulling the latest branch and using `--overwrite`.
- If LeRobot cannot find `local/arx_block_stack_bimanual`, check `HF_LEROBOT_HOME` and the output path.
- If `cam_high` is unavailable or poor, adjust `LeRobotArxDataConfig.repack_transforms` to use only `camera_r`.
- If full `pi05_arx_debug` OOMs during `init_train_state`, use `pi05_arx_lora_debug`.
