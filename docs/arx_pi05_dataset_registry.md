# ARX pi0.5 Dataset Registry

This registry tracks the local LeRobot dataset names, training configs, and prompts used for ARX pi0.5 experiments.

All datasets below use the 14D bi-manual ARX state/action layout:

```text
0:6   left arm joints
6     left gripper, openpi convention: 0=open, 1=closed
7:13  right arm joints
13    right gripper, openpi convention: 0=open, 1=closed
```

## Datasets

| Dataset | Repo ID | Config | Prompt | Source / Notes |
| --- | --- | --- | --- | --- |
| Block stack baseline | `local/arx_block_stack_bimanual` | `pi05_arx_lora_debug` | `place the red block on the blue block` | Main successful block-stack dataset. Convert from LeRobot trainable with `scripts/make_arx_bimanual_lerobot.py`. |
| Block stack lower-quality comparison | `local/arx_block_stack_bimanual_bad` | `pi05_arx_lora_bad_debug` | `place the red block on the blue block` | Lower-quality comparison dataset. Convert from LeRobot trainable with `scripts/make_arx_bimanual_lerobot.py`, or recover from ACT HDF5 with `scripts/make_arx_lerobot_from_hdf5.py`. |
| Watermelon to basket | `local/arx_watermelon_basket_bimanual` | `pi05_arx_lora_watermelon_basket_debug` | `scoop up the watermelon on the right and place it in the basket on the left` | New task: scoop the watermelon on the right and put it into the basket on the left. Convert from LeRobot trainable with `scripts/make_arx_bimanual_lerobot.py`. |

## Convert LeRobot Trainable Data

Use this for source directories with `data/`, `meta/`, and `videos/`:

```bash
export ARX_RAW=/path/to/original/trainable
export ARX_REPO=local/arx_watermelon_basket_bimanual
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

## Convert ACT HDF5 Data

Use this only when the source data exists as `episode_*.hdf5` files:

```bash
export ARX_HDF5=/path/to/hdf5_episodes
export ARX_REPO=local/arx_block_stack_bimanual_bad
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

## Train

Example for the watermelon task:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
WANDB_MODE=online \
uv run scripts/train.py \
  pi05_arx_lora_watermelon_basket_debug \
  --exp-name arx_watermelon_basket_lora_debug \
  --wandb-enabled \
  --overwrite
```

ARX debug configs save checkpoints every 5k steps to reduce disk usage.
