# ARX pi0.5 Deployment Notes

This document describes the first deployment path for the ARX-5 block stacking
demo trained with `pi05_arx_lora_debug`.

The intended setup is:

```text
4090 workstation: openpi policy server, JAX model inference
ARX controller PC: camera/state collection, ROS2/robot control, websocket client
```

The two machines must be on the same network and able to ping each other.

## 1. Checkpoint Choice

Use the best available checkpoint directory. For example, if the last complete
checkpoint is step 16000:

```bash
export OPENPI_ROOT=/home/user/code/openpi
export ARX_CKPT="${OPENPI_ROOT}/checkpoints/pi05_arx_lora_debug/arx_block_stack_lora_debug/16000"
```

The checkpoint directory should contain at least:

```text
assets/
params/
```

Check it with:

```bash
find "${ARX_CKPT}" -maxdepth 2 -type d | sort
```

## 2. Start Policy Server On The 4090 Machine

Run this on the GPU workstation:

```bash
conda activate openpi-jax
cd /home/user/code/openpi
git pull

export ARX_CKPT=/home/user/code/openpi/checkpoints/pi05_arx_lora_debug/arx_block_stack_lora_debug/16000

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}" \
uv run scripts/serve_policy.py \
  --port 8000 \
  --default-prompt "place the red block on the blue block" \
  policy:checkpoint \
  --policy.config=pi05_arx_lora_debug \
  --policy.dir="${ARX_CKPT}"
```

The server binds to `0.0.0.0`, so the robot controller can connect to the GPU
machine IP.

Inference normally only needs one GPU. If GPU 0 is busy or attached to display
work, use `CUDA_VISIBLE_DEVICES=1` instead.

Health check from the GPU machine:

```bash
curl http://127.0.0.1:8000/healthz
```

Health check from the ARX controller PC:

```bash
curl http://<GPU_MACHINE_IP>:8000/healthz
```

## 3. Install The Lightweight Client On The ARX Controller PC

The controller PC does not need the full training environment. It needs robot
dependencies plus `openpi-client`.

```bash
cd /path/to/openpi/packages/openpi-client
pip install -e .
```

If the controller PC should not clone the full repo, copy only
`packages/openpi-client` and install that package.

## 4. Observation Format Expected By The ARX Policy

The websocket client sends a Python dictionary. It must match
`src/openpi/policies/arx_policy.py`.

Required keys:

```python
{
    "state": np.ndarray,   # shape (14,), float32
    "images": dict,
    "prompt": str,
}
```

State ordering:

```text
0:6   left arm joints
6     left gripper, openpi convention: 0=open, 1=closed
7:13  right arm joints
13    right gripper, openpi convention: 0=open, 1=closed
```

Images used by the current training run:

```text
cam_high        -> head / middle camera
cam_right_wrist -> right wrist camera
```

`cam_left_wrist` is optional for this policy. If omitted, openpi inserts a black
image and marks that view as unavailable.

Images may be `[H, W, C]` or `[C, H, W]`. Use RGB `uint8`. If OpenCV captures
BGR images, convert BGR to RGB before sending.

## 5. Minimal Dry-Run Client

Run this first on the controller PC without commanding the robot.

```python
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


GPU_HOST = "GPU_MACHINE_IP"
TASK = "place the red block on the blue block"


def preprocess_rgb(image_rgb: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image_rgb, 224, 224))


client = websocket_client_policy.WebsocketClientPolicy(host=GPU_HOST, port=8000)
print("server metadata:", client.get_server_metadata())

# Replace these with real robot observations.
state14 = np.zeros((14,), dtype=np.float32)
head_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
right_wrist_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)

observation = {
    "state": state14,
    "images": {
        "cam_high": preprocess_rgb(head_rgb),
        "cam_right_wrist": preprocess_rgb(right_wrist_rgb),
    },
    "prompt": TASK,
}

result = client.infer(observation)
actions = result["actions"]

print("actions shape:", actions.shape)
print("first action:", actions[0])
print("right arm first action:", actions[0, 7:14])
print("policy timing:", result.get("policy_timing"))
print("server timing:", result.get("server_timing"))
```

Expected action shape:

```text
(50, 14)
```

If this fails, do not connect robot actuation yet. Fix the server/client/data
format first.

## 6. Mapping Model Actions To ARX Commands

The model outputs a 50-step action chunk:

```python
actions = result["actions"]  # shape (50, 14)
```

For the first real demo, execute only the right arm and hold the left arm at its
current pose:

```python
action0 = actions[0]

left_command = current_left_qpos.copy()
right_command = action0[7:14].copy()
right_command[6] = np.clip(right_command[6], 0.0, 1.0)
```

If the low-level ARX gripper controller expects physical gripper values instead
of openpi normalized values, map back from `0=open, 1=closed`:

```python
physical_open = -2.8
physical_closed = 0.0
right_gripper_physical = physical_open + right_command[6] * (physical_closed - physical_open)
```

Use the exact gripper range that matches the data export and robot controller.

## 7. Recommended Control Loop

Do not execute all 50 predicted actions open-loop at first.

Recommended first pass:

```text
1. Read latest cameras and joint state.
2. Send observation to policy server.
3. Receive action chunk.
4. Execute only the first 1-5 right-arm actions with low speed limits.
5. Re-observe and query the policy again.
```

Start with a low command rate and low velocity/acceleration limits. Increase only
after actions look stable.

## 8. Safety Checklist

Before enabling robot motion:

- Confirm emergency stop works.
- Run the dry-run client and inspect action ranges.
- Print the first several right-arm commands before sending them.
- Clamp joint commands to robot joint limits.
- Clamp gripper commands to the expected controller range.
- Hold the left arm fixed unless explicitly testing bimanual behavior.
- Start with the robot far from self-collision and table collision.
- Test one action at a time before running receding-horizon control.

Useful debug print:

```python
print("state14:", state14)
print("right action min/max:", actions[:, 7:14].min(axis=0), actions[:, 7:14].max(axis=0))
print("left action min/max:", actions[:, 0:7].min(axis=0), actions[:, 0:7].max(axis=0))
```

## 9. Network And Latency Checks

From the controller PC:

```bash
ping <GPU_MACHINE_IP>
curl http://<GPU_MACHINE_IP>:8000/healthz
```

Measure policy latency with the dry-run client. The websocket response includes:

```text
policy_timing["infer_ms"]
server_timing["infer_ms"]
server_timing["prev_total_ms"]
```

If latency is high, resize images on the controller PC before sending, keep
websocket messages uncompressed, and execute a small action horizon open-loop
between policy calls.

## 10. Common Failures

### Server cannot load checkpoint

Check that `--policy.dir` points at the numbered step directory, not the
experiment root:

```text
correct: checkpoints/pi05_arx_lora_debug/arx_block_stack_lora_debug/16000
wrong:   checkpoints/pi05_arx_lora_debug/arx_block_stack_lora_debug
```

### Client cannot connect

Check firewall, IP address, port, and that the server is listening on `0.0.0.0`.

### Colors look wrong

OpenCV returns BGR. Convert to RGB before sending:

```python
image_rgb = image_bgr[..., ::-1]
```

### Actions have wrong scale

Verify that the state and gripper values sent at inference use the same
convention as training:

```text
state shape: (14,)
gripper convention: 0=open, 1=closed
joint ordering: left first, right second
```

### Left arm moves unexpectedly

Ignore model left-arm output for the first demo and send `current_left_qpos` to
the left arm controller.
