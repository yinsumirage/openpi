"""Convert a LeRobot trainable dataset into legacy episode_*.hdf5 files for ACT training."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


CAMERA_NAME_MAP = {
    "head": "camera_h",
    "left_wrist": "camera_l",
    "right_wrist": "camera_r",
    "camera_h": "camera_h",
    "camera_l": "camera_l",
    "camera_r": "camera_r",
}


@dataclasses.dataclass(frozen=True)
class GripperMapping:
    open_value: float
    close_value: float

    def __post_init__(self) -> None:
        if self.open_value == self.close_value:
            raise ValueError("gripper_open_value and gripper_close_value must be different.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory containing the trainable/ dataset.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for episode_*.hdf5.",
    )
    parser.add_argument(
        "--camera-names",
        "--camera_names",
        nargs="+",
        default=["right_wrist"],
        choices=list(CAMERA_NAME_MAP),
        help="Cameras to export into HDF5.",
    )
    parser.add_argument("--episode-start", "--episode_start", type=int, default=0, help="Starting episode index.")
    parser.add_argument("--max-episodes", "--max_episodes", type=int, default=-1, help="Limit episodes; -1 means all.")
    parser.add_argument("--jpeg-quality", "--jpeg_quality", type=int, default=50, help="JPEG quality.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing episode files.")
    parser.add_argument(
        "--gripper-normalization",
        choices=("none", "physical"),
        default="none",
        help="Map physical gripper command values into openpi 0=open, 1=closed convention.",
    )
    parser.add_argument("--gripper-open-value", "--gripper_open_value", type=float, default=-2.8)
    parser.add_argument("--gripper-close-value", "--gripper_close_value", type=float, default=0.0)
    return parser.parse_args()


def load_info(trainable_dir: Path) -> dict[str, Any]:
    with (trainable_dir / "meta" / "info.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_episode_rows(trainable_dir: Path) -> list[dict[str, Any]]:
    episode_dir = trainable_dir / "meta" / "episodes"
    rows = []
    for parquet_path in sorted(episode_dir.rglob("*.parquet")):
        rows.extend(pq.read_table(parquet_path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def format_data_path(trainable_dir: Path, pattern: str, chunk_index: int, file_index: int) -> Path:
    return trainable_dir / pattern.format(chunk_index=chunk_index, file_index=file_index)


def format_video_path(trainable_dir: Path, pattern: str, video_key: str, chunk_index: int, file_index: int) -> Path:
    return trainable_dir / pattern.format(video_key=video_key, chunk_index=chunk_index, file_index=file_index)


def load_episode_numeric(data_path: Path, gripper_mapping: GripperMapping | None) -> dict[str, np.ndarray]:
    table = pq.read_table(data_path)
    payload = table.to_pydict()
    left = np.asarray(payload["observation.master_left_state"], dtype=np.float32)
    right = np.asarray(payload["observation.master_right_state"], dtype=np.float32)
    action = np.asarray(payload["action.joint_actions"], dtype=np.float32)

    if left.shape[1] != 27 or right.shape[1] != 27:
        raise ValueError(f"Unexpected master state shape in {data_path}: {left.shape}, {right.shape}")
    if action.shape[1] != 14:
        raise ValueError(f"Unexpected action shape in {data_path}: {action.shape}")

    left_qpos = left[:, :7].copy()
    right_qpos = right[:, :7].copy()
    action = action.copy()
    if gripper_mapping is not None:
        left_qpos[:, 6] = map_gripper_array(left_qpos[:, 6], gripper_mapping)
        right_qpos[:, 6] = map_gripper_array(right_qpos[:, 6], gripper_mapping)
        action[:, 6] = map_gripper_array(action[:, 6], gripper_mapping)
        action[:, 13] = map_gripper_array(action[:, 13], gripper_mapping)

    qpos = np.concatenate((left_qpos, right_qpos), axis=1)
    qvel = np.concatenate((left[:, 7:14], right[:, 7:14]), axis=1)
    effort = np.concatenate((left[:, 14:21], right[:, 14:21]), axis=1)
    eef = np.concatenate((left[:, 21:27], left_qpos[:, 6:7], right[:, 21:27], right_qpos[:, 6:7]), axis=1)

    return {
        "action": action,
        "action_eef": eef.copy(),
        "qpos": qpos,
        "qvel": qvel,
        "effort": effort,
        "eef": eef,
    }


def read_video_frames(video_path: Path, expected_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    try:
        while len(frames) < expected_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if len(frames) != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames from {video_path}, got {len(frames)}")
    return frames


def encode_and_pad_frames(frames: list[np.ndarray], jpeg_quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    encoded = []
    max_size = 0
    for frame in frames:
        ok, buffer = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            raise ValueError("Failed to encode frame to JPEG")
        encoded.append(buffer)
        max_size = max(max_size, len(buffer))

    padded = np.zeros((len(encoded), max_size), dtype=np.uint8)
    for index, buffer in enumerate(encoded):
        padded[index, : len(buffer)] = buffer
    return padded


def build_zero_block(length: int, width: int) -> np.ndarray:
    return np.zeros((length, width), dtype=np.float32)


def write_episode_hdf5(output_path: Path, task_name: str, numeric_payload: dict, image_payload: dict) -> None:
    frame_count = numeric_payload["action"].shape[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["task"] = task_name

        obs_group = root.create_group("observations")
        image_group = obs_group.create_group("images")

        for camera_name, padded_images in image_payload.items():
            image_group.create_dataset(
                camera_name,
                padded_images.shape,
                dtype="uint8",
                chunks=(1, padded_images.shape[1]),
            )
            image_group[camera_name][...] = padded_images

        obs_group.create_dataset("qpos", data=numeric_payload["qpos"])
        obs_group.create_dataset("eef", data=numeric_payload["eef"])
        obs_group.create_dataset("qvel", data=numeric_payload["qvel"])
        obs_group.create_dataset("effort", data=numeric_payload["effort"])
        obs_group.create_dataset("robot_base", data=build_zero_block(frame_count, 6))
        obs_group.create_dataset("base_velocity", data=build_zero_block(frame_count, 4))

        root.create_dataset("action", data=numeric_payload["action"])
        root.create_dataset("action_eef", data=numeric_payload["action_eef"])
        root.create_dataset("action_base", data=build_zero_block(frame_count, 6))
        root.create_dataset("action_velocity", data=build_zero_block(frame_count, 4))


def convert_episode(
    trainable_dir: Path,
    info: dict,
    row: dict,
    output_path: Path,
    camera_names: list[str],
    jpeg_quality: int,
    gripper_mapping: GripperMapping | None,
) -> None:
    data_path = format_data_path(
        trainable_dir,
        info["data_path"],
        int(row["data/chunk_index"]),
        int(row["data/file_index"]),
    )
    numeric_payload = load_episode_numeric(data_path, gripper_mapping)
    expected_frames = numeric_payload["action"].shape[0]

    if int(row["length"]) != expected_frames:
        raise ValueError(f"Episode length mismatch for {data_path}: meta={row['length']} parquet={expected_frames}")

    image_payload = {}
    for requested_name in camera_names:
        lerobot_camera = CAMERA_NAME_MAP[requested_name]
        video_key = f"observation.images.{lerobot_camera}"
        video_path = format_video_path(
            trainable_dir,
            info["video_path"],
            video_key,
            int(row[f"videos/{video_key}/chunk_index"]),
            int(row[f"videos/{video_key}/file_index"]),
        )
        frames = read_video_frames(video_path, expected_frames)
        image_payload[requested_name] = encode_and_pad_frames(frames, jpeg_quality)

    tasks = row.get("tasks") or []
    task_name = tasks[0] if tasks else output_path.parent.name
    write_episode_hdf5(output_path, task_name, numeric_payload, image_payload)


def make_gripper_mapping(
    gripper_normalization: str,
    *,
    gripper_open_value: float,
    gripper_close_value: float,
) -> GripperMapping | None:
    if gripper_normalization == "none":
        return None
    if gripper_normalization == "physical":
        return GripperMapping(open_value=gripper_open_value, close_value=gripper_close_value)
    raise ValueError(f"Unsupported gripper normalization: {gripper_normalization}")


def map_gripper_array(values: np.ndarray, mapping: GripperMapping) -> np.ndarray:
    normalized = (values.astype(np.float32) - mapping.open_value) / (mapping.close_value - mapping.open_value)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def main() -> None:
    args = parse_args()

    source_dir = args.source.resolve()
    trainable_dir = source_dir / "trainable"
    if not trainable_dir.is_dir():
        raise ValueError(f"Missing trainable directory: {trainable_dir}")

    gripper_mapping = make_gripper_mapping(
        args.gripper_normalization,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
    )
    output_dir = args.output_dir.resolve()
    info = load_info(trainable_dir)
    episode_rows = load_episode_rows(trainable_dir)

    if args.max_episodes > 0:
        episode_rows = episode_rows[: args.max_episodes]

    for offset, row in enumerate(episode_rows):
        output_index = args.episode_start + offset
        output_path = output_dir / f"episode_{output_index}.hdf5"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists, use --overwrite to replace it")

        print(f"[{offset + 1}/{len(episode_rows)}] episode_index={row['episode_index']} -> {output_path.name}")
        convert_episode(
            trainable_dir=trainable_dir,
            info=info,
            row=row,
            output_path=output_path,
            camera_names=args.camera_names,
            jpeg_quality=args.jpeg_quality,
            gripper_mapping=gripper_mapping,
        )


if __name__ == "__main__":
    main()
