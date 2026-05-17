"""Convert ACT-style ARX HDF5 episodes into a local LeRobot dataset for openpi.

The input is a directory containing `episode_*.hdf5` files with:

- `/observations/qpos`: 14D bimanual joint state.
- `/action`: 14D bimanual joint action.
- `/observations/images/head`: JPEG-padded head camera frames.
- `/observations/images/right_wrist`: JPEG-padded right wrist camera frames.

The output uses the v2.1-style per-episode parquet/video layout expected by the
pinned LeRobot loader used by this openpi checkout.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import polars as pl


LEGACY_CHUNKS_SIZE = 1000
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
DEFAULT_TASK = "place the red block on the blue block"
USED_CAMERAS = {
    "head": "observation.images.camera_h",
    "right_wrist": "observation.images.camera_r",
}
ARX_BIMANUAL_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_gripper",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_gripper",
]


@dataclasses.dataclass(frozen=True)
class GripperMapping:
    open_value: float
    close_value: float

    def __post_init__(self) -> None:
        if self.open_value == self.close_value:
            raise ValueError("gripper_open_value and gripper_close_value must be different.")


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    source_path: Path
    episode_index: int
    length: int
    task: str


def convert_hdf5_dataset(
    hdf5_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    fps: int = 30,
    task: str | None = None,
    gripper_normalization: str = "physical",
    gripper_open_value: float = -2.8,
    gripper_close_value: float = 0.0,
    image_jpeg_quality: int = 95,
) -> None:
    hdf5_dir = hdf5_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    mapping = _make_gripper_mapping(
        gripper_normalization,
        gripper_open_value=gripper_open_value,
        gripper_close_value=gripper_close_value,
    )

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"Missing HDF5 directory: {hdf5_dir}")
    hdf5_paths = _find_hdf5_episodes(hdf5_dir)
    if not hdf5_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {hdf5_dir}")

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    (output_dir / "data").mkdir(parents=True)
    (output_dir / "videos").mkdir(parents=True)
    (output_dir / "meta").mkdir(parents=True)

    records: list[EpisodeRecord] = []
    video_shapes: dict[str, tuple[int, int, int]] = {}
    global_index = 0
    for episode_index, hdf5_path in enumerate(hdf5_paths):
        print(f"[{episode_index + 1}/{len(hdf5_paths)}] {hdf5_path.name} -> episode_{episode_index:06d}")
        with h5py.File(hdf5_path, "r") as root:
            qpos = _read_required_array(root, "/observations/qpos", expected_width=14)
            action = _read_required_array(root, "/action", expected_width=14)
            if len(qpos) != len(action):
                raise ValueError(f"{hdf5_path}: qpos/action length mismatch: {len(qpos)} vs {len(action)}")

            state = _normalize_state_or_action(qpos[:, :14], mapping)
            normalized_action = _normalize_state_or_action(action[:, :14], mapping)
            frame_count = state.shape[0]
            episode_task = task or _read_task(root)

            data_path = output_dir / LEGACY_DATA_PATH.format(
                episode_chunk=episode_index // LEGACY_CHUNKS_SIZE,
                episode_index=episode_index,
            )
            data_path.parent.mkdir(parents=True, exist_ok=True)
            _write_episode_parquet(
                data_path,
                state=state,
                action=normalized_action,
                episode_index=episode_index,
                global_start_index=global_index,
                fps=fps,
            )

            image_group = _require_group(root, "/observations/images")
            for hdf5_camera, video_key in USED_CAMERAS.items():
                if hdf5_camera not in image_group:
                    raise KeyError(f"{hdf5_path}: missing /observations/images/{hdf5_camera}")
                video_path = output_dir / LEGACY_VIDEO_PATH.format(
                    episode_chunk=episode_index // LEGACY_CHUNKS_SIZE,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                shape = _write_camera_video(
                    image_group[hdf5_camera],
                    video_path,
                    fps=fps,
                    expected_frames=frame_count,
                    image_jpeg_quality=image_jpeg_quality,
                )
                video_shapes.setdefault(video_key, shape)

            records.append(
                EpisodeRecord(
                    source_path=hdf5_path,
                    episode_index=episode_index,
                    length=frame_count,
                    task=episode_task,
                )
            )
            global_index += frame_count

    _write_info(output_dir / "meta" / "info.json", records, video_shapes, fps=fps)
    _write_tasks_jsonl(output_dir / "meta" / "tasks.jsonl", records)
    _write_episodes_jsonl(output_dir / "meta" / "episodes.jsonl", records)
    _write_episodes_stats_jsonl(output_dir / "meta" / "episodes_stats.jsonl", output_dir, records)


def decode_hdf5_image(encoded_or_frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(encoded_or_frame)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        return frame.astype(np.uint8, copy=False)

    encoded = _trim_encoded_image(frame)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode HDF5 JPEG image")
    return decoded


def _find_hdf5_episodes(hdf5_dir: Path) -> list[Path]:
    paths = list(hdf5_dir.glob("episode_*.hdf5")) + list(hdf5_dir.glob("episode_*.h5"))
    return sorted(paths, key=_episode_sort_key)


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**12, path.name


def _make_gripper_mapping(
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


def _normalize_state_or_action(values: np.ndarray, mapping: GripperMapping | None) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32).copy()
    if output.ndim != 2 or output.shape[1] != 14:
        raise ValueError(f"Expected shape [T, 14], got {output.shape}")
    if mapping is not None:
        output[:, 6] = _map_gripper_values(output[:, 6], mapping)
        output[:, 13] = _map_gripper_values(output[:, 13], mapping)
    return output


def _map_gripper_values(values: np.ndarray, mapping: GripperMapping) -> np.ndarray:
    normalized = (np.asarray(values, dtype=np.float32) - mapping.open_value) / (
        mapping.close_value - mapping.open_value
    )
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    episode_index: int,
    global_start_index: int,
    fps: int,
) -> None:
    frame_count = len(state)
    frame_index = np.arange(frame_count, dtype=np.int64)
    table = pl.DataFrame(
        {
            "timestamp": (frame_index.astype(np.float32) / np.float32(fps)).tolist(),
            "frame_index": frame_index.tolist(),
            "episode_index": [episode_index] * frame_count,
            "index": (global_start_index + frame_index).tolist(),
            "task_index": [0] * frame_count,
        }
    ).with_columns(
        pl.Series("observation.state", state.tolist(), dtype=pl.List(pl.Float32)),
        pl.Series("action", action.tolist(), dtype=pl.List(pl.Float32)),
    )
    table.write_parquet(path)


def _write_camera_video(
    image_dataset: h5py.Dataset,
    video_path: Path,
    *,
    fps: int,
    expected_frames: int,
    image_jpeg_quality: int,
) -> tuple[int, int, int]:
    if len(image_dataset) != expected_frames:
        raise ValueError(f"Image/action length mismatch: {len(image_dataset)} vs {expected_frames}")

    first_frame = decode_hdf5_image(image_dataset[0])
    height, width = first_frame.shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Failed to open video writer: {video_path}")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), image_jpeg_quality]
    try:
        for index in range(expected_frames):
            frame = first_frame if index == 0 else decode_hdf5_image(image_dataset[index])
            if frame.shape[:2] != (height, width):
                raise ValueError(f"Frame shape changed in {video_path}: {frame.shape[:2]} vs {(height, width)}")
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            # Re-encode/decode once with a fixed JPEG quality if the source frame is uncompressed.
            # Padded-JPEG HDF5 inputs are already decoded at this point; the video writer expects BGR frames.
            if image_dataset[index].ndim == 3:
                ok, buffer = cv2.imencode(".jpg", frame, encode_params)
                if not ok:
                    raise ValueError("Failed to normalize raw image frame through JPEG")
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            writer.write(frame)
    finally:
        writer.release()

    return height, width, 3


def _trim_encoded_image(frame: np.ndarray) -> np.ndarray:
    encoded = np.asarray(frame, dtype=np.uint8).reshape(-1)
    if encoded.size == 0:
        raise ValueError("Cannot decode empty image payload")

    eoi_positions = np.where((encoded[:-1] == 0xFF) & (encoded[1:] == 0xD9))[0]
    if len(eoi_positions):
        return encoded[: int(eoi_positions[-1]) + 2]

    nonzero = np.flatnonzero(encoded)
    if len(nonzero) == 0:
        raise ValueError("Cannot decode all-zero image payload")
    return encoded[: int(nonzero[-1]) + 1]


def _write_info(info_path: Path, records: list[EpisodeRecord], video_shapes: dict[str, tuple[int, int, int]], *, fps: int) -> None:
    total_episodes = len(records)
    total_frames = sum(record.length for record in records)
    features: dict[str, Any] = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "observation.state": _float_feature(),
        "action": _float_feature(),
    }
    for video_key, shape in video_shapes.items():
        height, width, channels = shape
        features[video_key] = _video_feature(height=height, width=width, channels=channels, fps=fps)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "arx5",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(video_shapes),
        "total_chunks": _ceil_div(total_episodes, LEGACY_CHUNKS_SIZE),
        "chunks_size": LEGACY_CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": LEGACY_DATA_PATH,
        "video_path": LEGACY_VIDEO_PATH,
        "features": features,
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_tasks_jsonl(tasks_path: Path, records: list[EpisodeRecord]) -> None:
    task = records[0].task if records else DEFAULT_TASK
    tasks_path.write_text(json.dumps({"task_index": 0, "task": task}, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_episodes_jsonl(episodes_path: Path, records: list[EpisodeRecord]) -> None:
    lines = [
        json.dumps(
            {"episode_index": record.episode_index, "tasks": [record.task], "length": record.length},
            separators=(",", ":"),
        )
        for record in records
    ]
    episodes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_episodes_stats_jsonl(episodes_stats_path: Path, output_dir: Path, records: list[EpisodeRecord]) -> None:
    lines = []
    for record in records:
        parquet_path = output_dir / LEGACY_DATA_PATH.format(
            episode_chunk=record.episode_index // LEGACY_CHUNKS_SIZE,
            episode_index=record.episode_index,
        )
        df = pl.read_parquet(parquet_path, columns=["observation.state", "action"])
        stats = {
            "observation.state": _compute_stats(np.asarray(df["observation.state"].to_list(), dtype=np.float32)),
            "action": _compute_stats(np.asarray(df["action"].to_list(), dtype=np.float32)),
        }
        lines.append(json.dumps({"episode_index": record.episode_index, "stats": stats}, separators=(",", ":")))
    episodes_stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_required_array(root: h5py.File, key: str, *, expected_width: int) -> np.ndarray:
    if key not in root:
        raise KeyError(f"Missing required dataset: {key}")
    values = np.asarray(root[key], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < expected_width:
        raise ValueError(f"{key} must have shape [T, >= {expected_width}], got {values.shape}")
    return values


def _require_group(root: h5py.File, key: str) -> h5py.Group:
    if key not in root:
        raise KeyError(f"Missing required group: {key}")
    group = root[key]
    if not isinstance(group, h5py.Group):
        raise TypeError(f"{key} is not an HDF5 group")
    return group


def _read_task(root: h5py.File) -> str:
    task = root.attrs.get("task", DEFAULT_TASK)
    if isinstance(task, bytes):
        task = task.decode("utf-8")
    return str(task or DEFAULT_TASK)


def _float_feature() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": [14],
        "names": [ARX_BIMANUAL_NAMES],
    }


def _video_feature(*, height: int, width: int, channels: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "mp4v",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": False,
        },
    }


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", type=Path, required=True, help="Directory containing episode_*.hdf5 files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output local LeRobot dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate to write into metadata and videos.")
    parser.add_argument("--task", type=str, default=None, help="Override task prompt for all episodes.")
    parser.add_argument(
        "--gripper-normalization",
        choices=("none", "physical"),
        default="physical",
        help="Map physical gripper command values into openpi 0=open, 1=closed convention.",
    )
    parser.add_argument("--gripper-open-value", type=float, default=-2.8, help="Physical value for open gripper.")
    parser.add_argument("--gripper-close-value", type=float, default=0.0, help="Physical value for closed gripper.")
    parser.add_argument("--image-jpeg-quality", type=int, default=95, help="JPEG quality for raw image normalization.")
    args = parser.parse_args()

    convert_hdf5_dataset(
        args.hdf5_dir,
        args.output_dir,
        overwrite=args.overwrite,
        fps=args.fps,
        task=args.task,
        gripper_normalization=args.gripper_normalization,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        image_jpeg_quality=args.image_jpeg_quality,
    )


if __name__ == "__main__":
    main()
