"""Convert ARX-5 LeRobot data to the bi-manual action space expected by openpi.

The converted dataset keeps RGB/depth/video assets from the source dataset and adds:

- observation.state: left 7 DoF followed by right 7 DoF
- action: left 7 DoF followed by right 7 DoF

By default, gripper dimensions are expected to already follow the openpi ARX convention:
0.0 = fully open, 1.0 = fully closed. Use --gripper-normalization physical to map physical command values.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import polars as pl


LEGACY_CHUNKS_SIZE = 1000
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
USED_VIDEO_KEYS = ("observation.images.camera_h", "observation.images.camera_r")

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


def convert_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    gripper_normalization: str = "none",
    gripper_open_value: float = -2.8,
    gripper_close_value: float = 0.0,
) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    gripper_mapping = _make_gripper_mapping(
        gripper_normalization,
        gripper_open_value=gripper_open_value,
        gripper_close_value=gripper_close_value,
    )

    if not (raw_dir / "data").is_dir():
        raise FileNotFoundError(f"Missing LeRobot data directory: {raw_dir / 'data'}")
    if not (raw_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Missing LeRobot info.json: {raw_dir / 'meta' / 'info.json'}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    shutil.copytree(raw_dir, output_dir)

    parquet_paths = sorted((output_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {output_dir / 'data'}")

    for parquet_path in parquet_paths:
        _rewrite_parquet(parquet_path, gripper_mapping=gripper_mapping)

    info_path = output_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_records = _load_episode_records(output_dir / "meta" / "episodes", parquet_paths)
    _write_legacy_data_files(output_dir, episode_records, fps=float(info["fps"]))
    _write_legacy_video_files(output_dir, info, episode_records)
    _rewrite_info(info_path, episode_records)
    task = _ensure_tasks_jsonl(output_dir / "meta" / "tasks.jsonl", info_path)
    _ensure_episodes_jsonl(output_dir / "meta" / "episodes.jsonl", episode_records, task)
    _ensure_episodes_stats_jsonl(output_dir / "meta" / "episodes_stats.jsonl", output_dir, episode_records)


def _rewrite_parquet(parquet_path: Path, *, gripper_mapping: GripperMapping | None) -> None:
    df = pl.read_parquet(parquet_path)
    required = {
        "observation.master_left_state",
        "observation.master_right_state",
        "action.joint_actions",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{parquet_path} is missing required columns: {missing}")

    df = df.with_columns(
        pl.struct(["observation.master_left_state", "observation.master_right_state"])
        .map_elements(lambda row: _make_state(row, gripper_mapping), return_dtype=pl.List(pl.Float32))
        .alias("observation.state"),
        pl.col("action.joint_actions")
        .map_elements(
            lambda values: _make_action(values, gripper_mapping),
            return_dtype=pl.List(pl.Float32),
        )
        .alias("action"),
    )
    df.write_parquet(parquet_path)


def _make_state(row: dict[str, Any], gripper_mapping: GripperMapping | None) -> list[float]:
    left_state = _slice(row["observation.master_left_state"], 0, 7)
    right_state = _slice(row["observation.master_right_state"], 0, 7)
    if gripper_mapping is not None:
        left_state[6] = _map_gripper_value(left_state[6], gripper_mapping)
        right_state[6] = _map_gripper_value(right_state[6], gripper_mapping)
    return [*left_state, *right_state]


def _make_action(values: Any, gripper_mapping: GripperMapping | None) -> list[float]:
    action = _slice(values, 0, 14)
    if gripper_mapping is not None:
        action[6] = _map_gripper_value(action[6], gripper_mapping)
        action[13] = _map_gripper_value(action[13], gripper_mapping)
    return action


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


def _map_gripper_value(value: float, mapping: GripperMapping) -> float:
    normalized = (float(value) - mapping.open_value) / (mapping.close_value - mapping.open_value)
    return min(max(normalized, 0.0), 1.0)


def _slice(values: Any, start: int, stop: int) -> list[float]:
    if hasattr(values, "to_list"):
        values = values.to_list()
    return [float(value) for value in list(values)[start:stop]]


def _rewrite_info(info_path: Path, episode_records: list[dict[str, Any]]) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    total_episodes = len(episode_records)
    total_frames = sum(int(record["length"]) for record in episode_records)

    info["codebase_version"] = "v2.1"
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH
    info["chunks_size"] = LEGACY_CHUNKS_SIZE
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = _ceil_div(total_episodes, LEGACY_CHUNKS_SIZE)

    features = info.setdefault("features", {})
    features["observation.state"] = _feature()
    features["action"] = _feature()
    for key in list(features):
        if key.startswith("observation.images.") and key not in USED_VIDEO_KEYS:
            del features[key]
    video_keys = [key for key in USED_VIDEO_KEYS if key in features]
    info["total_videos"] = total_episodes * len(video_keys)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _ensure_tasks_jsonl(tasks_path: Path, info_path: Path) -> str:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    task = _infer_task(info)

    tasks_path.write_text(json.dumps({"task_index": 0, "task": task}, separators=(",", ":")) + "\n", encoding="utf-8")
    return task


def _ensure_episodes_jsonl(episodes_path: Path, episode_records: list[dict[str, Any]], task: str) -> None:
    lines = [
        json.dumps(
            {
                "episode_index": int(record["episode_index"]),
                "tasks": [task],
                "length": int(record["length"]),
            },
            separators=(",", ":"),
        )
        for record in sorted(episode_records, key=lambda item: int(item["episode_index"]))
    ]
    episodes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_episodes_stats_jsonl(
    episodes_stats_path: Path,
    output_dir: Path,
    episode_records: list[dict[str, Any]],
) -> None:
    lines = []
    for record in sorted(episode_records, key=lambda item: int(item["episode_index"])):
        episode_index = int(record["episode_index"])
        parquet_path = output_dir / LEGACY_DATA_PATH.format(
            episode_chunk=episode_index // LEGACY_CHUNKS_SIZE,
            episode_index=episode_index,
        )
        df = pl.read_parquet(parquet_path, columns=["observation.state", "action"])
        stats = {
            "observation.state": _compute_stats(np.asarray(df["observation.state"].to_list(), dtype=np.float32)),
            "action": _compute_stats(np.asarray(df["action"].to_list(), dtype=np.float32)),
        }
        lines.append(
            json.dumps(
                {"episode_index": episode_index, "stats": stats},
                separators=(",", ":"),
            )
        )
    episodes_stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _load_episode_records(episodes_dir: Path, parquet_paths: list[Path]) -> list[dict[str, Any]]:
    if episodes_dir.is_dir():
        records = []
        for path in sorted(episodes_dir.glob("chunk-*/*.parquet")):
            records.extend(pl.read_parquet(path).to_dicts())
        if records:
            return [_normalize_episode_record(record) for record in records]

    records = []
    for parquet_path in parquet_paths:
        columns = pl.read_parquet(parquet_path, n_rows=1).columns
        if "episode_index" in columns:
            df = pl.read_parquet(parquet_path, columns=["episode_index"])
            counts = df.group_by("episode_index").len().sort("episode_index")
        else:
            df = pl.read_parquet(parquet_path)
            counts = pl.DataFrame({"episode_index": [0], "len": [len(df)]})

        start = 0
        for episode_index, length in counts.iter_rows():
            records.append(
                {
                    "episode_index": int(episode_index),
                    "length": int(length),
                    "data/chunk_index": int(parquet_path.parent.name.split("-")[-1]),
                    "data/file_index": int(parquet_path.stem.split("-")[-1]),
                    "dataset_from_index": start,
                    "dataset_to_index": start + int(length),
                }
            )
            start += int(length)
    return records


def _normalize_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "length" not in normalized:
        normalized["length"] = int(normalized["dataset_to_index"]) - int(normalized["dataset_from_index"])
    return normalized


def _write_legacy_data_files(output_dir: Path, episode_records: list[dict[str, Any]], *, fps: float) -> None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in episode_records:
        key = (int(record["data/chunk_index"]), int(record["data/file_index"]))
        grouped.setdefault(key, []).append(record)

    for (chunk_index, file_index), records in grouped.items():
        source_path = output_dir / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        df = pl.read_parquet(source_path)
        for record in records:
            episode_index = int(record["episode_index"])
            if "episode_index" in df.columns:
                episode_df = df.filter(pl.col("episode_index") == episode_index)
            else:
                episode_df = pl.DataFrame()
            if len(episode_df) == 0:
                start = int(record["dataset_from_index"])
                stop = int(record["dataset_to_index"])
                episode_df = df.slice(start, stop - start)
            episode_df = _normalize_episode_timing(episode_df, fps=fps)
            dest_path = output_dir / LEGACY_DATA_PATH.format(
                episode_chunk=episode_index // LEGACY_CHUNKS_SIZE,
                episode_index=episode_index,
            )
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            episode_df.write_parquet(dest_path)

        source_path.unlink()


def _normalize_episode_timing(episode_df: pl.DataFrame, *, fps: float) -> pl.DataFrame:
    frame_indices = pl.int_range(0, pl.len(), dtype=pl.Int64)
    return episode_df.with_columns(
        frame_indices.alias("frame_index"),
        (frame_indices.cast(pl.Float32) / fps).alias("timestamp"),
    )


def _write_legacy_video_files(output_dir: Path, info: dict[str, Any], episode_records: list[dict[str, Any]]) -> None:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return

    fps = float(info["fps"])
    source_usage_counts = _video_source_usage_counts(output_dir, info, episode_records)
    for video_key in USED_VIDEO_KEYS:
        for record in episode_records:
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            from_key = f"videos/{video_key}/from_timestamp"
            if not all(key in record for key in (chunk_key, file_key)):
                continue
            if from_key in record:
                start_frame = round(float(record[from_key]) * fps)
            else:
                start_frame = int(record["dataset_from_index"])

            source_path = output_dir / video_path_template.format(
                video_key=video_key,
                chunk_index=int(record[chunk_key]),
                file_index=int(record[file_key]),
            )
            episode_index = int(record["episode_index"])
            dest_path = output_dir / LEGACY_VIDEO_PATH.format(
                episode_chunk=episode_index // LEGACY_CHUNKS_SIZE,
                video_key=video_key,
                episode_index=episode_index,
            )
            _write_video_segment(
                source_path,
                dest_path,
                start_frame=start_frame,
                frame_count=int(record["length"]),
                fps=fps,
                allow_move=source_usage_counts.get(source_path, 0) == 1,
            )


def _video_source_usage_counts(
    output_dir: Path,
    info: dict[str, Any],
    episode_records: list[dict[str, Any]],
) -> dict[Path, int]:
    video_path_template = info.get("video_path")
    if not video_path_template:
        return {}

    counts: dict[Path, int] = {}
    for video_key in USED_VIDEO_KEYS:
        for record in episode_records:
            chunk_key = f"videos/{video_key}/chunk_index"
            file_key = f"videos/{video_key}/file_index"
            if not all(key in record for key in (chunk_key, file_key)):
                continue
            source_path = output_dir / video_path_template.format(
                video_key=video_key,
                chunk_index=int(record[chunk_key]),
                file_index=int(record[file_key]),
            )
            counts[source_path] = counts.get(source_path, 0) + 1
    return counts


def _write_video_segment(
    source_path: Path,
    dest_path: Path,
    *,
    start_frame: int,
    frame_count: int,
    fps: float,
    allow_move: bool,
) -> None:
    if dest_path.exists():
        return
    if not source_path.is_file():
        raise FileNotFoundError(f"Expected source video not found: {source_path}")

    if allow_move and _move_whole_video_if_possible(source_path, dest_path, start_frame, frame_count):
        return

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {source_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dest_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Failed to open video writer: {dest_path}")

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Failed to read frame from {source_path}")
            writer.write(frame)
    finally:
        writer.release()
        capture.release()


def _move_whole_video_if_possible(source_path: Path, dest_path: Path, start_frame: int, frame_count: int) -> bool:
    if start_frame != 0:
        return False

    source_frame_count = _video_frame_count(source_path)
    if source_frame_count is None or source_frame_count != frame_count:
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(dest_path))
    return True


def _video_frame_count(video_path: Path) -> int | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return frame_count if frame_count > 0 else None


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _infer_task(info: dict[str, Any]) -> str:
    tasks = info.get("tasks")
    if isinstance(tasks, dict) and tasks:
        first_key = sorted(tasks, key=lambda key: int(key))[0]
        return str(tasks[first_key])
    if isinstance(tasks, list) and tasks:
        first_task = tasks[0]
        if isinstance(first_task, dict):
            return str(first_task.get("task", "place the red block on the blue block"))
        return str(first_task)
    return "place the red block on the blue block"


def _feature() -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": [14],
        "names": [ARX_BIMANUAL_NAMES],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Original LeRobot dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Converted LeRobot dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it already exists.")
    parser.add_argument(
        "--gripper-normalization",
        choices=("none", "physical"),
        default="none",
        help="Map physical gripper command values into openpi 0=open, 1=closed convention.",
    )
    parser.add_argument("--gripper-open-value", type=float, default=-2.8, help="Physical value for fully open gripper.")
    parser.add_argument("--gripper-close-value", type=float, default=0.0, help="Physical value for closed gripper.")
    args = parser.parse_args()

    convert_dataset(
        args.raw_dir,
        args.output_dir,
        overwrite=args.overwrite,
        gripper_normalization=args.gripper_normalization,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
    )


if __name__ == "__main__":
    main()
