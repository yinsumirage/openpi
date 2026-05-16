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

import polars as pl


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

    _rewrite_info(output_dir / "meta" / "info.json")


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


def _rewrite_info(info_path: Path) -> None:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    features["observation.state"] = _feature()
    features["action"] = _feature()
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
