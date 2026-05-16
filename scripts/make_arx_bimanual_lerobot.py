"""Convert ARX-5 LeRobot data to the bi-manual action space expected by openpi.

The converted dataset keeps RGB/depth/video assets from the source dataset and adds:

- observation.state: left 7 DoF followed by right 7 DoF
- action: left 7 DoF followed by right 7 DoF

The gripper dimensions are expected to already follow the openpi ARX convention:
0.0 = fully open, 1.0 = fully closed.
"""

from __future__ import annotations

import argparse
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


def convert_dataset(raw_dir: Path, output_dir: Path, *, overwrite: bool = False) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

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
        _rewrite_parquet(parquet_path)

    _rewrite_info(output_dir / "meta" / "info.json")


def _rewrite_parquet(parquet_path: Path) -> None:
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
        .map_elements(_make_state, return_dtype=pl.List(pl.Float32))
        .alias("observation.state"),
        pl.col("action.joint_actions")
        .map_elements(lambda values: _slice(values, 0, 14), return_dtype=pl.List(pl.Float32))
        .alias("action"),
    )
    df.write_parquet(parquet_path)


def _make_state(row: dict[str, Any]) -> list[float]:
    return [
        *_slice(row["observation.master_left_state"], 0, 7),
        *_slice(row["observation.master_right_state"], 0, 7),
    ]


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
    args = parser.parse_args()

    convert_dataset(args.raw_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
