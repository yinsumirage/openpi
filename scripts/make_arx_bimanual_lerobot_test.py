import json

import numpy as np
import polars as pl

from scripts import make_arx_bimanual_lerobot


def test_convert_dataset_rewrites_state_action_features_and_preserves_assets(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "converted"
    data_dir = raw_dir / "data" / "chunk-000"
    meta_dir = raw_dir / "meta"
    videos_dir = raw_dir / "videos"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()
    videos_dir.mkdir()

    pl.DataFrame(
        {
            "observation.master_left_state": [
                [float(i) for i in range(27)],
                [float(i) for i in range(100, 127)],
            ],
            "observation.master_right_state": [
                [float(i) for i in range(50, 77)],
                [float(i) for i in range(150, 177)],
            ],
            "action.joint_actions": [
                [float(i) for i in range(14)],
                [float(i) for i in range(200, 214)],
            ],
            "episode_index": [0, 0],
            "task_index": [0, 0],
            "observation.images.camera_h": ["video/frame-0", "video/frame-1"],
        }
    ).write_parquet(data_dir / "file-000.parquet")
    (videos_dir / "placeholder.txt").write_text("kept", encoding="utf-8")
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "features": {
                    "observation.master_left_state": {"dtype": "float32", "shape": [27]},
                    "observation.master_right_state": {"dtype": "float32", "shape": [27]},
                    "action.joint_actions": {"dtype": "float32", "shape": [14]},
                },
            }
        ),
        encoding="utf-8",
    )

    make_arx_bimanual_lerobot.convert_dataset(raw_dir, output_dir)

    converted = pl.read_parquet(output_dir / "data" / "chunk-000" / "file-000.parquet")
    assert converted["observation.state"].to_list() == [
        [*map(float, range(7)), *map(float, range(50, 57))],
        [*map(float, range(100, 107)), *map(float, range(150, 157))],
    ]
    assert converted["action"].to_list() == [
        [*map(float, range(14))],
        [*map(float, range(200, 214))],
    ]
    assert converted["observation.images.camera_h"].to_list() == ["video/frame-0", "video/frame-1"]
    assert (output_dir / "videos" / "placeholder.txt").read_text(encoding="utf-8") == "kept"

    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"] == {
        "dtype": "float32",
        "shape": [14],
        "names": [
            [
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
        ],
    }
    assert info["features"]["action"]["shape"] == [14]
    assert (output_dir / "meta" / "tasks.jsonl").read_text(encoding="utf-8") == (
        '{"task_index":0,"task":"place the red block on the blue block"}\n'
    )
    assert (output_dir / "meta" / "episodes.jsonl").read_text(encoding="utf-8") == (
        '{"episode_index":0,"tasks":["place the red block on the blue block"],"length":2}\n'
    )
    episodes_stats = (output_dir / "meta" / "episodes_stats.jsonl").read_text(encoding="utf-8")
    assert '"episode_index":0' in episodes_stats
    assert '"observation.state"' in episodes_stats
    assert '"action"' in episodes_stats
    assert '"count":[2]' in episodes_stats


def test_convert_dataset_can_map_physical_gripper_values_to_openpi_range(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "converted"
    data_dir = raw_dir / "data" / "chunk-000"
    meta_dir = raw_dir / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()

    left_state = [0.0] * 27
    right_state = [0.0] * 27
    action = [0.0] * 14
    left_state[6] = -2.8
    right_state[6] = -0.45
    action[6] = -3.4
    action[13] = 0.1

    pl.DataFrame(
        {
            "observation.master_left_state": [left_state],
            "observation.master_right_state": [right_state],
            "action.joint_actions": [action],
        }
    ).write_parquet(data_dir / "file-000.parquet")
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": {},
            }
        ),
        encoding="utf-8",
    )

    make_arx_bimanual_lerobot.convert_dataset(
        raw_dir,
        output_dir,
        gripper_normalization="physical",
        gripper_open_value=-2.8,
        gripper_close_value=0.0,
    )

    converted = pl.read_parquet(output_dir / "data" / "chunk-000" / "file-000.parquet")
    state = converted["observation.state"][0].to_list()
    converted_action = converted["action"][0].to_list()

    assert state[6] == 0.0
    assert np.isclose(state[13], (-0.45 - -2.8) / (0.0 - -2.8))
    assert converted_action[6] == 0.0
    assert converted_action[13] == 1.0
