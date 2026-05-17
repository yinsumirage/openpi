import json

import cv2
import h5py
import numpy as np
import polars as pl

from scripts import make_arx_lerobot_from_hdf5


def _encode_jpeg(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    assert ok
    return buffer


def _write_padded_jpegs(group, name, frames):
    encoded = [_encode_jpeg(frame) for frame in frames]
    width = max(len(item) for item in encoded) + 8
    payload = np.zeros((len(encoded), width), dtype=np.uint8)
    for index, item in enumerate(encoded):
        payload[index, : len(item)] = item
    group.create_dataset(name, data=payload)


def test_convert_hdf5_dataset_writes_loader_compatible_lerobot_dataset(tmp_path):
    hdf5_dir = tmp_path / "hdf5"
    output_dir = tmp_path / "lerobot"
    hdf5_dir.mkdir()

    qpos = np.zeros((2, 14), dtype=np.float32)
    action = np.zeros((2, 14), dtype=np.float32)
    qpos[:, 6] = [-2.8, 0.0]
    qpos[:, 13] = [-1.4, 0.0]
    action[:, 6] = [-2.8, -1.4]
    action[:, 13] = [0.0, -2.8]
    qvel = np.ones((2, 14), dtype=np.float32)
    effort = np.ones((2, 14), dtype=np.float32) * 2.0
    eef = np.ones((2, 14), dtype=np.float32) * 3.0
    head_frames = [
        np.full((6, 8, 3), (0, 0, 255), dtype=np.uint8),
        np.full((6, 8, 3), (0, 255, 0), dtype=np.uint8),
    ]
    wrist_frames = [
        np.full((6, 8, 3), (255, 0, 0), dtype=np.uint8),
        np.full((6, 8, 3), (255, 255, 0), dtype=np.uint8),
    ]

    with h5py.File(hdf5_dir / "episode_3.hdf5", "w") as root:
        root.attrs["sim"] = False
        root.attrs["task"] = "place the red block on the blue block"
        root.create_dataset("action", data=action)
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos)
        obs.create_dataset("qvel", data=qvel)
        obs.create_dataset("effort", data=effort)
        obs.create_dataset("eef", data=eef)
        image_group = obs.create_group("images")
        _write_padded_jpegs(image_group, "head", head_frames)
        _write_padded_jpegs(image_group, "right_wrist", wrist_frames)

    make_arx_lerobot_from_hdf5.convert_hdf5_dataset(
        hdf5_dir,
        output_dir,
        overwrite=True,
        fps=30,
        gripper_normalization="physical",
        gripper_open_value=-2.8,
        gripper_close_value=0.0,
    )

    parquet_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"
    converted = pl.read_parquet(parquet_path)
    assert converted["episode_index"].to_list() == [0, 0]
    assert converted["frame_index"].to_list() == [0, 1]
    assert converted["timestamp"].to_list() == [0.0, np.float32(1 / 30).item()]
    assert converted["index"].to_list() == [0, 1]
    assert converted["task_index"].to_list() == [0, 0]

    state = converted["observation.state"].to_list()
    converted_action = converted["action"].to_list()
    assert state[0][6] == 0.0
    assert state[0][13] == 0.5
    assert state[1][6] == 1.0
    assert state[1][13] == 1.0
    assert converted_action[0][6] == 0.0
    assert converted_action[0][13] == 1.0
    assert converted_action[1][6] == 0.5
    assert converted_action[1][13] == 0.0

    info = json.loads((output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    assert set(info["features"]) >= {
        "observation.state",
        "action",
        "observation.images.camera_h",
        "observation.images.camera_r",
    }
    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["action"]["shape"] == [14]
    assert info["features"]["observation.images.camera_h"]["shape"] == [6, 8, 3]

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

    for video_key in ("observation.images.camera_h", "observation.images.camera_r"):
        video_path = output_dir / "videos" / "chunk-000" / video_key / "episode_000000.mp4"
        assert video_path.is_file()
        capture = cv2.VideoCapture(str(video_path))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        finally:
            capture.release()


def test_decode_padded_jpeg_ignores_zero_padding():
    frame = np.full((5, 7, 3), (10, 20, 30), dtype=np.uint8)
    encoded = _encode_jpeg(frame)
    padded = np.zeros((len(encoded) + 32,), dtype=np.uint8)
    padded[: len(encoded)] = encoded

    decoded = make_arx_lerobot_from_hdf5.decode_hdf5_image(padded)

    assert decoded.shape == frame.shape
    assert decoded.dtype == np.uint8
