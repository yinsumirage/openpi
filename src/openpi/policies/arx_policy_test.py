import numpy as np

from openpi.policies import arx_policy


def test_arx_inputs_maps_bimanual_state_actions_and_available_cameras():
    high = np.ones((3, 4, 5), dtype=np.float32) * 0.5
    right_wrist = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    state = np.arange(14, dtype=np.float32)
    actions = np.arange(2 * 14, dtype=np.float32).reshape(2, 14)

    transformed = arx_policy.ArxInputs()(
        {
            "images": {
                "cam_high": high,
                "cam_right_wrist": right_wrist,
            },
            "state": state,
            "actions": actions,
            "prompt": "place the red block on the blue block",
        }
    )

    assert set(transformed["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert transformed["image"]["base_0_rgb"].shape == (4, 5, 3)
    assert transformed["image"]["base_0_rgb"].dtype == np.uint8
    assert transformed["image"]["right_wrist_0_rgb"].shape == (4, 5, 3)
    assert transformed["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.False_,
        "right_wrist_0_rgb": np.True_,
    }
    np.testing.assert_array_equal(transformed["state"], state)
    np.testing.assert_array_equal(transformed["actions"], actions)
    assert transformed["prompt"] == "place the red block on the blue block"


def test_arx_outputs_returns_physical_14d_action_space_without_aloha_adaptation():
    actions = np.arange(2 * 32, dtype=np.float32).reshape(2, 32)

    transformed = arx_policy.ArxOutputs()({"actions": actions})

    np.testing.assert_array_equal(transformed["actions"], actions[:, :14])
