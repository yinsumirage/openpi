import dataclasses
import pathlib

from openpi.models import pi0_config
from openpi.policies import arx_policy
from openpi.training import config as _config


def test_lerobot_arx_data_config_uses_arx_transforms_and_delta_joint_actions(tmp_path: pathlib.Path):
    factory = _config.LeRobotArxDataConfig(
        repo_id="local/test_arx",
        assets=_config.AssetsConfig(asset_id="arx"),
    )

    data_config = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True))

    assert data_config.repo_id == "local/test_arx"
    assert data_config.asset_id == "arx"
    assert data_config.action_sequence_keys == ("action",)
    assert isinstance(data_config.data_transforms.inputs[0], arx_policy.ArxInputs)
    assert isinstance(data_config.data_transforms.outputs[-1], arx_policy.ArxOutputs)
    assert data_config.use_quantile_norm is True


def test_pi05_arx_debug_config_is_registered_for_local_bimanual_dataset():
    train_config = _config.get_config("pi05_arx_debug")

    assert train_config.model.pi05 is True
    assert train_config.model.action_horizon == 50
    assert train_config.data.repo_id == "local/arx_block_stack_bimanual"
    assert train_config.data.assets.assets_dir == "gs://openpi-assets/checkpoints/pi05_base/assets"
    assert train_config.data.assets.asset_id == "arx"
    assert train_config.data.default_prompt == "place the red block on the blue block"
    assert train_config.batch_size == 8
    assert train_config.num_train_steps == 5_000
    assert train_config.save_interval == 500
    assert train_config.wandb_enabled is False

    # The debug config should be cheap to customize from the CLI/dataclass overrides.
    updated = dataclasses.replace(train_config, batch_size=4)
    assert updated.batch_size == 4


def test_pi05_arx_debug_fresh_stats_config_is_registered_for_dataset_norm_stats():
    train_config = _config.get_config("pi05_arx_debug_fresh_stats")

    assert train_config.model.pi05 is True
    assert train_config.data.repo_id == "local/arx_block_stack_bimanual"
    assert train_config.data.assets.assets_dir is None
    assert train_config.data.assets.asset_id is None
    assert train_config.data.default_prompt == "place the red block on the blue block"
    assert train_config.batch_size == 8
    assert train_config.num_train_steps == 5_000
