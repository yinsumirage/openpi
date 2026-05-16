import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.ones((14,), dtype=np.float32),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "place the red block on the blue block",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Inputs for a bi-manual ARX-5 policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width] or [height, width, channel].
    - state: [14], ordered left arm 7 DoF then right arm 7 DoF.
    - actions: [action_horizon, 14], using the same ordering.
    """

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")
        if not in_images:
            raise ValueError("At least one ARX camera image is required.")

        images_dict = {name: _parse_image(img) for name, img in in_images.items()}
        template = next(iter(images_dict.values()))

        images = {
            "base_0_rgb": images_dict.get("cam_high", np.zeros_like(template)),
            "left_wrist_0_rgb": images_dict.get("cam_left_wrist", np.zeros_like(template)),
            "right_wrist_0_rgb": images_dict.get("cam_right_wrist", np.zeros_like(template)),
        }
        image_masks = {
            "base_0_rgb": np.True_ if "cam_high" in images_dict else np.False_,
            "left_wrist_0_rgb": np.True_ if "cam_left_wrist" in images_dict else np.False_,
            "right_wrist_0_rgb": np.True_ if "cam_right_wrist" in images_dict else np.False_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"]),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for a bi-manual ARX-5 policy."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}


def _parse_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(255 * img, 0, 255).astype(np.uint8)

    if img.ndim != 3:
        raise ValueError(f"Expected image to have 3 dimensions, got shape {img.shape}")
    if img.shape[0] in (1, 3, 4):
        return einops.rearrange(img, "c h w -> h w c")
    if img.shape[-1] in (1, 3, 4):
        return img
    raise ValueError(f"Expected image to be channel-first or channel-last, got shape {img.shape}")
