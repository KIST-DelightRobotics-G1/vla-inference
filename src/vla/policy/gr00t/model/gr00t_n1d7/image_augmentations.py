# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eval-time image pipeline for the N1.7 recipe (trimmed to inference).

Upstream also carries the training transforms (FractionalRandomCrop,
ColorJitter/rotation, mask-based domain randomization, a ReplayCompose replay
mechanism for cross-view consistency, and a torchvision variant) — all dead
at inference and removed here; restore from Isaac-GR00T@5ac4e6b if needed.
What remains is exactly what a deterministic eval pass executes:

    [LetterBoxPad] -> SmallestMaxSize -> FractionalCenterCrop -> SmallestMaxSize
"""

import albumentations as A
import cv2
import numpy as np
import torch


def apply_images(transform, images):
    """Apply a deterministic albumentations Compose to a list of PIL images.

    (Upstream: ``apply_with_replay`` — the replay/mask machinery served the
    training transforms and was trimmed; a plain Compose has no replay.)

    Returns:
        List of transformed torch tensors (C, H, W) as uint8.
    """
    transformed_tensors = []
    for img in images:
        img_array = transform(image=np.array(img))["image"]
        # Convert to uint8 if needed (albumentations may return float32 in [0,1])
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")
        transformed_tensors.append(torch.from_numpy(img_array).permute(2, 0, 1))
    return transformed_tensors


class FractionalCenterCrop(A.DualTransform):
    """Crop the center part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        crop_fraction: float = 0.9,
        p: float = 1.0,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        # Center the crop
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class LetterBoxPad(A.DualTransform):
    """Pad non-square images to square by adding black bars (letterboxing).

    Ensures all images have the same spatial dimensions after padding,
    regardless of their original aspect ratio.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # Padding is derived from the input image itself rather than from saved
        # params so the transform stays correct when views differ in size.
        h, w = img.shape[:2]
        if h == w:
            return img
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


def build_eval_image_transform(
    image_target_size,
    image_crop_size,
    shortest_image_edge,
    crop_fraction,
    letter_box_transform: bool = False,
) -> A.Compose:
    """The deterministic eval transform of the N1.7 fine-tuning recipe.

    (Upstream: ``build_image_transformations_albumentations`` returned a
    (train, eval) pair; the training half was trimmed. Same eval semantics,
    same fraction/size fallbacks.)

    Args:
        image_target_size: Target size fallback when shortest_image_edge is unset
            (list of [height, width])
        image_crop_size: Crop size fallback when crop_fraction is unset (list of [height, width])
        shortest_image_edge: Shortest edge size for resizing
        crop_fraction: Fraction of image to crop
        letter_box_transform: When True, prepend a LetterBoxPad so mixed-aspect views are padded
            to square before resizing, keeping per-sample views torch.stack-able (cf. #541).
    """
    if crop_fraction is None:
        if image_crop_size is None or image_target_size is None:
            raise ValueError(
                "image_crop_size and image_target_size are required when crop_fraction is None"
            )
        fraction_to_use = image_crop_size[0] / image_target_size[0]
    else:
        fraction_to_use = crop_fraction

    if shortest_image_edge is None:
        if image_target_size is None:
            raise ValueError("image_target_size is required when shortest_image_edge is None")
        max_size = image_target_size[0]
    else:
        max_size = shortest_image_edge

    eval_transform_list = []
    if letter_box_transform:
        eval_transform_list.append(LetterBoxPad())
    eval_transform_list.extend(
        [
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )
    return A.Compose(eval_transform_list)
