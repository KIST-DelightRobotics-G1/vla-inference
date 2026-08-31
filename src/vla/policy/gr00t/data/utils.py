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

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from vla.policy.gr00t.configs.data.embodiment_configs import ModalityConfig


def nested_dict_to_numpy(data):
    """
    Recursively converts bottom-level list of lists to NumPy arrays.

    Args:
        data: A nested dictionary where bottom nodes are list of lists,
              and parent nodes are strings (keys)

    Returns:
        The same dictionary structure with bottom-level lists converted to NumPy arrays

    Example:
        >>> data = {"a": {"b": [[0, 1], [2, 3]]}}
        >>> result = nested_dict_to_numpy(data)
        >>> print(result["a"]["b"])
        [[0 1]
         [2 3]]
    """
    if isinstance(data, dict):
        return {key: nested_dict_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        # Convert lists to numpy arrays
        # NumPy will handle both 1D and 2D cases appropriately
        return np.array(data)
    else:
        return data


def normalize_values_minmax(values, params):
    """
    Normalize values using min-max normalization to [-1, 1] range.

    Args:
        values: Input values to normalize
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Can handle 2D or 3D arrays where last axis represents features
        params: Dictionary with "min" and "max" keys
            - params["min"]: Minimum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Maximum values for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
        joint_group: Optional indexing for joint groups (legacy parameter)

    Returns:
        Normalized values in [-1, 1] range
            - Same shape as input values: (T, D) or (B, T, D)
            - Values are linearly mapped from [min, max] to [-1, 1]
            - For features where min == max, normalized value is 0

    Examples:
        # 1D bounds - same normalization for all steps
        values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step normalization
        values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)
    """
    min_vals = params["min"]
    max_vals = params["max"]
    normalized = np.zeros_like(values)

    mask = ~np.isclose(max_vals, min_vals)

    normalized[..., mask] = (values[..., mask] - min_vals[..., mask]) / (
        max_vals[..., mask] - min_vals[..., mask]
    )
    normalized[..., mask] = 2 * normalized[..., mask] - 1

    return normalized


def unnormalize_values_minmax(normalized_values, params):
    """
    Min-max unnormalization from [-1, 1] range back to original range.

    Args:
        normalized_values: Normalized input values in [-1, 1] range
            - Shape: (T, D) or (B, T, D) where B is batch, T is time/step, D is feature dimension
            - Values outside [-1, 1] are automatically clipped
        params: Dictionary with "min" and "max" keys
            - params["min"]: Original minimum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step
            - params["max"]: Original maximum values used for normalization
                * Case 1 - 1D bounds: Shape (D,) - same min/max for all steps
                * Case 2 - 2D bounds: Shape (T, D) - different min/max per step

    Returns:
        Unnormalized values in original range [min, max]
            - Same shape as input normalized_values: (T, D) or (B, T, D)
            - Values are linearly mapped from [-1, 1] back to [min, max]
            - Input values are clipped to [-1, 1] before unnormalization

    Examples:
        # 1D bounds - same unnormalization for all steps
        normalized_values: (10, 5), params["min"]: (5,), params["max"]: (5,)

        # 2D bounds - per-step unnormalization
        normalized_values: (8, 4), params["min"]: (8, 4), params["max"]: (8, 4)
    """

    min_vals = params["min"]
    max_vals = params["max"]
    range_vals = max_vals - min_vals

    # Unnormalize from [-1, 1]
    unnormalized = (np.clip(normalized_values, -1.0, 1.0) + 1.0) / 2.0 * range_vals + min_vals
    return unnormalized


def parse_modality_configs(
    modality_configs: dict[str, dict[str, ModalityConfig]],
) -> dict[str, dict[str, ModalityConfig]]:
    parsed_modality_configs = {}
    for embodiment_tag, modality_config in modality_configs.items():
        parsed_modality_configs[embodiment_tag] = {}
        for modality, config in modality_config.items():
            if isinstance(config, dict):
                parsed_modality_configs[embodiment_tag][modality] = ModalityConfig(**config)
            else:
                parsed_modality_configs[embodiment_tag][modality] = config
    return parsed_modality_configs
