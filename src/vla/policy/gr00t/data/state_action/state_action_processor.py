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

"""
Processor for robot state and action data (trimmed to the inference path).

Handles:
- State normalization (min/max to [-1, 1])
- Action denormalization (min/max, for decoding model output)

Upstream also handles mean/std and sin/cos state encodings, training-side
action normalization, and ABSOLUTE<->RELATIVE conversion — none of which a
SONIC checkpoint (minmax q01/q99, all-ABSOLUTE) reaches at inference. Those
paths fail loud here; restore from Isaac-GR00T@5ac4e6b if ever needed.
"""

from copy import deepcopy
import logging

from vla.policy.gr00t.configs.data.embodiment_configs import (
    ActionRepresentation,
    ModalityConfig,
)
from vla.policy.gr00t.data.utils import (
    nested_dict_to_numpy,
    normalize_values_minmax,
    parse_modality_configs,
    unnormalize_values_minmax,
)
import numpy as np


logger = logging.getLogger(__name__)

_TRIM_NOTE = (
    "was trimmed from this extraction — SONIC checkpoints use minmax q01/q99 "
    "normalization and ABSOLUTE actions only. Restore "
    "gr00t/data/state_action/state_action_processor.py from "
    "Isaac-GR00T@5ac4e6b to run a checkpoint that needs it."
)


class StateActionProcessor:
    """
    Processor for robot state and action data (inference path only).

    Handles:
    - State normalization (min/max to [-1, 1])
    - Action denormalization (min/max)
    """

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        apply_sincos_state_encoding: bool = False,
        use_relative_action: bool = False,
    ):
        """
        Initialize unified state and action processor.

        Args:
            modality_configs: Nested dict with structure:
                {embodiment_tag: {modality: ModalityConfig}}
                where modality in ["state", "action"]
                Example: {"gr1": {"state": ModalityConfig(...), "action": ModalityConfig(...)}}
            statistics: Optional nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
                where modality in ["state", "action", "relative_action"]
                and stat_type in ["min", "max", "mean", "std", "q01", "q99"]
                Example: {"gr1": {"state": {"left_arm": {"min": [...], "max": [...], ...}}}}
            use_percentiles: Whether to use percentiles (q01/q99) instead of min/max
            clip_outliers: Whether to clip normalized values to [-1, 1]
            apply_sincos_state_encoding: Global flag to enable sin/cos encoding for states
        """
        self.modality_configs = parse_modality_configs(modality_configs)
        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
        self.use_percentiles = use_percentiles
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        # Normalization parameters computed from statistics
        self.norm_params: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
        # Format: norm_params[embodiment_tag][modality][joint_group][stat_type]
        # where stat_type in ["min", "max", "mean", "std", "dim"]

        if statistics is not None:
            self.set_statistics(statistics)

        self.train()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """
        Set dataset statistics for normalization.

        Args:
            statistics: Nested dict with structure:
                {embodiment_tag: {modality: {joint_group: {stat_type: values}}}}
        """
        for key in statistics:
            if key not in self.statistics or override:
                self.statistics[key] = deepcopy(statistics[key])
            else:
                # Surfaced as a warning (not print) because callers running with
                # override_pretraining_statistics=False on a mixture dataset will
                # otherwise silently keep the pre-existing pretraining stats and
                # discard newly-merged per-dataset stats — training proceeds with
                # the wrong mean/std and only an easy-to-miss stdout line records
                # the drop.
                logger.warning(
                    "Statistics for embodiment %r already present; new stats "
                    "DISCARDED (override=False). If the new data differs from "
                    "the existing distribution this will cause silent "
                    "normalization mismatch — pass override=True (or "
                    "override_pretraining_statistics=True at the dataset level) "
                    "to use the merged stats instead.",
                    key,
                )
        self._compute_normalization_parameters()

    def _compute_normalization_parameters(self) -> None:
        """Compute and cache normalization parameters from statistics for all embodiments and modalities."""
        for embodiment_tag in self.statistics:
            self.norm_params[embodiment_tag] = {}

            for modality in ["state", "action"]:
                if modality not in self.statistics[embodiment_tag]:
                    continue

                self.norm_params[embodiment_tag][modality] = {}

                for joint_group, stats in self.statistics[embodiment_tag][modality].items():
                    if self.use_percentiles:
                        min_vals = np.array(stats["q01"])
                        max_vals = np.array(stats["q99"])
                    else:
                        min_vals = np.array(stats["min"])
                        max_vals = np.array(stats["max"])

                    mean_vals = np.array(stats["mean"])
                    std_vals = np.array(stats["std"])

                    # Compute range, ensuring it's not zero
                    range_vals = max_vals - min_vals
                    range_vals = np.maximum(range_vals, 1e-8)

                    self.norm_params[embodiment_tag][modality][joint_group] = {
                        "min": min_vals,
                        "max": max_vals,
                        "dim": np.array(range_vals.shape[0]),
                        "mean": mean_vals,
                        "std": std_vals,
                    }

            # Override absolute action stats with relative stats where specified.
            # NOT trimmed although our embodiment is all-ABSOLUTE: the checkpoint
            # stores use_relative_action=True and carries RELATIVE-rep embodiments
            # (xdof, oxe_droid, ...) in its modality configs/statistics, so this
            # runs for THEM at processor construction. It only rewrites those
            # embodiments' norm_params; the SONIC groups are untouched.
            if "action" in self.modality_configs[embodiment_tag]:
                modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys
                action_configs = self.modality_configs[embodiment_tag]["action"].action_configs

                if action_configs is not None:
                    for key, action_config in zip(modality_keys, action_configs):
                        if (
                            action_config.rep == ActionRepresentation.RELATIVE
                            and self.use_relative_action
                        ):
                            if "relative_action" not in self.statistics[embodiment_tag]:
                                raise ValueError(
                                    f"Relative action statistics required for embodiment '{embodiment_tag}' "
                                    f"but 'relative_action' not found in statistics"
                                )
                            if key not in self.statistics[embodiment_tag]["relative_action"]:
                                raise ValueError(
                                    f"Relative action statistics required for key '{key}' "
                                    f"in embodiment '{embodiment_tag}' but not found"
                                )
                            action_dim = self.norm_params[embodiment_tag]["action"][key]["dim"]
                            self.norm_params[embodiment_tag]["action"][key] = nested_dict_to_numpy(
                                self.statistics[embodiment_tag]["relative_action"][key]
                            )
                            self.norm_params[embodiment_tag]["action"][key]["dim"] = action_dim

    def apply_state(
        self,
        state: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> dict[str, np.ndarray]:
        """
        Apply state processing (normalization, encoding).

        Args:
            state: Dict mapping joint_group -> raw state values
                Shape per group: (..., D) where D is state dimension
            embodiment_tag: Embodiment identifier (e.g., "gr1")

        Returns:
            Dict mapping joint_group -> processed state values
                - Sin/cos encoded groups: (..., 2*D)
                - Other groups: (..., D)
        """
        normalized_values = {}
        state = deepcopy(state)  # Avoid modifying input

        # Upstream strategies 1 (sin/cos) and 2 (mean/std) were trimmed —
        # every SONIC state group normalizes minmax. Fail loud if configured.
        state_config = self.modality_configs[embodiment_tag]["state"]
        if self.apply_sincos_state_encoding and getattr(
            state_config, "sin_cos_embedding_keys", None
        ):
            raise NotImplementedError(f"sin/cos state encoding {_TRIM_NOTE}")
        if getattr(state_config, "mean_std_embedding_keys", None):
            raise NotImplementedError(f"mean/std state normalization {_TRIM_NOTE}")

        for joint_group in state_config.modality_keys:
            if joint_group not in state:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in state dict for embodiment '{embodiment_tag}'"
                )

            # Min/max normalization to [-1, 1]
            params = self.norm_params[embodiment_tag]["state"][joint_group]
            normalized = normalize_values_minmax(state[joint_group], params)

            if self.clip_outliers:
                normalized = np.clip(normalized, -1.0, 1.0)

            normalized_values[joint_group] = normalized

        return normalized_values

    def unapply_action(
        self,
        action: dict[str, np.ndarray],
        embodiment_tag: str,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Reverse action processing (denormalization).

        Upstream follows this with a RELATIVE->absolute conversion step;
        trimmed — every SONIC action is ABSOLUTE, so denormalized IS absolute.

        Args:
            action: Dict mapping joint_group -> processed action values
                Shape per group: (T, D) or (B, T, D) for batched
            embodiment_tag: Embodiment identifier
            state: Unused for ABSOLUTE actions (upstream API compatibility)

        Returns:
            Dict mapping joint_group -> raw absolute action values
                Shape per group: (T, D) or (B, T, D) for batched
        """
        # Unnormalize actions
        unnormalized_values = {}
        modality_keys = self.modality_configs[embodiment_tag]["action"].modality_keys

        if self.modality_configs[embodiment_tag]["action"].mean_std_embedding_keys:
            raise NotImplementedError(f"mean/std action normalization {_TRIM_NOTE}")

        for joint_group in modality_keys:
            if joint_group not in action:
                raise KeyError(
                    f"Joint group '{joint_group}' not found in action dict for embodiment '{embodiment_tag}'"
                )

            params = self.norm_params[embodiment_tag]["action"][joint_group]
            unnormalized_values[joint_group] = unnormalize_values_minmax(
                action[joint_group], params
            )

        # Upstream step 2 (RELATIVE -> absolute) was trimmed; fail loud.
        action_configs = self.modality_configs[embodiment_tag]["action"].action_configs
        if action_configs is not None:
            for key, action_config in zip(modality_keys, action_configs):
                if action_config.rep == ActionRepresentation.RELATIVE and self.use_relative_action:
                    raise NotImplementedError(
                        f"RELATIVE->absolute conversion (key '{key}') {_TRIM_NOTE}"
                    )

        return unnormalized_values

    def apply(
        self,
        state: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        embodiment_tag: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Apply state processing (inference never carries actions to normalize).

        Args:
            state: Dict mapping joint_group -> raw state values
            action: Must be empty — training-side action normalization
                (``apply_action``) was trimmed from this extraction
            embodiment_tag: Embodiment identifier

        Returns:
            Tuple of (processed_state, {})
        """
        processed_state = self.apply_state(state, embodiment_tag)
        if action:
            raise NotImplementedError(
                f"training-side action normalization (apply_action) {_TRIM_NOTE}"
            )
        return processed_state, {}

    def __str__(self) -> str:
        return f"StateActionProcessor(modality_configs={self.modality_configs}, statistics={self.statistics}, use_percentiles={self.use_percentiles}, clip_outliers={self.clip_outliers}, apply_sincos_state_encoding={self.apply_sincos_state_encoding}, use_relative_action={self.use_relative_action})"
