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

from typing import Any

import numpy as np
from transformers import ProcessorMixin

from vla.policy.gr00t.data.types import EmbodimentTag, ModalityConfig


class BaseProcessor(ProcessorMixin):
    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Process a list of messages and return a dictionary of model inputs.

        Args:
            messages (list[dict[str, Any]]): List of messages to process.

        Returns:
            dict[str, Any]: Dictionary of model inputs.

        Example:
        >>> processor = BaseProcessor()
        >>> messages = [
        >>>    {"type": MessageType.START_OF_EPISODE.value, "content": ""},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.TEXT.value, "role" : "user", "content": "Please give me the apple"},
        >>>    {"type": MessageType.TEXT.value, "role" : "assistant", "content": "I need to move my left hand to get the apple"},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.EPISODE_STEP.value, "content": VLAStepData},
        >>>    {"type": MessageType.END_OF_EPISODE.value, "content": ""},
        >>> ]
        >>> model_input = processor(messages)
        >>> print(model_input)
        """
        raise NotImplementedError("Subclasses must implement __call__")

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Decode the action from the model output."""
        raise NotImplementedError("Subclasses must implement decode_action")

    @property
    def collator(self):
        raise NotImplementedError("Subclasses must implement collator")

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def get_modality_configs(self) -> dict[str, dict[str, ModalityConfig]]:
        """Get the modality configurations.

        Returns:
            dict[str, dict[str, ModalityConfig]]: The modality configurations, where
                modality_configs[embodiment_tag][modality] = ModalityConfig
        """
        return getattr(self, "modality_configs")
