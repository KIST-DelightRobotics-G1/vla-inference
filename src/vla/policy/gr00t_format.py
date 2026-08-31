"""Observation <-> gr00t format conversion (pure numpy, no torch).

The vendored Gr00tPolicy speaks batched, time-axed nested dicts; our
Observation is one unbatched moment. The conversion is pure reshaping — all
normalization, image transforms and padding live inside the vendored
processor — so it is testable on the host without the inference stack.
"""

import numpy as np

from .action_chunk import ActionChunk
from vla.observation import Observation


def to_gr00t_observation(observation: Observation, language_key: str) -> dict:
    """Wrap one Observation into Gr00tPolicy's batched observation dict.

    video:    view -> (1, 1, H, W, 3) uint8   (batch 1, history 1)
    state:    group -> (1, 1, D) float32
    language: language_key -> [[prompt]]      (list[list[str]], (B, T))
    """
    return {
        "video": {
            view: np.ascontiguousarray(frame)[None, None]
            for view, frame in observation.video.items()
        },
        "state": {
            group: np.asarray(values, dtype=np.float32)[None, None]
            for group, values in observation.state.items()
        },
        "language": {language_key: [[observation.prompt]]},
    }


def to_action_chunk(action: dict) -> ActionChunk:
    """Unbatch Gr00tPolicy's (B, T, D) action dict into one ActionChunk."""
    return ActionChunk(
        motion_token=np.asarray(action["motion_token"][0], dtype=np.float32),
        left_hand_joints=np.asarray(action["left_hand_joints"][0], dtype=np.float32),
        right_hand_joints=np.asarray(action["right_hand_joints"][0], dtype=np.float32),
    )
