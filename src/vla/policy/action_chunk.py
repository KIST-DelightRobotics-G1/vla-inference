"""ActionChunk — one model prediction, the stage's contract with chunking.

40 future steps at the 50 Hz control rate (0.78 s span), decoded to physical
units — all three components are ABSOLUTE targets. Each motion_token row is
one SONIC latent (the gearsonic decoder's input); the hand columns are joint
targets in the Dex3 wire order the model was trained on.
"""

from dataclasses import dataclass

import numpy as np

ACTION_HORIZON = 40
MOTION_TOKEN_DIM = 64
HAND_DIM = 7


@dataclass(frozen=True)
class ActionChunk:
    """One decoded prediction: what the robot should do for the next 0.78 s.

    Holding one proves the policy ran to completion — shapes are validated at
    construction so downstream stages never re-check.
    """

    motion_token: np.ndarray  # (40, 64) float32, SONIC latents
    left_hand_joints: np.ndarray  # (40, 7) float32, absolute targets
    right_hand_joints: np.ndarray  # (40, 7) float32, absolute targets

    def __post_init__(self):
        expected = {
            "motion_token": (ACTION_HORIZON, MOTION_TOKEN_DIM),
            "left_hand_joints": (ACTION_HORIZON, HAND_DIM),
            "right_hand_joints": (ACTION_HORIZON, HAND_DIM),
        }
        for name, shape in expected.items():
            arr = getattr(self, name)
            if arr.shape != shape or arr.dtype != np.float32:
                raise ValueError(
                    f"{name} must be float32 {shape}, got {arr.dtype} {arr.shape}"
                )
