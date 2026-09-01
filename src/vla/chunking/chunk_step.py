"""ChunkStep — one 20 ms tick of action, the stage's contract with publisher.

What the wire needs for a single LatentActionStep, cut out of an
ActionChunk by the cursor: one SONIC token and the two hand targets. The
publisher adds the wire bookkeeping (frame_index, timing).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChunkStep:
    """One tick's action content, sliced from the active ActionChunk."""

    motion_token: np.ndarray  # (64,) float32
    left_hand_joints: np.ndarray  # (7,) float32
    right_hand_joints: np.ndarray  # (7,) float32
