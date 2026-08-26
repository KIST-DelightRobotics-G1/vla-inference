"""LatentActionStep — one publish tick (the pipeline's output side)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LatentActionStep:
    """One publish tick — the pure-Python twin of `kist_msgs::LatentActionStep`.

    Carries exactly the fields the replay decides. The wire struct's `seq`
    and `stamp_ns` are deliberately absent: they are publish-clock facts the
    DDS writer stamps at send time, not replay data.
    """

    frame_index: int
    token_state: np.ndarray  # (64,) float32
    left_hand_joints: np.ndarray  # (7,) float32
    right_hand_joints: np.ndarray  # (7,) float32
