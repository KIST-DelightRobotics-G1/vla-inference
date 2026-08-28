"""HandState — one converted Dex3 rt/dex3/{side}/state sample.

Not part of gearsonic's UnitreeStateReader (that side only WRITES hand
commands); the VLA observation needs the measured hand joints, so this
carries them in the same struct style.
"""

from dataclasses import dataclass

import numpy as np

NUM_HAND_MOTORS = 7  # Dex3-1: thumb x3, index x2, middle x2


@dataclass(frozen=True)
class HandState:
    q: np.ndarray  # (7,) float64, rad — HandState_.motor_state[0:7].q, Dex motor order
