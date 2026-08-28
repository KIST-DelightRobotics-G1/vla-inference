"""HandState — one converted Dex3 rt/dex3/{side}/state sample.

Not part of gearsonic's UnitreeStateReader (that side only WRITES hand
commands); the VLA observation needs the measured hand joints, so this
carries them in the same struct style.
"""

from dataclasses import dataclass

import numpy as np

# Dex3-1 wire order, MEASURED on the real hands (2026-08-28, both hands):
#   q[0:3] thumb, q[3:5] the MIDDLE-position finger, q[5:7] the INDEX-position
#   finger. NOTE: the reference stack labels q[3:5] "index" and q[5:7]
#   "middle" — physically swapped, but training data and inference both use
#   the same wire order, so the model sees a consistent signal either way.
#   Never "fix" the order in code; it would desync from the checkpoint.
NUM_HAND_MOTORS = 7


@dataclass(frozen=True)
class HandState:
    q: np.ndarray  # (7,) float64, rad — HandState_.motor_state[0:7].q, wire order (see above)
