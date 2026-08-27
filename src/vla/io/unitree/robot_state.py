"""RobotState — one combined robot-state snapshot, the subscriber's product."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """The newest whole-robot state, streams combined onto one snapshot.

    Wire semantics (verified against the reference deploy):

    - `body_q`: 29 absolute joint angles in Unitree motor order (left leg,
      right leg, waist, left arm, right arm) — LowState motor_state[0:29].q.
    - `base_quat`: **pelvis** IMU quaternion (w, x, y, z) — LowState
      imu_state.quaternion, NOT the torso IMU on rt/secondary_imu.
    - `left_hand_q` / `right_hand_q`: Dex3 motor order (thumb x3, index x2,
      middle x2) — HandState motor_state[0:7].q, hardware couplings NOT
      applied here (that is observation assembly's job).
    """

    body_q: np.ndarray  # (29,) float64
    left_hand_q: np.ndarray  # (7,) float64
    right_hand_q: np.ndarray  # (7,) float64
    base_quat: np.ndarray  # (4,) float64, wxyz
