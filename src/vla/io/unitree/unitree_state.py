"""UnitreeState + IMU — the lowstate snapshot, mirroring gearsonic's structs.

Field-for-field twins of kist-gearsonic-inference's `unitree_state.hpp` /
`imu.hpp` (same names, same units), with the motor array-of-structs
flattened to numpy struct-of-arrays.
"""

from dataclasses import dataclass

import numpy as np

NUM_MOTORS = 29  # Unitree G1 (gearsonic kNumMotors)


@dataclass(frozen=True)
class IMU:
    """gearsonic `IMU` — quaternion is (w, x, y, z), scalar-first."""

    quaternion: np.ndarray  # (4,) float64, wxyz
    gyroscope: np.ndarray  # (3,) float64, rad/s
    accelerometer: np.ndarray  # (3,) float64, m/s^2


@dataclass(frozen=True)
class UnitreeState:
    """gearsonic `UnitreeState` — one converted rt/lowstate sample.

    `q`/`dq`/`tau` are the 29 body motors in Unitree motor order (left leg,
    right leg, waist, left arm, right arm) = LowState motor_state[0:29];
    `imu_pelvis` is LowState.imu_state — the PELVIS IMU, not the torso IMU
    on rt/secondary_imu.
    """

    q: np.ndarray  # (29,) float64, rad
    dq: np.ndarray  # (29,) float64, rad/s
    tau: np.ndarray  # (29,) float64, Nm (tau_est)
    imu_pelvis: IMU
    tick: int
    mode_machine: int
