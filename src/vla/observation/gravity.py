"""Gravity projection for the SONIC state observation (pure numpy).

The reference implementation (gear_sonic transforms.compute_projected_gravity)
used scipy's Rotation; this is the same math without the dependency: rotate
the world gravity direction into the body frame with the conjugate
quaternion.
"""

import numpy as np


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Project the world gravity direction [0, 0, -1] into the body frame.

    Args:
        base_quat: base quaternion [w, x, y, z] (scalar-first — the pelvis
            IMU convention, hardware-verified 2026-08-28).

    Returns:
        (3,) float32 gravity direction in the robot's body frame. Invariant
        to yaw (rotation about the world z axis), so the power-on heading
        does not matter.
    """
    base_quat = np.asarray(base_quat, dtype=np.float64)
    if base_quat.shape != (4,):
        raise ValueError(f"base_quat must have shape (4,), got {base_quat.shape}")

    w, x, y, z = base_quat / np.linalg.norm(base_quat)
    # Third column of R(q)^T applied to [0, 0, -1]: g_body = -R^T e_z.
    return np.array(
        [
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ],
        dtype=np.float32,
    )
