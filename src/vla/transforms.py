"""Gravity projection for the SONIC state observation.

Vendored from GR00T-WholeBodyControl
``gear_sonic/utils/data_collection/transforms.py`` (compute_projected_gravity
only) so this package does not depend on gear_sonic.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_projected_gravity(base_quat: np.ndarray) -> np.ndarray:
    """Project the world gravity direction [0, 0, -1] into the body frame.

    Args:
        base_quat: Base quaternion [qw, qx, qy, qz], shape (4,), scalar-first.

    Returns:
        (3,) float32 gravity direction in the robot's body frame.
    """
    base_quat = np.asarray(base_quat, dtype=np.float64)
    if base_quat.shape != (4,):
        raise ValueError(f"base_quat must have shape (4,), got {base_quat.shape}")

    gravity_vec_world = np.array([0.0, 0.0, -1.0])
    base_rotation = R.from_quat(base_quat, scalar_first=True)
    projected_gravity = base_rotation.inv().apply(gravity_vec_world)

    return projected_gravity.astype(np.float32)
