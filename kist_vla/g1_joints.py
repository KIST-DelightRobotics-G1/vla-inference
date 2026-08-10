"""Static Unitree G1 joint-index tables and state assembly.

Replaces gear_sonic's pinocchio-backed RobotModel for the only two things the
VLA runner needs from it: assembling the full configuration vector from
actuated joint readings, and splitting it into the per-group state arrays the
UNITREE_G1_SONIC embodiment expects.

The tables below were dumped from the reference model
(``gear_sonic.data.robot_model.instantiation.g1.instantiate_g1_robot_model``
with ``waist_location="lower_and_upper_body"``, URDF
``g1_29dof_with_hand.urdf``) rather than hand-authored — do not edit them
without re-dumping. ``tests/test_g1_joints.py`` pins them.

Wire conventions:
- ``body_q`` (29,) arrives in Unitree motor order: left leg(6), right leg(6),
  waist yaw/roll/pitch(3), left arm(7), right arm(7).
- ``left_hand_q`` / ``right_hand_q`` (7,) arrive in Dex hand motor order:
  thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1 — which is
  NOT the URDF order (index, middle, thumb); the *_HAND_ACTUATED_INDICES
  permutations encode that difference.
"""

import numpy as np

G1_NUM_JOINTS = 43

# Full-configuration joint order (pinocchio / URDF g1_29dof_with_hand).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]

# body_q[i] (Unitree 29-DoF motor order) lands at full_q[BODY_ACTUATED_INDICES[i]].
BODY_ACTUATED_INDICES = [
    0, 1, 2, 3, 4, 5,            # left leg
    6, 7, 8, 9, 10, 11,          # right leg
    12, 13, 14,                  # waist
    15, 16, 17, 18, 19, 20, 21,  # left arm
    29, 30, 31, 32, 33, 34, 35,  # right arm
]

# hand_q[i] (Dex motor order: thumb×3, index×2, middle×2) lands at
# full_q[*_HAND_ACTUATED_INDICES[i]].
LEFT_HAND_ACTUATED_INDICES = [26, 27, 28, 22, 23, 24, 25]
RIGHT_HAND_ACTUATED_INDICES = [40, 41, 42, 36, 37, 38, 39]

# Per-group indices into full_q — key names and per-group joint order match
# the UNITREE_G1_SONIC "state" modality keys.
JOINT_GROUP_INDICES = {
    "left_leg": [0, 1, 2, 3, 4, 5],
    "right_leg": [6, 7, 8, 9, 10, 11],
    "waist": [12, 13, 14],
    "left_arm": [15, 16, 17, 18, 19, 20, 21],
    "right_arm": [29, 30, 31, 32, 33, 34, 35],
    "left_hand": [22, 23, 24, 25, 26, 27, 28],
    "right_hand": [36, 37, 38, 39, 40, 41, 42],
}

STATE_GROUPS = list(JOINT_GROUP_INDICES.keys())

# Closed-hand joint targets in Dex motor order, precomputed from the reference
# G1GripperInverseKinematicsSolver._get_middle_close_q_desired() ("middle
# gesture"). Right hand has a mirrored joint convention (negated).
_MIDDLE_CLOSE_LEFT = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32)
CLOSED_HAND_Q = {"left": _MIDDLE_CLOSE_LEFT, "right": -_MIDDLE_CLOSE_LEFT}
OPEN_HAND_Q = np.zeros(7, dtype=np.float32)


def apply_hand_hardware_coupling(left_hand_q: np.ndarray) -> np.ndarray:
    """Copy index-finger readings onto the middle-finger slots (left hand).

    The left Dex hand's middle finger is mechanically coupled to the index
    finger on the KIST gripper setup, so its encoders don't report real
    values; the reference pipeline overwrites them with the index readings
    before building the observation. Returns a copy.
    """
    q = np.array(left_hand_q, dtype=np.float32, copy=True)
    q[5] = q[3]
    q[6] = q[4]
    return q


def assemble_full_q(
    body_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
) -> np.ndarray:
    """Build the 43-dim full configuration from actuated joint readings.

    Args:
        body_q: (..., 29) in Unitree motor order.
        left_hand_q: (..., 7) in Dex motor order.
        right_hand_q: (..., 7) in Dex motor order.

    Returns:
        (..., 43) full configuration, float32, URDF joint order.
    """
    body_q = np.asarray(body_q, dtype=np.float32)
    left_hand_q = np.asarray(left_hand_q, dtype=np.float32)
    right_hand_q = np.asarray(right_hand_q, dtype=np.float32)
    assert body_q.shape[-1] == len(BODY_ACTUATED_INDICES), (
        f"body_q must have {len(BODY_ACTUATED_INDICES)} joints, got {body_q.shape}"
    )
    assert left_hand_q.shape[-1] == 7 and right_hand_q.shape[-1] == 7

    q = np.zeros(body_q.shape[:-1] + (G1_NUM_JOINTS,), dtype=np.float32)
    q[..., BODY_ACTUATED_INDICES] = body_q
    q[..., LEFT_HAND_ACTUATED_INDICES] = left_hand_q
    q[..., RIGHT_HAND_ACTUATED_INDICES] = right_hand_q
    return q


def split_state(full_q: np.ndarray) -> dict[str, np.ndarray]:
    """Split the full configuration into the 7 per-group state arrays.

    Args:
        full_q: (..., 43) full configuration.

    Returns:
        Dict of group name -> (..., group_size) float32 arrays.
    """
    full_q = np.asarray(full_q, dtype=np.float32)
    assert full_q.shape[-1] == G1_NUM_JOINTS, (
        f"full_q must have {G1_NUM_JOINTS} joints, got {full_q.shape}"
    )
    return {group: full_q[..., idx] for group, idx in JOINT_GROUP_INDICES.items()}
