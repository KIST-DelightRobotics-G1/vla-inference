"""Pin the G1 joint tables against the reference robot-model dump.

The expected index values were dumped from
``gear_sonic...instantiate_g1_robot_model(waist_location="lower_and_upper_body")``
(pinocchio, g1_29dof_with_hand.urdf). If these tests fail after an edit, the
edit is wrong — regenerate the tables from the reference model instead.
"""

import numpy as np

from common.g1_joints import (
    BODY_ACTUATED_INDICES,
    CLOSED_HAND_Q,
    G1_JOINT_NAMES,
    G1_NUM_JOINTS,
    JOINT_GROUP_INDICES,
    LEFT_HAND_ACTUATED_INDICES,
    RIGHT_HAND_ACTUATED_INDICES,
    apply_hand_hardware_coupling,
    assemble_full_q,
    split_state,
)


def test_table_matches_reference_dump():
    assert G1_NUM_JOINTS == 43
    assert len(G1_JOINT_NAMES) == 43
    assert BODY_ACTUATED_INDICES == list(range(22)) + list(range(29, 36))
    assert LEFT_HAND_ACTUATED_INDICES == [26, 27, 28, 22, 23, 24, 25]
    assert RIGHT_HAND_ACTUATED_INDICES == [40, 41, 42, 36, 37, 38, 39]
    assert JOINT_GROUP_INDICES["left_leg"] == [0, 1, 2, 3, 4, 5]
    assert JOINT_GROUP_INDICES["right_leg"] == [6, 7, 8, 9, 10, 11]
    assert JOINT_GROUP_INDICES["waist"] == [12, 13, 14]
    assert JOINT_GROUP_INDICES["left_arm"] == [15, 16, 17, 18, 19, 20, 21]
    assert JOINT_GROUP_INDICES["right_arm"] == [29, 30, 31, 32, 33, 34, 35]
    assert JOINT_GROUP_INDICES["left_hand"] == [22, 23, 24, 25, 26, 27, 28]
    assert JOINT_GROUP_INDICES["right_hand"] == [36, 37, 38, 39, 40, 41, 42]


def test_group_indices_partition_full_q():
    """The 7 groups exactly cover all 43 joints with no overlap."""
    all_indices = [i for idx in JOINT_GROUP_INDICES.values() for i in idx]
    assert sorted(all_indices) == list(range(G1_NUM_JOINTS))


def test_hand_motor_order_semantics():
    """Dex motor order (thumb×3, index×2, middle×2) maps to the right URDF joints."""
    left_names = [G1_JOINT_NAMES[i] for i in LEFT_HAND_ACTUATED_INDICES]
    assert left_names == [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
    ]


def test_assemble_split_roundtrip():
    rng = np.random.default_rng(0)
    body_q = rng.normal(size=29).astype(np.float32)
    left_hand_q = rng.normal(size=7).astype(np.float32)
    right_hand_q = rng.normal(size=7).astype(np.float32)

    full_q = assemble_full_q(body_q, left_hand_q, right_hand_q)
    groups = split_state(full_q)

    # Body groups come back in Unitree motor order slices
    np.testing.assert_array_equal(groups["left_leg"], body_q[0:6])
    np.testing.assert_array_equal(groups["right_leg"], body_q[6:12])
    np.testing.assert_array_equal(groups["waist"], body_q[12:15])
    np.testing.assert_array_equal(groups["left_arm"], body_q[15:22])
    np.testing.assert_array_equal(groups["right_arm"], body_q[22:29])

    # Hand groups are in URDF order (index, middle, thumb), i.e. the inverse
    # permutation of the Dex motor order (thumb, index, middle).
    np.testing.assert_array_equal(
        groups["left_hand"],
        left_hand_q[[3, 4, 5, 6, 0, 1, 2]],
    )
    np.testing.assert_array_equal(
        groups["right_hand"],
        right_hand_q[[3, 4, 5, 6, 0, 1, 2]],
    )


def test_hand_hardware_coupling():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 9.9, 9.9], dtype=np.float32)
    fixed = apply_hand_hardware_coupling(q)
    assert fixed[5] == fixed[3] == np.float32(0.4)
    assert fixed[6] == fixed[4] == np.float32(0.5)
    # Original untouched
    assert q[5] == np.float32(9.9)


def test_closed_hand_values_match_reference_formula():
    # G1GripperInverseKinematicsSolver._get_middle_close_q_desired()
    expected_left = np.array([0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5], dtype=np.float32)
    np.testing.assert_allclose(CLOSED_HAND_Q["left"], expected_left)
    np.testing.assert_allclose(CLOSED_HAND_Q["right"], -expected_left)
