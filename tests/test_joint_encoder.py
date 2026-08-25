"""Joint re-encoding: obs_dict assembly pinned against token_encoder.cpp.

The layout fixtures mirror gearsonic's fill_obs() g1 branch (offsets, joint
permutation, future-frame clamping, anchor columns); a fake encoder stands in
for the ONNX model so no checkpoint is needed. Skipped without pyarrow.
"""

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from replay.io.joint_encoder import (  # noqa: E402
    BODY_IN_STATE43,
    ENCODER_INPUT_DIM,
    FRAME_STEP,
    MUJOCO_TO_ISAACLAB,
    NUM_BODY_JOINTS,
    NUM_FRAMES,
    OFF_ANCHOR_ORI,
    OFF_MODE,
    OFF_MOTION_DQ,
    OFF_MOTION_Q,
    assemble_encoder_obs,
    encode_episode_joints,
)
from replay.session import load_reencoded_episode  # noqa: E402

IDENTITY_COLUMNS = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def make_inputs(ticks: int):
    """q_mujoco encodes (tick + joint/100); identity base quats."""
    t = np.arange(ticks, dtype=np.float64)[:, None]
    j = np.arange(NUM_BODY_JOINTS, dtype=np.float64)[None, :] / 100.0
    q = t + j
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (ticks, 1))
    return q, quat


def test_permutation_matches_joint_order_hpp():
    # Spot checks against gearsonic joint_order.hpp: mujoco waist_yaw (idx 12)
    # sits at IsaacLab index 2; mujoco left_hip_pitch (0) stays at 0;
    # mujoco right_wrist_yaw (28, last) stays at 28.
    assert MUJOCO_TO_ISAACLAB[12] == 2
    assert MUJOCO_TO_ISAACLAB[0] == 0
    assert MUJOCO_TO_ISAACLAB[28] == 28
    # A permutation: every IsaacLab slot hit exactly once.
    assert sorted(MUJOCO_TO_ISAACLAB.tolist()) == list(range(29))


def test_obs_layout_positions_and_mode():
    q, quat = make_inputs(60)
    obs = assemble_encoder_obs(q, quat)

    assert obs.shape == (60, ENCODER_INPUT_DIM)
    assert np.all(obs[:, OFF_MODE] == 0.0)  # g1 mode
    # Frame f of tick t holds the joints of tick t + 5f (IsaacLab order):
    # mujoco joint 12 (value t + 0.12) lands at IsaacLab slot 2.
    t, f = 3, 2
    slot = OFF_MOTION_Q + f * NUM_BODY_JOINTS + MUJOCO_TO_ISAACLAB[12]
    assert obs[t, slot] == pytest.approx(t + f * FRAME_STEP + 0.12)
    # teleop-mode slots stay zero (everything past the g1 anchor block).
    assert np.all(obs[:, OFF_ANCHOR_ORI + NUM_FRAMES * 6:] == 0.0)


def test_future_frames_clamp_at_episode_end():
    q, quat = make_inputs(20)  # tick 19 is the last
    obs = assemble_encoder_obs(q, quat)
    # At the last tick every future frame clamps to tick 19.
    last = obs[19, OFF_MOTION_Q:OFF_MOTION_Q + NUM_FRAMES * NUM_BODY_JOINTS]
    frames = last.reshape(NUM_FRAMES, NUM_BODY_JOINTS)
    np.testing.assert_array_equal(frames, np.tile(frames[0], (NUM_FRAMES, 1)))


def test_velocities_are_finite_difference():
    q, quat = make_inputs(60)  # positions grow by 1.0 per tick (dt = 0.02)
    obs = assemble_encoder_obs(q, quat)
    dq = obs[5, OFF_MOTION_DQ:OFF_MOTION_DQ + NUM_BODY_JOINTS]
    np.testing.assert_allclose(dq, 50.0, rtol=1e-6)  # 1.0 / 0.02


def test_anchor_first_frame_is_identity():
    # f=0 compares the base to itself -> identity rotation columns,
    # regardless of the actual orientation.
    q, _ = make_inputs(30)
    rng = np.random.default_rng(0)
    quat = rng.normal(size=(30, 4))
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    obs = assemble_encoder_obs(q, quat)
    np.testing.assert_allclose(
        obs[:, OFF_ANCHOR_ORI:OFF_ANCHOR_ORI + 6],
        np.tile(IDENTITY_COLUMNS, (30, 1)),
        atol=1e-6,
    )


def test_anchor_encodes_relative_yaw():
    # 90° yaw between current and future base -> R = Rz(90°):
    # columns [[0,-1],[1,0],[0,0]] -> [0,-1,1,0,0,0].
    q, _ = make_inputs(10)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (10, 1))
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    quat[5:] = [c, 0.0, 0.0, s]  # ticks 5+ yawed 90°
    obs = assemble_encoder_obs(q, quat)
    # tick 0, frame f=1 -> tick 5 (yawed): conj(q0)*q5 = Rz(90°)
    np.testing.assert_allclose(
        obs[0, OFF_ANCHOR_ORI + 6:OFF_ANCHOR_ORI + 12],
        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        atol=1e-6,
    )


# ── episode-level integration (fake encoder, real parquet fixture) ───────────


def write_joint_episode(path, *, ticks):
    """A LeRobot episode with the columns joint re-encoding needs."""
    state43 = np.zeros((ticks, 43))
    state43[:, BODY_IN_STATE43] = np.arange(ticks)[:, None] / 10.0
    cols = {
        "action.motion_token": [[9.9] * 64] * ticks,  # recorded tokens (to be replaced)
        "teleop.left_hand_joints": [[0.5] * 7] * ticks,
        "teleop.right_hand_joints": [[0.25] * 7] * ticks,
        "timestamp": [[i / 50.0] for i in range(ticks)],
        "frame_index": [[i] for i in range(ticks)],
        "observation.state": state43.tolist(),
        "action.wbc": (state43 + 1.0).tolist(),
        "observation.root_orientation": [[1.0, 0.0, 0.0, 0.0]] * ticks,
    }
    pq.write_table(pa.table(cols), path)
    return path


def fake_encoder(obs: np.ndarray) -> np.ndarray:
    # Token[0] = the g1-mode first joint position -> provenance is checkable.
    out = np.zeros((len(obs), 64), dtype=np.float32)
    out[:, 0] = obs[:, OFF_MOTION_Q]
    return out


def test_encode_episode_replaces_tokens_only(tmp_path):
    path = write_joint_episode(tmp_path / "e.parquet", ticks=12)
    rows, hands = encode_episode_joints(path, fake_encoder, joint_source="state")

    # Tokens are the re-encoded ones, not the recorded 9.9s.
    np.testing.assert_allclose(rows.tokens[:, 0], np.arange(12) / 10.0, atol=1e-6)
    assert not np.any(rows.tokens == 9.9)
    # Everything else still comes from the normal reader.
    assert len(rows) == 12 and hands["left"][1][0][0] == 0.5


def test_wbc_source_shifts_values(tmp_path):
    path = write_joint_episode(tmp_path / "e.parquet", ticks=8)
    rows, _ = encode_episode_joints(path, fake_encoder, joint_source="wbc")
    np.testing.assert_allclose(rows.tokens[:, 0], np.arange(8) / 10.0 + 1.0, atol=1e-6)


def test_load_reencoded_episode_builds_a_timeline(tmp_path):
    path = write_joint_episode(tmp_path / "e.parquet", ticks=15)
    timeline = load_reencoded_episode(path, fake_encoder)
    assert len(timeline) == 15
    assert not timeline.compressed_gaps
    np.testing.assert_allclose(timeline.tokens[:, 0], np.arange(15) / 10.0, atol=1e-6)
