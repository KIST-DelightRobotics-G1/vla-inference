"""Joint loading + re-encoding: pinned against gearsonic's token_encoder.cpp.

The obs-assembly fixtures mirror fill_obs()'s g1 branch (offsets, joint
permutation, future-frame clamping, anchor columns); a fake encoder stands in
for the ONNX model so no checkpoint is needed. The loader fixtures write the
real collector/LeRobot formats. Skipped without pyarrow.
"""

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from replay.encoder.encoder import (  # noqa: E402
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
    encode_tokens_from_joints,
)
from replay.aligner import align_joints, align_tokens  # noqa: E402
from replay.io.parquet_io import BODY_IN_STATE43  # noqa: E402
from replay.io import csv_io, parquet_io  # noqa: E402
from replay.builder import build_timeline  # noqa: E402

IDENTITY_COLUMNS = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def make_inputs(ticks: int):
    """q_mujoco encodes (tick + joint/100); identity base quats."""
    t = np.arange(ticks, dtype=np.float64)[:, None]
    j = np.arange(NUM_BODY_JOINTS, dtype=np.float64)[None, :] / 100.0
    q = t + j
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (ticks, 1))
    return q, quat


# ── obs_dict assembly (pure encoding transform) ──────────────────────────────


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


def test_velocities_finite_difference_and_measured():
    q, quat = make_inputs(60)  # positions grow by 1.0 per tick (dt = 0.02)
    obs = assemble_encoder_obs(q, quat)
    dq = obs[5, OFF_MOTION_DQ:OFF_MOTION_DQ + NUM_BODY_JOINTS]
    np.testing.assert_allclose(dq, 50.0, rtol=1e-6)  # 1.0 / 0.02

    # Measured velocities take precedence over finite differences.
    obs = assemble_encoder_obs(q, quat, dq_mujoco=np.full_like(q, 7.0))
    dq = obs[5, OFF_MOTION_DQ:OFF_MOTION_DQ + NUM_BODY_JOINTS]
    np.testing.assert_allclose(dq, 7.0)


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


# ── parquet joint loading + composition ──────────────────────────────────────


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


def joints_timeline(tokens, joints, **kw):
    """Align + encoding stage + resampling, as cli.main wires them."""
    encoded = encode_tokens_from_joints(
        align_tokens(tokens), align_joints(tokens, joints), fake_encoder
    )
    return build_timeline(encoded, **kw)


def test_parquet_load_joints_reads_disk_faithfully(tmp_path):
    path = write_joint_episode(tmp_path / "e.parquet", ticks=12)
    joints = parquet_io.read_joints(path)

    np.testing.assert_allclose(joints.q[:, 0], np.arange(12) / 10.0, atol=1e-6)
    np.testing.assert_allclose(joints.base_quat[0], [1, 0, 0, 0])
    assert joints.dq is None  # the export carries no velocities

    tokens = parquet_io.read_tokens(path)
    assert np.all(tokens.values == np.float32(9.9))  # recorded tokens untouched
    assert tokens.left_hand[1][0][0] == 0.5


def test_parquet_joints_compose_into_a_timeline(tmp_path):
    path = write_joint_episode(tmp_path / "e.parquet", ticks=15)
    timeline = joints_timeline(parquet_io.read_tokens(path), parquet_io.read_joints(path))

    assert len(timeline) == 15
    assert not timeline.compressed_gaps
    np.testing.assert_allclose(timeline.tokens[:, 0], np.arange(15) / 10.0, atol=1e-6)


# ── collector-session (CSV) joint loading ────────────────────────────────────

T0 = 1_700_000_000_000_000_000
CONTROL_DT_NS = 20_000_000


def write_lowstate_csv(path, *, rows, period_ns=CONTROL_DT_NS // 5):
    """lowstate.csv per the collector's lowstate_rows.hpp: 35 motor slots,
    m*_q/dq/ddq/tau + IMU columns; runs on its own (faster) clock."""
    header = "recv_ns,tick,mode_machine,mode_pr,quat_w,quat_x,quat_y,quat_z," \
             "gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z," \
             "rpy_roll,rpy_pitch,rpy_yaw,imu_temp"
    header += "".join(f",m{i:02d}_q,m{i:02d}_dq,m{i:02d}_ddq,m{i:02d}_tau" for i in range(35))
    lines = [header]
    for r in range(rows):
        recv = T0 + r * period_ns
        t_s = r * period_ns / 1e9
        motors = ",".join(f"{t_s + i / 100.0:.7g},{2.5:.7g},0,0" for i in range(35))
        lines.append(f"{recv},{r},0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,30,{motors}")
    path.write_text("\n".join(lines) + "\n")
    return path


def write_token_grid_csv(path, *, ticks):
    header = "recv_ns,stamp_ns,seq,arbiter_mode,encoder_mode"
    header += "".join(f",t{i:02d}" for i in range(64))
    lines = [header]
    for i in range(ticks):
        stamp = T0 + i * CONTROL_DT_NS
        token = ",".join("9.9" for _ in range(64))
        lines.append(f"{stamp},{stamp},{i + 1},1,1,{token}")
    path.write_text("\n".join(lines) + "\n")


def test_read_session_joints_shapes_and_order(tmp_path):
    write_lowstate_csv(tmp_path / "lowstate.csv", rows=10)
    joints = csv_io.read_joints(tmp_path)

    assert joints.q.shape == (10, 29) and joints.dq.shape == (10, 29)
    assert joints.base_quat.shape == (10, 4)
    # row r, motor i -> value t_s + i/100 (only the 29 body motors are read)
    assert joints.q[5, 12] == pytest.approx(5 * 0.004 + 0.12)
    np.testing.assert_allclose(joints.dq, 2.5)
    np.testing.assert_allclose(joints.base_quat[0], [1, 0, 0, 0])


def test_csv_joints_align_and_compose(tmp_path):
    write_token_grid_csv(tmp_path / "motion_token.csv", ticks=20)
    write_lowstate_csv(tmp_path / "lowstate.csv", rows=100)  # 5x faster clock

    tokens = csv_io.read_tokens(tmp_path)
    joints = csv_io.read_joints(tmp_path)
    np.testing.assert_allclose(joints.dq, 2.5)  # measured velocities carried

    timeline = joints_timeline(tokens, joints)
    assert len(timeline) == 20
    # Newest lowstate row aligned onto each 20 ms tick: token[0] = tick time.
    np.testing.assert_allclose(timeline.tokens[:, 0], np.arange(20) * 0.02, atol=1e-6)
    assert not np.any(timeline.tokens == np.float32(9.9))  # recorded tokens replaced


def test_csv_load_joints_requires_lowstate(tmp_path):
    write_token_grid_csv(tmp_path / "motion_token.csv", ticks=5)
    with pytest.raises(FileNotFoundError, match="lowstate.csv"):
        csv_io.read_joints(tmp_path)
