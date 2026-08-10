"""Latent protocol v4 wire-format tests.

The byte layout is a contract with the gearsonic C++ parser — these tests pin
it (topic prefix, 1280-byte NUL-padded JSON header, little-endian payload
order) independently of the pack/unpack helpers sharing bugs.
"""

import json

import numpy as np
import pytest

from kist_vla.protocol import (
    HEADER_SIZE,
    build_command_message,
    pack_latent_action_message,
    pack_pose_message,
    unpack_message,
)


def test_header_size_matches_reference():
    # gear_sonic/utils/teleop/zmq/zmq_planner_sender.py: HEADER_SIZE = 1280
    assert HEADER_SIZE == 1280


def test_latent_action_roundtrip():
    token = np.linspace(-1, 1, 64, dtype=np.float32)
    left = np.arange(7, dtype=np.float32)
    right = -np.arange(7, dtype=np.float32)

    msg = pack_latent_action_message(
        motion_token=token,
        frame_index=np.array([17], dtype=np.int64),
        left_hand_joints=left,
        right_hand_joints=right,
    )

    header, fields = unpack_message(msg, topic="pose")
    assert header["v"] == 4
    assert header["endian"] == "le"
    assert header["count"] == 1

    np.testing.assert_array_equal(fields["token_state"], token.reshape(1, 64))
    np.testing.assert_array_equal(fields["frame_index"], np.array([17], dtype=np.int64))
    np.testing.assert_array_equal(fields["left_hand_joints"], left.reshape(1, 7))
    np.testing.assert_array_equal(fields["right_hand_joints"], right.reshape(1, 7))


def test_latent_action_raw_layout():
    """Pin the exact byte layout the C++ parser reads."""
    token = np.zeros(64, dtype=np.float32)
    msg = pack_latent_action_message(token, np.array([0], dtype=np.int64))

    assert msg.startswith(b"pose")
    header_raw = msg[4 : 4 + HEADER_SIZE]
    header = json.loads(header_raw.rstrip(b"\x00"))
    assert [f["name"] for f in header["fields"]] == ["token_state", "frame_index"]
    assert header["fields"][0] == {"name": "token_state", "dtype": "f32", "shape": [1, 64]}
    assert header["fields"][1] == {"name": "frame_index", "dtype": "i64", "shape": [1]}
    # Payload = 64 f32 + 1 i64
    assert len(msg) == 4 + HEADER_SIZE + 64 * 4 + 8


def test_latent_action_shape_validation():
    token = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError):
        pack_latent_action_message(
            token, np.array([0]), left_hand_joints=np.zeros(6, dtype=np.float32)
        )


def test_latent_action_without_hands():
    msg = pack_latent_action_message(np.zeros((1, 64), dtype=np.float32), 3)
    _, fields = unpack_message(msg, topic="pose")
    assert set(fields.keys()) == {"token_state", "frame_index"}
    assert fields["frame_index"][0] == 3


def test_command_message_layout():
    msg = build_command_message(start=True, stop=False, planner=True)
    assert msg.startswith(b"command")
    header, fields = unpack_message(msg, topic="command")
    assert header["v"] == 1
    assert fields["start"][0] == 1
    assert fields["stop"][0] == 0
    assert fields["planner"][0] == 1
    assert "delta_heading" not in fields


def test_command_message_with_heading():
    msg = build_command_message(start=False, stop=True, planner=False, delta_heading=0.5)
    _, fields = unpack_message(msg, topic="command")
    assert fields["stop"][0] == 1
    np.testing.assert_allclose(fields["delta_heading"], [0.5], rtol=1e-6)


def test_pose_message_casts_unknown_float_dtypes():
    msg = pack_pose_message({"x": np.array([1.0, 2.0], dtype=np.float16)})
    _, fields = unpack_message(msg, topic="pose")
    assert fields["x"].dtype == np.float32
