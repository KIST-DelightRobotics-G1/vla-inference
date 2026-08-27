"""DDS input-source loopback tests (state from unitree topics, camera H.264).

Fake publishers speak the exact wire types the real robot / kist-ext-sensor-io
publish, so these pin the field mappings (motor order slicing, IMU quaternion,
Dex hand passthrough, H.264 keyframe resync) without hardware.
"""

import time

import numpy as np
import pytest

cyclonedds = pytest.importorskip("cyclonedds")

from cyclonedds.core import Policy, Qos  # noqa: E402
from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.pub import DataWriter  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402

TEST_DOMAIN = 218

# ---------------------------------------------------------------------------
# State source
# ---------------------------------------------------------------------------

unitree_idl = pytest.importorskip("unitree_sdk2py.idl.unitree_hg.msg.dds_")

from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_hg_msg_dds__HandState_,
    unitree_hg_msg_dds__LowState_,
)

from vla_old.state_source import (  # noqa: E402
    LEFT_HAND_TOPIC,
    LOWSTATE_TOPIC,
    RIGHT_HAND_TOPIC,
    DdsStateSource,
)


def _writer_qos():
    return Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))


@pytest.fixture
def state_setup():
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

    participant = DomainParticipant(TEST_DOMAIN)
    low_writer = DataWriter(
        participant, Topic(participant, LOWSTATE_TOPIC, LowState_), qos=_writer_qos()
    )
    left_writer = DataWriter(
        participant, Topic(participant, LEFT_HAND_TOPIC, HandState_), qos=_writer_qos()
    )
    right_writer = DataWriter(
        participant, Topic(participant, RIGHT_HAND_TOPIC, HandState_), qos=_writer_qos()
    )
    source = DdsStateSource(domain_id=TEST_DOMAIN)
    time.sleep(0.3)  # discovery
    yield source, low_writer, left_writer, right_writer
    source.close()


def _make_lowstate():
    low = unitree_hg_msg_dds__LowState_()
    for i, motor in enumerate(low.motor_state):
        motor.q = i * 0.1
    low.imu_state.quaternion = [0.5, -0.5, 0.5, -0.5]
    return low


def _make_handstate(offset: float):
    hand = unitree_hg_msg_dds__HandState_()
    for i, motor in enumerate(hand.motor_state[:7]):
        motor.q = offset + i
    return hand


def test_state_mapping(state_setup):
    source, low_writer, left_writer, right_writer = state_setup

    low_writer.write(_make_lowstate())
    left_writer.write(_make_handstate(100.0))
    right_writer.write(_make_handstate(200.0))
    time.sleep(0.2)

    msg = None
    deadline = time.monotonic() + 5.0
    while msg is None and time.monotonic() < deadline:
        msg = source.get_msg()
        if msg is None:
            low_writer.write(_make_lowstate())
            time.sleep(0.05)
    assert msg is not None

    # body_q = first 29 motors in wire order
    np.testing.assert_allclose(msg["body_q"], np.arange(29) * 0.1, atol=1e-6)
    # base_quat = pelvis IMU quaternion, wxyz passthrough
    np.testing.assert_allclose(msg["base_quat"], [0.5, -0.5, 0.5, -0.5])
    # hands = first 7 motors, Dex motor order passthrough
    np.testing.assert_allclose(msg["left_hand_q"], 100.0 + np.arange(7))
    np.testing.assert_allclose(msg["right_hand_q"], 200.0 + np.arange(7))


def test_state_requires_fresh_lowstate(state_setup):
    source, low_writer, left_writer, right_writer = state_setup

    # No lowstate yet -> None
    assert source.get_msg() is None

    left_writer.write(_make_handstate(1.0))
    right_writer.write(_make_handstate(2.0))
    low_writer.write(_make_lowstate())
    time.sleep(0.3)

    assert source.get_msg() is not None
    # Consumed; no new lowstate since -> None (hands alone don't refresh)
    assert source.get_msg() is None


def test_state_requires_hands_by_default(state_setup):
    source, low_writer, _, _ = state_setup
    low_writer.write(_make_lowstate())
    time.sleep(0.3)
    # Hands never arrived -> None under require_hands=True (default)
    assert source.get_msg() is None


# ---------------------------------------------------------------------------
# Camera source
# ---------------------------------------------------------------------------

av = pytest.importorskip("av")

from vla_old.camera_source import DdsCameraSource, get_camera_types  # noqa: E402

CAMERA_TOPIC = "rt/kist/camera/test/color/h264"


def _encode_h264_frames(images: list[np.ndarray]):
    """Encode RGB frames; return [(bytes, is_keyframe)] per encoded packet."""
    codec = av.CodecContext.create("h264", "w")
    codec.width = images[0].shape[1]
    codec.height = images[0].shape[0]
    codec.pix_fmt = "yuv420p"
    codec.framerate = 30
    codec.options = {"g": "5", "tune": "zerolatency"}

    packets = []
    for img in images:
        frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format="yuv420p")
        packets.extend(codec.encode(frame))
    packets.extend(codec.encode(None))  # flush
    return [(bytes(p), bool(p.is_keyframe)) for p in packets]


def test_camera_decode_roundtrip():
    CompressedColorFrame = get_camera_types()
    participant = DomainParticipant(TEST_DOMAIN)
    writer = DataWriter(
        participant,
        Topic(participant, CAMERA_TOPIC, CompressedColorFrame),
        qos=Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(32)),
    )
    source = DdsCameraSource(domain_id=TEST_DOMAIN, topic=CAMERA_TOPIC)
    time.sleep(0.3)  # discovery

    # Solid-gray frames compress predictably; the stream ends bright. Two
    # bright frames because the decoder-side H.264 parser holds the final
    # NAL until it sees the next boundary (one-frame latency on a live
    # stream) — the last packet stays buffered, the first bright one decodes.
    images = [np.full((48, 64, 3), 60, dtype=np.uint8) for _ in range(4)]
    images.append(np.full((48, 64, 3), 200, dtype=np.uint8))
    images.append(np.full((48, 64, 3), 200, dtype=np.uint8))
    encoded = _encode_h264_frames(images)
    assert encoded and encoded[0][1], "first packet must be a keyframe"

    for seq, (data, is_key) in enumerate(encoded):
        writer.write(
            CompressedColorFrame(
                width=64, height=48, seq=seq, stamp_ns=time.time_ns(),
                is_keyframe=is_key, frame_id="test", data=list(data),
            )
        )
    time.sleep(0.3)

    msg = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        msg = source.read()
        # Poll until the newest (bright) frame has made it through decode.
        if msg is not None and msg["images"]["ego_view"].mean() > 130:
            break
        time.sleep(0.05)

    assert msg is not None, "no decoded frame"
    img = msg["images"]["ego_view"]
    assert img.shape == (48, 64, 3)
    assert img.dtype == np.uint8
    # Newest frame wins: mean should be near the bright frame's value.
    assert img.mean() > 130, f"expected newest (bright) frame, mean={img.mean():.1f}"
    assert msg["timestamps"]["ego_view"] > 0

    source.close()


def test_camera_waits_for_keyframe():
    CompressedColorFrame = get_camera_types()
    participant = DomainParticipant(TEST_DOMAIN)
    writer = DataWriter(
        participant,
        Topic(participant, CAMERA_TOPIC + "/nokey", CompressedColorFrame),
        qos=Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(32)),
    )
    source = DdsCameraSource(domain_id=TEST_DOMAIN, topic=CAMERA_TOPIC + "/nokey")
    time.sleep(0.3)

    images = [np.full((48, 64, 3), i * 40, dtype=np.uint8) for i in range(5)]
    encoded = _encode_h264_frames(images)
    # Drop the leading keyframe: a mid-GOP joiner must decode nothing.
    for seq, (data, is_key) in enumerate(e for e in encoded if not e[1]):
        writer.write(
            CompressedColorFrame(
                width=64, height=48, seq=seq, stamp_ns=time.time_ns(),
                is_keyframe=False, frame_id="test", data=list(data),
            )
        )
    time.sleep(0.4)
    assert source.read() is None

    source.close()
