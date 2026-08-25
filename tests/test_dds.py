"""DDS action-sink loopback test (skipped when cyclonedds is not installed).

Publishes through DdsActionSink and reads back with a plain CycloneDDS
reader in the same process, pinning the IdlStruct field layout that the
gearsonic C++ side will codegen from idl/kist_latent_action.idl.
"""

import time

import numpy as np
import pytest

cyclonedds = pytest.importorskip("cyclonedds")

from cyclonedds.domain import DomainParticipant  # noqa: E402
from cyclonedds.sub import DataReader  # noqa: E402
from cyclonedds.topic import Topic  # noqa: E402

from common.io.dds import (  # noqa: E402
    LATENT_ACTION_TOPIC,
    WBC_COMMAND_TOPIC,
    DdsActionSink,
    command_qos,
    get_dds_types,
    latent_action_qos,
)

# Isolated domain so the test never crosses real robot traffic.
TEST_DOMAIN = 217


def _take_one(reader, deadline_s=5.0):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        samples = reader.take()
        if samples:
            return samples[-1]
        time.sleep(0.01)
    pytest.fail("no DDS sample arrived")


@pytest.fixture
def sink_and_readers():
    LatentActionStep, WbcCommand = get_dds_types()
    participant = DomainParticipant(TEST_DOMAIN)
    action_reader = DataReader(
        participant,
        Topic(participant, LATENT_ACTION_TOPIC, LatentActionStep),
        qos=latent_action_qos(),
    )
    command_reader = DataReader(
        participant,
        Topic(participant, WBC_COMMAND_TOPIC, WbcCommand),
        qos=command_qos(),
    )
    sink = DdsActionSink(domain_id=TEST_DOMAIN)
    time.sleep(0.3)  # discovery
    yield sink, action_reader, command_reader
    sink.close()


def test_latent_action_roundtrip(sink_and_readers):
    sink, action_reader, _ = sink_and_readers

    token = np.linspace(-1, 1, 64, dtype=np.float32)
    left = np.arange(7, dtype=np.float32)
    right = -np.arange(7, dtype=np.float32)
    sink.send_latent_action(token, frame_index=17, left_hand_joints=left, right_hand_joints=right)

    sample = _take_one(action_reader)
    assert sample.frame_index == 17
    assert sample.seq == 1
    assert sample.stamp_ns > 0
    np.testing.assert_allclose(np.array(sample.token_state, dtype=np.float32), token)
    np.testing.assert_allclose(np.array(sample.left_hand_joints, dtype=np.float32), left)
    np.testing.assert_allclose(np.array(sample.right_hand_joints, dtype=np.float32), right)


def test_latent_action_validates_shapes(sink_and_readers):
    sink, _, _ = sink_and_readers
    with pytest.raises(ValueError):
        sink.send_latent_action(
            np.zeros(63, dtype=np.float32), 0, np.zeros(7), np.zeros(7)
        )
    with pytest.raises(ValueError):
        sink.send_latent_action(
            np.zeros(64, dtype=np.float32), 0, np.zeros(6), np.zeros(7)
        )


def test_keep_last_semantics(sink_and_readers):
    """KeepLast(1): a slow reader sees only the newest token."""
    sink, action_reader, _ = sink_and_readers
    for i in range(10):
        sink.send_latent_action(
            np.full(64, i, dtype=np.float32), i, np.zeros(7), np.zeros(7)
        )
    time.sleep(0.2)
    sample = _take_one(action_reader)
    assert sample.frame_index == 9
    assert not action_reader.take(), "KeepLast(1) reader should hold a single sample"


def test_command_roundtrip(sink_and_readers):
    sink, _, command_reader = sink_and_readers

    sink.send_command(start=True, planner=True)
    sample = _take_one(command_reader)
    assert sample.start is True
    assert sample.stop is False
    assert sample.planner is True
    assert sample.has_delta_heading is False

    sink.send_command(start=False)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        samples = command_reader.take()
        if samples:
            sample = samples[-1]
            break
        time.sleep(0.01)
    assert sample.stop is True
