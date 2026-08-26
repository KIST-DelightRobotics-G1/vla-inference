"""LatentActionPublisher: Tx-thread lifecycle over an injected fake writer."""

import threading
import time

import numpy as np

from replay.publisher import LatentActionPublisher, publish_via
from replay.builder import ActionStream


class FakeWriter:
    def __init__(self):
        self.steps = []
        self.closed = False

    def send_latent_action(self, motion_token, frame_index, left_hand_joints, right_hand_joints):
        self.steps.append((int(frame_index), np.asarray(motion_token).copy()))

    def close(self):
        self.closed = True


def make_stream(ticks: int) -> ActionStream:
    tokens = np.tile(np.arange(ticks, dtype=np.float32).reshape(-1, 1), (1, 64))
    hands = np.zeros((ticks, 7), dtype=np.float32)
    return ActionStream(tokens=tokens, left_hand=hands, right_hand=hands)


def test_publish_via_sends_every_tick_in_order():
    writer = FakeWriter()
    published = publish_via(writer, make_stream(10))

    assert published == 10
    assert [i for i, _ in writer.steps] == list(range(10))
    assert writer.steps[7][1][0] == 7.0


def test_publish_via_holds_the_50hz_schedule():
    writer = FakeWriter()
    t0 = time.monotonic()
    publish_via(writer, make_stream(25))  # 0.5 s nominal
    elapsed = time.monotonic() - t0
    assert 0.45 <= elapsed <= 0.75  # absolute-deadline pacing, some slack


def test_publisher_owns_thread_and_closes_writer():
    writer = FakeWriter()
    pub = LatentActionPublisher()
    pub.start(make_stream(5), domain_id=0, writer=writer)
    # start() returns immediately; the Tx worker does the publishing.
    assert threading.current_thread().name != "latent-action-tx"
    pub.wait()
    pub.stop()

    assert len(writer.steps) == 5
    assert writer.closed


def test_stop_ends_the_stream_early():
    writer = FakeWriter()
    pub = LatentActionPublisher()
    pub.start(make_stream(500), domain_id=0, writer=writer)  # 10 s if left alone
    time.sleep(0.3)
    pub.stop()

    assert 0 < len(writer.steps) < 500
    assert writer.closed


def test_start_twice_is_refused():
    import pytest

    pub = LatentActionPublisher()
    pub.start(make_stream(200), domain_id=0, writer=FakeWriter())
    try:
        with pytest.raises(RuntimeError, match="already started"):
            pub.start(make_stream(5), domain_id=0, writer=FakeWriter())
    finally:
        pub.stop()
