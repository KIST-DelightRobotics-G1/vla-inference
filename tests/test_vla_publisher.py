"""VLA LatentActionPublisher: tick-driven Tx thread over an injected writer."""

import threading
import time

import numpy as np
import pytest

from vla.latent_action_publisher import LatentActionPublisher


class FakeWriter:
    def __init__(self):
        self.actions = []
        self.commands = []
        self.closed = False

    def send_latent_action(self, motion_token, frame_index, left_hand_joints, right_hand_joints):
        self.actions.append(int(frame_index))

    def send_command(self, start, planner=False):
        self.commands.append((start, planner))

    def close(self):
        self.closed = True


def test_tick_runs_on_tx_thread_at_rate():
    writer = FakeWriter()
    ticks = []
    pub = LatentActionPublisher()

    def tick():
        ticks.append(threading.current_thread().name)
        pub.send(
            motion_token=np.zeros(64, dtype=np.float32),
            frame_index=len(ticks),
            left_hand_joints=np.zeros(7, dtype=np.float32),
            right_hand_joints=np.zeros(7, dtype=np.float32),
        )

    pub.start(tick, rate_hz=100, domain_id=0, writer=writer)
    time.sleep(0.25)
    pub.stop()

    # ~25 ticks at 100 Hz in 0.25 s, every one on the Tx thread.
    assert 15 <= len(ticks) <= 35
    assert set(ticks) == {"latent-action-tx"}
    assert len(writer.actions) == len(ticks)
    assert writer.closed


def test_stop_interrupts_promptly_and_commands_pass_through():
    writer = FakeWriter()
    pub = LatentActionPublisher()
    pub.start(lambda: None, rate_hz=2, domain_id=0, writer=writer)  # 500 ms period
    pub.send_command(start=True, planner=True)

    t0 = time.monotonic()
    pub.stop()  # must not wait out the 500 ms sleep
    assert time.monotonic() - t0 < 0.5
    assert writer.commands == [(True, True)]
    assert writer.closed


def test_tick_exception_does_not_kill_the_loop():
    writer = FakeWriter()
    calls = []

    def tick():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    pub = LatentActionPublisher()
    pub.start(tick, rate_hz=100, domain_id=0, writer=writer)
    time.sleep(0.1)
    pub.stop()
    assert len(calls) > 1  # survived the first-tick exception


def test_start_twice_is_refused():
    pub = LatentActionPublisher()
    pub.start(lambda: None, rate_hz=100, domain_id=0, writer=FakeWriter())
    try:
        with pytest.raises(RuntimeError, match="already started"):
            pub.start(lambda: None, rate_hz=100, domain_id=0, writer=FakeWriter())
    finally:
        pub.stop()
