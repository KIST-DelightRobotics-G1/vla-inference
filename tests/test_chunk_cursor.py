"""Chunking stage: cursor swap/skip/hold/starve policies, live streamer Tx.

Pure-logic tests — the cursor has no threads or clocks of its own, so every
policy is asserted tick by tick; the streamer runs against an injected fake
writer (no DDS).
"""

import time

import numpy as np
import pytest

from vla.chunking import ChunkCursor, ChunkStep
from vla.policy import ActionChunk
from vla.publisher import LatentActionStreamer


def make_chunk(fill: float) -> ActionChunk:
    return ActionChunk(
        motion_token=np.full((40, 64), fill, dtype=np.float32),
        left_hand_joints=np.full((40, 7), fill, dtype=np.float32),
        right_hand_joints=np.full((40, 7), -fill, dtype=np.float32),
    )


def indexed_chunk() -> ActionChunk:
    # motion_token[t, 0] = t, so a step's source index is readable back
    tokens = np.zeros((40, 64), dtype=np.float32)
    tokens[:, 0] = np.arange(40)
    return ActionChunk(
        motion_token=tokens,
        left_hand_joints=np.zeros((40, 7), dtype=np.float32),
        right_hand_joints=np.zeros((40, 7), dtype=np.float32),
    )


def test_empty_cursor_starves():
    cursor = ChunkCursor()
    assert cursor.step() is None
    assert cursor.stats()["starved"] == 1


def test_steps_play_through_in_order():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk())
    played = [cursor.step().motion_token[0] for _ in range(40)]
    assert played == list(range(40))


def test_push_skips_the_in_flight_ticks():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk(), skip_ticks=7)
    assert cursor.step().motion_token[0] == 7
    assert cursor.stats()["skipped"] == 7


def test_fresh_push_replaces_mid_playback():
    cursor = ChunkCursor()
    cursor.push(make_chunk(1.0))
    for _ in range(10):
        cursor.step()
    cursor.push(make_chunk(2.0), skip_ticks=8)
    step = cursor.step()
    assert isinstance(step, ChunkStep)
    assert step.motion_token[0] == 2.0
    assert step.right_hand_joints[0] == -2.0


def test_exhausted_chunk_holds_then_goes_silent():
    cursor = ChunkCursor(hold_ticks=3)
    cursor.push(indexed_chunk())
    for _ in range(40):
        cursor.step()
    # bounded hold on the last step...
    for _ in range(3):
        assert cursor.step().motion_token[0] == 39
    # ...then silence, forever until a new push
    assert cursor.step() is None
    assert cursor.step() is None
    stats = cursor.stats()
    assert stats["held"] == 3 and stats["starved"] == 2


def test_push_revives_a_starved_stream():
    cursor = ChunkCursor(hold_ticks=0)
    cursor.push(indexed_chunk())
    for _ in range(40):
        cursor.step()
    assert cursor.step() is None
    cursor.push(make_chunk(5.0))
    assert cursor.step().motion_token[0] == 5.0


def test_entirely_stale_push_lands_on_the_last_step():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk(), skip_ticks=100)
    assert cursor.step().motion_token[0] == 39
    assert cursor.stats()["stale_pushes"] == 1


def test_negative_skip_is_refused():
    with pytest.raises(ValueError, match="skip_ticks"):
        ChunkCursor().push(indexed_chunk(), skip_ticks=-1)


class FakeWriter:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send_latent_action(self, *, motion_token, frame_index, left_hand_joints, right_hand_joints):
        self.sent.append((frame_index, float(motion_token[0])))

    def close(self):
        self.closed = True


def test_streamer_publishes_ticks_and_skips_silence():
    cursor = ChunkCursor(hold_ticks=0)
    cursor.push(indexed_chunk())
    writer = FakeWriter()
    streamer = LatentActionStreamer()
    streamer.start(cursor, domain_id=0, writer=writer)
    deadline = time.monotonic() + 3.0
    while len(writer.sent) < 40 and time.monotonic() < deadline:
        time.sleep(0.02)
    time.sleep(0.1)  # a few silent (starved) ticks after exhaustion
    streamer.stop()

    assert writer.closed
    assert [f for f, _ in writer.sent] == list(range(len(writer.sent)))  # 연속 frame_index
    assert [v for _, v in writer.sent][:40] == list(range(40))  # chunk 순서 그대로
    assert len(writer.sent) == 40  # 침묵 틱은 발행 안 됨
    assert cursor.stats()["starved"] > 0
