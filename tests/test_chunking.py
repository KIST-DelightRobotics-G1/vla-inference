"""Chunk playback, latency compensation, and inference-trigger tests."""

import numpy as np

from vla.chunking import (
    ActionChunkPlayer,
    calculate_latency_compensated_index,
    should_trigger_new_inference,
)


def _make_chunk(horizon=40, batch=True):
    token = np.tile(np.arange(horizon, dtype=np.float32)[:, None], (1, 64))
    left = np.tile(np.arange(horizon, dtype=np.float32)[:, None], (1, 7))
    right = left.copy()
    chunk = {"motion_token": token, "left_hand_joints": left, "right_hand_joints": right}
    if batch:
        chunk = {k: v[None] for k, v in chunk.items()}  # (1, T, D)
    return chunk


def test_latency_index():
    # 0.4s at 50Hz -> skip 20 steps
    assert calculate_latency_compensated_index(0.4, 50, 40) == 20
    assert calculate_latency_compensated_index(0.0, 50, 40) == 0
    # Clamped to the last step
    assert calculate_latency_compensated_index(10.0, 50, 40) == 39


def test_trigger_logic():
    # No chunk yet -> always trigger
    assert should_trigger_new_inference(False, True, 0.0, 0.4)
    # Worker busy -> never trigger
    assert not should_trigger_new_inference(True, True, 99.0, 0.4)
    # Idle + interval elapsed -> trigger
    assert should_trigger_new_inference(True, False, 0.5, 0.4)
    assert not should_trigger_new_inference(True, False, 0.3, 0.4)


def test_player_squeezes_batch_and_compensates_latency():
    player = ActionChunkPlayer(action_horizon=40)
    assert not player.has_chunk
    assert player.step() is None

    player.update(_make_chunk(), inference_delay=0.4, publish_rate=50)
    step = player.step()
    assert step["motion_token"].shape == (64,)
    # Started at index 20 due to latency compensation
    assert step["motion_token"][0] == 20.0
    assert player.step()["motion_token"][0] == 21.0


def test_player_holds_last_step_when_exhausted():
    player = ActionChunkPlayer(action_horizon=4)
    player.update(_make_chunk(horizon=4), inference_delay=0.0, publish_rate=50)
    values = [player.step()["motion_token"][0] for _ in range(6)]
    assert values == [0.0, 1.0, 2.0, 3.0, 3.0, 3.0]


def test_player_accepts_unbatched_chunk():
    player = ActionChunkPlayer(action_horizon=40)
    player.update(_make_chunk(batch=False), inference_delay=0.0, publish_rate=50)
    assert player.step()["motion_token"][0] == 0.0


def test_player_clear():
    player = ActionChunkPlayer(action_horizon=40)
    player.update(_make_chunk(), inference_delay=0.0, publish_rate=50)
    player.clear()
    assert not player.has_chunk
    assert player.step() is None
