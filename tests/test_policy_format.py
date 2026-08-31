"""Policy-stage format conversion: Observation -> gr00t dict -> ActionChunk.

Pure-numpy seams only (gr00t_format, action_chunk) — SonicPolicy itself
needs the inference container and is exercised by tests/smoke_policy.py.
"""

import numpy as np
import pytest

from vla.observation import Observation
from vla.policy import ActionChunk
from vla.policy.gr00t_format import to_action_chunk, to_gr00t_observation

PROMPT = "pick up the cup"


def make_observation():
    video = {
        view: np.full((480, 640, 3), fill, dtype=np.uint8)
        for fill, view in enumerate(["ego_view", "left_wrist", "right_wrist"])
    }
    state = {
        "left_leg": np.arange(6, dtype=np.float32),
        "projected_gravity": np.array([0, 0, -1], dtype=np.float32),
    }
    return Observation(video=video, state=state, prompt=PROMPT)


def test_gr00t_observation_shapes_and_types():
    obs = to_gr00t_observation(make_observation(), "annotation.human.task_description")

    assert set(obs) == {"video", "state", "language"}
    for view in ("ego_view", "left_wrist", "right_wrist"):
        frame = obs["video"][view]
        assert frame.shape == (1, 1, 480, 640, 3) and frame.dtype == np.uint8
    assert obs["state"]["left_leg"].shape == (1, 1, 6)
    assert obs["state"]["left_leg"].dtype == np.float32
    assert obs["language"] == {"annotation.human.task_description": [[PROMPT]]}


def test_gr00t_observation_preserves_values():
    obs = to_gr00t_observation(make_observation(), "task")
    np.testing.assert_array_equal(obs["state"]["left_leg"][0, 0], np.arange(6))
    assert obs["video"]["left_wrist"][0, 0, 0, 0, 0] == 1


def test_action_chunk_from_batched_dict():
    action = {
        "motion_token": np.zeros((1, 40, 64), dtype=np.float32),
        "left_hand_joints": np.ones((1, 40, 7), dtype=np.float32),
        "right_hand_joints": np.full((1, 40, 7), 2, dtype=np.float32),
    }
    chunk = to_action_chunk(action)

    assert isinstance(chunk, ActionChunk)
    assert chunk.motion_token.shape == (40, 64)
    np.testing.assert_array_equal(chunk.left_hand_joints, 1.0)
    np.testing.assert_array_equal(chunk.right_hand_joints, 2.0)


def test_action_chunk_rejects_wrong_shape():
    with pytest.raises(ValueError, match="motion_token"):
        ActionChunk(
            motion_token=np.zeros((40, 63), dtype=np.float32),
            left_hand_joints=np.zeros((40, 7), dtype=np.float32),
            right_hand_joints=np.zeros((40, 7), dtype=np.float32),
        )
