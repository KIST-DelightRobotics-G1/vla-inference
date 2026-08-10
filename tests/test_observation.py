"""Observation assembly tests.

Shapes/dtypes mirror ``Gr00tPolicy.check_observation``'s strict validation
(uint8 (B,T,H,W,C) video, float32 (B,T,D) states, [[str]] language) so a
regression here fails fast instead of inside the policy server.
"""

import numpy as np
import pytest

from kist_vla.observation import ObservationBuilder


def _camera_msg(with_wrists=False):
    images = {"ego_view": np.zeros((480, 640, 3), dtype=np.uint8)}
    if with_wrists:
        images["left_wrist"] = np.zeros((240, 320, 3), dtype=np.uint8)
        images["right_wrist"] = np.zeros((240, 320, 3), dtype=np.uint8)
    return {"timestamps": {k: 0.0 for k in images}, "images": images}


def _state_msg():
    return {
        "body_q": np.arange(29, dtype=np.float64),
        "left_hand_q": np.arange(7, dtype=np.float64),
        "right_hand_q": np.arange(7, dtype=np.float64),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),  # identity, wxyz
    }


def test_returns_none_when_sensors_missing():
    builder = ObservationBuilder()
    assert builder.build(None, _state_msg(), "demo") is None
    assert builder.build(_camera_msg(), None, "demo") is None
    incomplete = _state_msg()
    del incomplete["base_quat"]
    assert builder.build(_camera_msg(), incomplete, "demo") is None


def test_observation_shapes_and_dtypes():
    builder = ObservationBuilder()
    obs = builder.build(_camera_msg(), _state_msg(), "pick up the cup")
    assert obs is not None

    video = obs["video"]["ego_view"]
    assert video.shape == (1, 1, 480, 640, 3)
    assert video.dtype == np.uint8

    expected_dims = {
        "left_leg": 6, "right_leg": 6, "waist": 3,
        "left_arm": 7, "right_arm": 7,
        "left_hand": 7, "right_hand": 7,
        "projected_gravity": 3,
    }
    assert set(obs["state"].keys()) == set(expected_dims)
    for key, dim in expected_dims.items():
        arr = obs["state"][key]
        assert arr.shape == (1, 1, dim), key
        assert arr.dtype == np.float32, key

    assert obs["language"] == {"annotation.human.task_description": [["pick up the cup"]]}


def test_identity_quat_gravity():
    builder = ObservationBuilder()
    obs = builder.build(_camera_msg(), _state_msg(), "demo")
    np.testing.assert_allclose(
        obs["state"]["projected_gravity"][0, 0], [0.0, 0.0, -1.0], atol=1e-6
    )


def test_wrist_views_mapped():
    builder = ObservationBuilder()
    obs = builder.build(_camera_msg(with_wrists=True), _state_msg(), "demo")
    assert "left_wrist" in obs["video"]
    assert "wrist_view" in obs["video"]  # right_wrist -> wrist_view
    assert obs["video"]["wrist_view"].shape == (1, 1, 240, 320, 3)


def test_custom_language_key():
    builder = ObservationBuilder(language_key="task")
    obs = builder.build(_camera_msg(), _state_msg(), "demo")
    assert obs["language"] == {"task": [["demo"]]}


def test_hand_coupling_applied_to_left_hand_only():
    builder = ObservationBuilder()
    state = _state_msg()
    # left_hand_q motor order: [thumb0, thumb1, thumb2, index0, index1, mid0, mid1]
    state["left_hand_q"] = np.array([0, 0, 0, 0.4, 0.5, 99.0, 99.0])
    state["right_hand_q"] = np.array([0, 0, 0, 0.4, 0.5, 99.0, 99.0])
    obs = builder.build(_camera_msg(), state, "demo")

    # left_hand state (URDF order: index0, index1, mid0, mid1, thumb×3):
    # middle values replaced by index readings
    np.testing.assert_allclose(
        obs["state"]["left_hand"][0, 0], [0.4, 0.5, 0.4, 0.5, 0, 0, 0]
    )
    # right hand untouched
    np.testing.assert_allclose(
        obs["state"]["right_hand"][0, 0], [0.4, 0.5, 99.0, 99.0, 0, 0, 0]
    )


def test_rejects_bad_quat_shape():
    builder = ObservationBuilder()
    state = _state_msg()
    state["base_quat"] = np.zeros(3)
    with pytest.raises(AssertionError):
        builder.build(_camera_msg(), state, "demo")
