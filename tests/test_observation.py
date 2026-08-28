"""Observation assembly: state-group layout, hand coupling, gravity, staleness.

Fake sources implement the io `latest*()` contract, so the builder is
tested against the exact seam the runner will wire.
"""

import numpy as np
import pytest

from common.g1_joints import JOINT_GROUP_INDICES
from vla.io.realsense import ColorFrame
from vla.io.unitree import IMU, HandState, UnitreeState
from vla.observation import Observation, ObservationBuilder
from vla.observation.gravity import compute_projected_gravity


class FakeCamera:
    def __init__(self, frame, age=0.01):
        self.frame, self.age = frame, age

    def latest(self):
        return self.frame, self.age


class FakeStateReader:
    def __init__(self, state, left, right, *, state_age=0.01, hand_age=0.01):
        self.state, self.left, self.right = state, left, right
        self.state_age, self.hand_age = state_age, hand_age

    def latest_state(self):
        return self.state, self.state_age

    def latest_left_hand(self):
        return self.left, self.hand_age

    def latest_right_hand(self):
        return self.right, self.hand_age


def make_frame(fill=7):
    return ColorFrame(rgb=np.full((4, 6, 3), fill, dtype=np.uint8), stamp_ns=123)


def make_state(q=None, quat=(1.0, 0.0, 0.0, 0.0)):
    q = np.arange(29, dtype=np.float64) / 100.0 if q is None else q
    imu = IMU(
        quaternion=np.array(quat, dtype=np.float64),
        gyroscope=np.zeros(3),
        accelerometer=np.zeros(3),
    )
    return UnitreeState(q=q, dq=np.zeros(29), tau=np.zeros(29), imu_pelvis=imu, tick=1, mode_machine=5)


def make_hand(base):
    return HandState(q=base + np.arange(7, dtype=np.float64) / 10.0)


def make_builder(**overrides):
    defaults = dict(
        cameras={"ego_view": FakeCamera(make_frame())},
        state_reader=FakeStateReader(make_state(), make_hand(1.0), make_hand(2.0)),
    )
    defaults.update(overrides)
    return ObservationBuilder(**defaults)


def test_builds_the_embodiment_groups(ateach=None):
    obs = make_builder().build("pick up the cup")

    assert isinstance(obs, Observation)
    assert obs.prompt == "pick up the cup"
    assert set(obs.video) == {"ego_view"}
    assert obs.video["ego_view"].shape == (4, 6, 3)
    expected = set(JOINT_GROUP_INDICES) | {"projected_gravity"}
    assert set(obs.state) == expected
    for group, idx in JOINT_GROUP_INDICES.items():
        assert obs.state[group].shape == (len(idx),)
        assert obs.state[group].dtype == np.float32


def test_body_joints_land_in_their_groups():
    obs = make_builder().build("p")
    # body_q[i] = i/100 in Unitree motor order: left leg 0-5, right leg 6-11,
    # waist 12-14, left arm 15-21, right arm 22-28.
    np.testing.assert_allclose(obs.state["left_leg"], np.arange(6) / 100.0, atol=1e-6)
    np.testing.assert_allclose(obs.state["waist"], np.arange(12, 15) / 100.0, atol=1e-6)
    np.testing.assert_allclose(obs.state["right_arm"], np.arange(22, 29) / 100.0, atol=1e-6)


def test_left_hand_coupling_copies_the_live_pair():
    obs = make_builder().build("p")
    # Left wire q = 1.0 + [0, .1, .2, .3, .4, .5, .6]; the coupling copies
    # wire slots [3:5] onto [5:7] before the full_q scatter, so the model's
    # left-hand group sees 1.3, 1.4 twice. Right hand is untouched.
    left = obs.state["left_hand"]
    right = obs.state["right_hand"]
    assert 1.5 not in np.round(left, 6) and 1.6 not in np.round(left, 6)
    assert np.isclose(sorted(left)[-1], 1.4) or 1.4 in np.round(left, 6)
    assert 2.5 in np.round(right, 6) and 2.6 in np.round(right, 6)


def test_projected_gravity_upright_and_rolled():
    upright = make_builder().build("p")
    np.testing.assert_allclose(upright.state["projected_gravity"], [0, 0, -1], atol=1e-6)

    # 90° roll about x (wxyz): gravity should appear along the body -y axis.
    s = np.sin(np.pi / 4)
    rolled = compute_projected_gravity(np.array([s, s, 0.0, 0.0]))
    np.testing.assert_allclose(rolled, [0, -1, 0], atol=1e-6)

    # Yaw invariance: a pure 90° yaw leaves gravity at [0, 0, -1].
    yawed = compute_projected_gravity(np.array([s, 0.0, 0.0, s]))
    np.testing.assert_allclose(yawed, [0, 0, -1], atol=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cameras": {"ego_view": FakeCamera(None, age=float("inf"))}},
        {"cameras": {"ego_view": FakeCamera(make_frame(), age=2.0)}},
        {"state_reader": FakeStateReader(None, make_hand(1.0), make_hand(2.0))},
        {"state_reader": FakeStateReader(make_state(), make_hand(1.0), make_hand(2.0), state_age=0.5)},
        {"state_reader": FakeStateReader(make_state(), None, make_hand(2.0))},
        {"state_reader": FakeStateReader(make_state(), make_hand(1.0), make_hand(2.0), hand_age=2.0)},
    ],
)
def test_missing_or_stale_stream_yields_none(overrides):
    assert make_builder(**overrides).build("p") is None


def test_every_configured_view_is_required():
    builder = make_builder(
        cameras={
            "ego_view": FakeCamera(make_frame()),
            "wrist_view": FakeCamera(None, age=float("inf")),
        }
    )
    assert builder.build("p") is None
