"""L1 loopback harness: the full runner over real ZMQ, no GPU/model/robot.

Spins up fake versions of every external process on ephemeral localhost
ports — camera server (JPEG frames), gearsonic state publisher (g1_debug),
operator keyboard — plus a deterministic fake policy, then runs the real
VLARunner and collects what it publishes on the action port.

Verifies end to end: observation assembly from real wire data, the
two-thread inference/publish machinery, latency-compensated chunk playback,
~50 Hz streaming, protocol v4 bytes, runtime prompt changes, and control
commands.
"""

import queue
import socket
import threading
import time

import cv2
import msgpack
import numpy as np
import pytest
import zmq

from common.config import IOConfig, RunnerConfig
from common.io.keyboard import KeyboardPublisher
from common.protocol import unpack_message
from vla.runner import VLARunner

HORIZON = 40
INFERENCE_SLEEP = 0.15  # fake model latency (s)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakePolicy:
    """Deterministic stand-in for the GR00T backend.

    Returns a chunk whose token value at step t is ``t * 0.01`` so the
    receiver can recover which step it is looking at.
    """

    def __init__(self):
        self.call_count = 0
        self.last_observation = None

    def get_action(self, observation):
        self.last_observation = observation
        self.call_count += 1
        time.sleep(INFERENCE_SLEEP)
        steps = np.arange(HORIZON, dtype=np.float32)[:, None] * 0.01
        action = {
            "action.motion_token": np.tile(steps, (1, 64))[None],       # (1, 40, 64)
            "action.left_hand_joints": np.tile(steps, (1, 7))[None],    # (1, 40, 7)
            "action.right_hand_joints": np.tile(steps, (1, 7))[None],
            "action.task_progress": np.zeros((1, HORIZON, 1), dtype=np.float32),
        }
        return action, {}

    def ping(self):
        return True

    def close(self):
        pass


class FakeSensorPublishers:
    """Camera + state publishers speaking the real wire formats."""

    def __init__(self, camera_port: int, state_port: int):
        self._ctx = zmq.Context()
        self._camera = self._ctx.socket(zmq.PUB)
        self._camera.bind(f"tcp://127.0.0.1:{camera_port}")
        self._state = self._ctx.socket(zmq.PUB)
        self._state.bind(f"tcp://127.0.0.1:{state_port}")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

        ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
        assert ok
        self._jpeg = jpeg.tobytes()

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            now = time.time()
            self._camera.send(
                msgpack.packb(
                    {"timestamps": {"ego_view": now}, "images": {"ego_view": self._jpeg}}
                )
            )
            state = {
                "body_q": [0.1] * 29,
                "left_hand_q": [0.0] * 7,
                "right_hand_q": [0.0] * 7,
                "base_quat": [1.0, 0.0, 0.0, 0.0],
            }
            self._state.send(b"g1_debug" + msgpack.packb(state))
            time.sleep(0.02)  # 50 Hz sensors

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._camera.close()
        self._state.close()
        self._ctx.term()


class ActionCollector:
    """SUB on the action port; parses pose/command messages as they arrive."""

    def __init__(self, port: int):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVTIMEO, 50)
        self._socket.connect(f"tcp://127.0.0.1:{port}")
        self.pose_msgs: queue.Queue = queue.Queue()
        self.command_msgs: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                raw = self._socket.recv()
            except zmq.Again:
                continue
            if raw.startswith(b"pose"):
                _, fields = unpack_message(raw, "pose")
                self.pose_msgs.put((time.monotonic(), fields))
            elif raw.startswith(b"command"):
                _, fields = unpack_message(raw, "command")
                self.command_msgs.put(fields)

    def drain_poses(self):
        out = []
        while True:
            try:
                out.append(self.pose_msgs.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close()
        self._ctx.term()


@pytest.fixture
def harness():
    ports = {
        "camera": _free_port(),
        "state": _free_port(),
        "action": _free_port(),
        "keyboard": _free_port(),
    }
    config = RunnerConfig(
        io=IOConfig(
            action_host="127.0.0.1",
            action_port=ports["action"],
            state_host="127.0.0.1",
            state_port=ports["state"],
            camera_host="127.0.0.1",
            camera_port=ports["camera"],
            keyboard_host="127.0.0.1",
            keyboard_port=ports["keyboard"],
        ),
        action_horizon=HORIZON,
        initial_pose_blend_duration=0.0,
    )

    sensors = FakeSensorPublishers(ports["camera"], ports["state"])
    sensors.start()
    collector = ActionCollector(ports["action"])
    collector.start()
    keyboard = KeyboardPublisher(port=ports["keyboard"], host="127.0.0.1")

    policy = FakePolicy()
    runner = VLARunner(config, policy=policy)
    stop = threading.Event()
    thread = threading.Thread(target=runner.run, args=(stop,), daemon=True)
    thread.start()

    yield runner, policy, collector, keyboard

    stop.set()
    thread.join(timeout=3.0)
    sensors.close()
    collector.close()
    keyboard.close()
    assert not thread.is_alive(), "runner failed to shut down"


def test_end_to_end_token_stream(harness):
    runner, policy, collector, _ = harness

    # Let sensors/first inference land, then unpause.
    time.sleep(0.6)
    runner.pause_loop = False

    stream_duration = 1.5
    time.sleep(stream_duration)
    poses = collector.drain_poses()

    # --- observation actually reached the policy with the right structure
    obs = policy.last_observation
    assert obs is not None
    assert obs["video"]["ego_view"].shape == (1, 1, 48, 64, 3)
    assert obs["video"]["ego_view"].dtype == np.uint8
    assert obs["state"]["left_arm"].shape == (1, 1, 7)
    assert obs["language"] == {"annotation.human.task_description": [["demo"]]}

    # --- a real stream came out at roughly the publish rate
    assert len(poses) >= 40, f"expected a ~50Hz stream, got {len(poses)} msgs"
    timestamps = [t for t, _ in poses]
    measured_rate = (len(poses) - 1) / (timestamps[-1] - timestamps[0])
    assert 35 < measured_rate < 65, f"measured publish rate {measured_rate:.1f} Hz"

    # --- protocol v4 content
    first = poses[0][1]
    assert first["token_state"].shape == (1, 64)
    assert first["left_hand_joints"].shape == (1, 7)
    assert first["right_hand_joints"].shape == (1, 7)

    # --- frame_index strictly increasing by 1
    frames = [int(f["frame_index"][0]) for _, f in poses]
    assert frames == list(range(frames[0], frames[0] + len(frames)))

    # --- latency compensation: the first published step skips ~the steps
    # that went stale during the fake 0.15s inference (7-8 at 50Hz).
    first_step_value = float(first["token_state"][0, 0])
    assert 0.04 <= first_step_value <= 0.20, (
        f"first chunk step {first_step_value / 0.01:.0f} not in latency-compensated range"
    )

    # --- inference kept retriggering at ~inference_rate
    assert policy.call_count >= 2


def test_prompt_change_and_commands(harness):
    runner, policy, collector, keyboard = harness
    time.sleep(0.4)  # let the keyboard SUB connect

    # Runtime prompt change (idempotent message — safe to repeat for slow joiners)
    for _ in range(5):
        keyboard.send_prompt("pick up the red cup")
        time.sleep(0.1)
        if runner.prompt == "pick up the red cup":
            break
    assert runner.prompt == "pick up the red cup"

    # A later observation must carry the new prompt
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        obs = policy.last_observation
        if obs and obs["language"]["annotation.human.task_description"] == [["pick up the red cup"]]:
            break
        time.sleep(0.05)
    else:
        pytest.fail("new prompt never reached the policy")

    # 'k' starts the gearsonic loop in PLANNER mode -> command message on the wire
    keyboard.send_key("k")
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            fields = collector.command_msgs.get(timeout=0.1)
            break
        except queue.Empty:
            continue
    else:
        pytest.fail("command message never arrived")
    assert fields["start"][0] == 1
    assert fields["planner"][0] == 1

    # The runner updates its own mode just after the send — poll briefly.
    deadline = time.monotonic() + 1.0
    while runner.cpp_mode != "PLANNER" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.cpp_mode == "PLANNER"
