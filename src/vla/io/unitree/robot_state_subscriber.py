"""RobotStateSubscriber — unitree DDS topics -> latest RobotState, poll-based.

No thread: the DDS reader's KeepLast(1) cache already keeps the newest
sample (the same transport-level latest-value the reference stack got from
ZMQ CONFLATE), and the conversion is trivial, so `latest()` polls the
readers on the caller's thread. Same consumer contract as the camera
subscriber — `(snapshot | None, age_s)` — the threading is an internal
detail of each source.

Intake validation (gearsonic VlaTokenReceiver rule): a sample with any
non-finite value is dropped whole, so consumers never see NaN/Inf.
"""

import time

import numpy as np

from .robot_state import RobotState

LOWSTATE_TOPIC = "rt/lowstate"
LEFT_HAND_TOPIC = "rt/dex3/left/state"
RIGHT_HAND_TOPIC = "rt/dex3/right/state"

NUM_BODY_MOTORS = 29
NUM_HAND_MOTORS = 7


class RobotStateSubscriber:
    """Latest combined robot state from rt/lowstate + rt/dex3/{left,right}/state.

    Usage:
        sub = RobotStateSubscriber()
        sub.start(domain_id=0)
        state, age_s = sub.latest()   # (None, inf) until every stream arrived
        sub.stop()

    The age is measured from the newest accepted LowState sample (the
    500 Hz stream that dominates freshness); hands ride along with their
    latest cached values, on their own clocks.
    """

    def __init__(
        self,
        *,
        lowstate_topic: str = LOWSTATE_TOPIC,
        left_hand_topic: str = LEFT_HAND_TOPIC,
        right_hand_topic: str = RIGHT_HAND_TOPIC,
        require_hands: bool = True,
    ):
        self._topics = (lowstate_topic, left_hand_topic, right_hand_topic)
        self.require_hands = require_hands
        self._participant = None
        self._readers = None
        self._body_q = None
        self._base_quat = None
        self._left_hand_q = None
        self._right_hand_q = None
        self._lowstate_received_at: float | None = None

    def start(self, *, domain_id: int) -> None:
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self._participant = DomainParticipant(domain_id)
        lowstate_topic, left_topic, right_topic = self._topics
        self._readers = tuple(
            DataReader(self._participant, Topic(self._participant, name, kind), qos=qos)
            for name, kind in (
                (lowstate_topic, LowState_),
                (left_topic, HandState_),
                (right_topic, HandState_),
            )
        )
        print(
            f"[RobotStateSubscriber] domain {domain_id}: {lowstate_topic}, "
            f"{left_topic}, {right_topic} (require_hands={self.require_hands})"
        )

    def latest(self) -> tuple[RobotState | None, float]:
        """The newest combined state and the age of its LowState in seconds
        ((None, inf) until every required stream has arrived)."""
        self._poll()

        if self._body_q is None:
            return None, float("inf")
        left, right = self._left_hand_q, self._right_hand_q
        if left is None or right is None:
            if self.require_hands:
                return None, float("inf")
            left = left if left is not None else np.zeros(NUM_HAND_MOTORS)
            right = right if right is not None else np.zeros(NUM_HAND_MOTORS)

        return (
            RobotState(
                body_q=self._body_q,
                left_hand_q=left,
                right_hand_q=right,
                base_quat=self._base_quat,
            ),
            time.monotonic() - self._lowstate_received_at,
        )

    def stop(self) -> None:
        self._readers = None
        self._participant = None

    # ── polling ───────────────────────────────────────────────────────────────

    @staticmethod
    def _take_latest(reader):
        samples = reader.take()
        return samples[-1] if samples else None

    @staticmethod
    def _finite(*arrays: np.ndarray) -> bool:
        return all(np.isfinite(a).all() for a in arrays)

    def _poll(self) -> None:
        lowstate_reader, left_reader, right_reader = self._readers

        lowstate = self._take_latest(lowstate_reader)
        if lowstate is not None:
            body_q = np.array(
                [m.q for m in lowstate.motor_state[:NUM_BODY_MOTORS]], dtype=np.float64
            )
            base_quat = np.array(lowstate.imu_state.quaternion, dtype=np.float64)
            if self._finite(body_q, base_quat):
                self._body_q, self._base_quat = body_q, base_quat
                self._lowstate_received_at = time.monotonic()

        for reader, attr in ((left_reader, "_left_hand_q"), (right_reader, "_right_hand_q")):
            sample = self._take_latest(reader)
            if sample is not None:
                q = np.array([m.q for m in sample.motor_state[:NUM_HAND_MOTORS]], dtype=np.float64)
                if self._finite(q):
                    setattr(self, attr, q)
