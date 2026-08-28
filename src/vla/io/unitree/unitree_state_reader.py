"""UnitreeStateReader — unitree DDS state topics, per-stream latest snapshots.

The python twin of gearsonic's `UnitreeStateReader` (unitree_state_reader.cpp):
same topics, same converted structs, and the same per-stream exposure —
each stream lands in its own latest-value slot with its own age, so a
consumer judges freshness per stream (gearsonic does this with per-stream
DataBuffers plus a watchdog; here the exposed age carries that decision to
the consumer instead).

Deliberate differences from the C++ reader:

- Dex3 hand states are ADDED (rt/dex3/{left,right}/state): gearsonic only
  writes hand commands, but the VLA observation needs the measured hands.
- The torso IMU (rt/secondary_imu) is OMITTED: nothing in the observation
  consumes it — the pelvis IMU inside LowState is the base quaternion.
- LowState's CRC field is not checked (it covers the C struct's memory
  layout); intake validation is a non-finite check instead, so consumers
  never see NaN/Inf (the VlaTokenReceiver rule).

A small Rx thread polls the readers every ~2 ms and stamps each accepted
sample with its ARRIVAL time on this machine's monotonic clock — that is
what makes the exposed ages trustworthy for staleness decisions. (gearsonic
gets the same from the SDK's callback threads; polling on the consumer's
thread instead would report a sample as fresh whenever it was *read*, not
when it arrived — and DDS's source_timestamp is the publisher's clock,
which cross-machine skew can lie about.)
"""

import threading
import time

import numpy as np

from .hand_state import NUM_HAND_MOTORS, HandState
from .unitree_state import NUM_MOTORS, IMU, UnitreeState

LOWSTATE_TOPIC = "rt/lowstate"
LEFT_HAND_STATE_TOPIC = "rt/dex3/left/state"
RIGHT_HAND_STATE_TOPIC = "rt/dex3/right/state"


def _convert_imu(imu_state) -> IMU:
    return IMU(
        quaternion=np.array(imu_state.quaternion, dtype=np.float64),
        gyroscope=np.array(imu_state.gyroscope, dtype=np.float64),
        accelerometer=np.array(imu_state.accelerometer, dtype=np.float64),
    )


def _convert_lowstate(low_state) -> UnitreeState:
    motors = low_state.motor_state[:NUM_MOTORS]
    return UnitreeState(
        q=np.array([m.q for m in motors], dtype=np.float64),
        dq=np.array([m.dq for m in motors], dtype=np.float64),
        tau=np.array([m.tau_est for m in motors], dtype=np.float64),
        imu_pelvis=_convert_imu(low_state.imu_state),
        tick=int(low_state.tick),
        mode_machine=int(low_state.mode_machine),
    )


def _convert_hand(hand_state) -> HandState:
    return HandState(
        q=np.array([m.q for m in hand_state.motor_state[:NUM_HAND_MOTORS]], dtype=np.float64)
    )


class UnitreeStateReader:
    """Latest converted state of each unitree stream.

    Usage:
        reader = UnitreeStateReader()
        reader.start(domain_id=0)
        state, age_s = reader.latest_state()       # (None, inf) until data
        left, age_s = reader.latest_left_hand()
        right, age_s = reader.latest_right_hand()
        reader.stop()
    """

    def __init__(
        self,
        *,
        lowstate_topic: str = LOWSTATE_TOPIC,
        left_hand_topic: str = LEFT_HAND_STATE_TOPIC,
        right_hand_topic: str = RIGHT_HAND_STATE_TOPIC,
    ):
        self._topics = (lowstate_topic, left_hand_topic, right_hand_topic)
        self._participant = None
        self._readers = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-stream latest-value slots: (converted, monotonic recv time).
        self._state: tuple[UnitreeState, float] | None = None
        self._left: tuple[HandState, float] | None = None
        self._right: tuple[HandState, float] | None = None

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
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._rx_loop, name="unitree-rx", daemon=True
        )
        self._thread.start()
        print(
            f"[UnitreeStateReader] domain {domain_id}: {lowstate_topic}, "
            f"{left_topic}, {right_topic}"
        )

    # ── per-stream latest snapshots ───────────────────────────────────────────

    def latest_state(self) -> tuple[UnitreeState | None, float]:
        """The newest converted LowState and its arrival age in seconds."""
        return self._age_of(self._state)

    def latest_left_hand(self) -> tuple[HandState | None, float]:
        return self._age_of(self._left)

    def latest_right_hand(self) -> tuple[HandState | None, float]:
        return self._age_of(self._right)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._readers = None
        self._participant = None

    # ── Rx thread ─────────────────────────────────────────────────────────────

    def _rx_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll()
            except Exception as e:
                # A malformed sample must not kill the Rx thread — all three
                # streams' ages would silently grow stale. Log and keep going.
                print(f"[UnitreeStateReader] poll error ({e}); continuing")
            self._stop_event.wait(0.002)

    @staticmethod
    def _age_of(slot):
        if slot is None:
            return None, float("inf")
        data, received = slot
        return data, time.monotonic() - received

    @staticmethod
    def _take_latest(reader):
        samples = reader.take()
        return samples[-1] if samples else None

    def _poll(self) -> None:
        lowstate_reader, left_reader, right_reader = self._readers

        lowstate = self._take_latest(lowstate_reader)
        if lowstate is not None:
            state = _convert_lowstate(lowstate)
            if np.isfinite(state.q).all() and np.isfinite(state.imu_pelvis.quaternion).all():
                self._state = (state, time.monotonic())

        for reader, attr in ((left_reader, "_left"), (right_reader, "_right")):
            sample = self._take_latest(reader)
            if sample is not None:
                hand = _convert_hand(sample)
                if np.isfinite(hand.q).all():
                    setattr(self, attr, (hand, time.monotonic()))
