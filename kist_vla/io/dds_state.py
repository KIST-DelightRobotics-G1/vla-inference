"""Robot-state source reading unitree DDS topics directly.

Subscribes to the same topics the gearsonic C++ side reads
(``rt/lowstate`` for the 29 body motors + pelvis IMU, ``rt/dex3/{left,right}/state``
for the Dex3-1 hands) and assembles the state dict the ObservationBuilder
expects — no gearsonic-side re-publisher needed.

Wire semantics verified against the reference stack
(``gear_sonic_deploy .../zmq_output_handler.hpp``):

- ``body_q``: 29 absolute joint angles in Unitree motor order (left leg,
  right leg, waist, left arm, right arm) — identical to LowState
  ``motor_state[0:29].q``.
- ``base_quat``: pelvis IMU quaternion (w, x, y, z) — LowState
  ``imu_state.quaternion`` (the torso IMU on ``rt/secondary_imu`` is NOT
  the base quaternion).
- ``left/right_hand_q``: Dex3 motor order, HandState ``motor_state[0:7].q``.

Readers are BestEffort + KeepLast(1) — compatible with any writer
reliability, latest value wins. ``get_msg`` returns a combined dict only
when a fresh LowState sample has arrived since the last call (matching the
ZMQ subscriber's "None until new data" contract); hand states use the
latest cached sample since they publish on independent clocks.
"""

from typing import Any

import numpy as np

LOWSTATE_TOPIC = "rt/lowstate"
LEFT_HAND_TOPIC = "rt/dex3/left/state"
RIGHT_HAND_TOPIC = "rt/dex3/right/state"

NUM_BODY_MOTORS = 29
NUM_HAND_MOTORS = 7


class DdsStateSource:
    """Latest-value robot state from unitree DDS topics."""

    def __init__(
        self,
        domain_id: int = 0,
        lowstate_topic: str = LOWSTATE_TOPIC,
        left_hand_topic: str = LEFT_HAND_TOPIC,
        right_hand_topic: str = RIGHT_HAND_TOPIC,
        require_hands: bool = True,
    ):
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_, LowState_

        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(1))
        self._participant = DomainParticipant(domain_id)
        self._lowstate_reader = DataReader(
            self._participant, Topic(self._participant, lowstate_topic, LowState_), qos=qos
        )
        self._left_hand_reader = DataReader(
            self._participant, Topic(self._participant, left_hand_topic, HandState_), qos=qos
        )
        self._right_hand_reader = DataReader(
            self._participant, Topic(self._participant, right_hand_topic, HandState_), qos=qos
        )

        self.require_hands = require_hands
        self._left_hand_q: np.ndarray | None = None
        self._right_hand_q: np.ndarray | None = None
        print(
            f"[DdsStateSource] domain {domain_id}: {lowstate_topic}, "
            f"{left_hand_topic}, {right_hand_topic} (require_hands={require_hands})"
        )

    @staticmethod
    def _take_latest(reader):
        samples = reader.take()
        return samples[-1] if samples else None

    def _poll_hands(self) -> None:
        left = self._take_latest(self._left_hand_reader)
        if left is not None:
            self._left_hand_q = np.array(
                [m.q for m in left.motor_state[:NUM_HAND_MOTORS]], dtype=np.float64
            )
        right = self._take_latest(self._right_hand_reader)
        if right is not None:
            self._right_hand_q = np.array(
                [m.q for m in right.motor_state[:NUM_HAND_MOTORS]], dtype=np.float64
            )

    def get_msg(self, clear: bool = True) -> dict[str, Any] | None:
        """Return the combined state dict, or None if no fresh LowState."""
        self._poll_hands()

        lowstate = self._take_latest(self._lowstate_reader)
        if lowstate is None:
            return None

        if self._left_hand_q is None or self._right_hand_q is None:
            if self.require_hands:
                return None
            left = self._left_hand_q if self._left_hand_q is not None else np.zeros(7)
            right = self._right_hand_q if self._right_hand_q is not None else np.zeros(7)
        else:
            left, right = self._left_hand_q, self._right_hand_q

        return {
            "body_q": np.array(
                [m.q for m in lowstate.motor_state[:NUM_BODY_MOTORS]], dtype=np.float64
            ),
            "left_hand_q": np.asarray(left, dtype=np.float64),
            "right_hand_q": np.asarray(right, dtype=np.float64),
            "base_quat": np.array(lowstate.imu_state.quaternion, dtype=np.float64),
        }

    def close(self) -> None:
        del self._lowstate_reader
        del self._left_hand_reader
        del self._right_hand_reader
        del self._participant
