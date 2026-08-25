"""DDS (CycloneDDS) implementation of ActionSink.

Publishes ``kist_msgs::LatentActionStep`` / ``kist_msgs::WbcCommand`` as
defined in ``idl/kist_latent_action.idl`` — the IDL file is the shared
contract with the gearsonic C++ side; the IdlStruct dataclasses here mirror
it and must be kept in sync.

QoS choices mirror the ZMQ semantics this replaces:

- latent actions: Reliable + KeepLast(1) — "latest value wins", like the
  CONFLATE PUB/SUB pair. A late-joining or slow reader sees the newest
  token, never a backlog. Reliable (not BestEffort) so the writer matches
  any reader reliability — see ``latent_action_qos``.
- commands: Reliable + KeepLast(8) — lifecycle commands must not drop.

``cyclonedds`` is imported lazily so the package works without it when the
ZMQ transport is selected.
"""

import time
from dataclasses import dataclass

import numpy as np

# rt/kist/* naming follows the kist-ext-sensor-io convention; must match
# kist-gearsonic-inference include/vla/vla_token_receiver.hpp.
LATENT_ACTION_TOPIC = "rt/kist/latent_action"
WBC_COMMAND_TOPIC = "rt/kist/wbc_command"


def _idl_types():
    """Import cyclonedds lazily and build the IDL-mirroring types once."""
    from cyclonedds.idl import IdlStruct
    import cyclonedds.idl.types as t

    @dataclass
    class LatentActionStep(IdlStruct, typename="kist_msgs::LatentActionStep"):
        seq: t.uint64
        stamp_ns: t.int64
        frame_index: t.int64
        token_state: t.array[t.float32, 64]
        left_hand_joints: t.array[t.float32, 7]
        right_hand_joints: t.array[t.float32, 7]

    @dataclass
    class WbcCommand(IdlStruct, typename="kist_msgs::WbcCommand"):
        seq: t.uint64
        stamp_ns: t.int64
        start: bool
        stop: bool
        planner: bool
        has_delta_heading: bool
        delta_heading: t.float32

    return LatentActionStep, WbcCommand


_types_cache = None


def get_dds_types():
    """Return (LatentActionStep, WbcCommand) IdlStruct classes (cached)."""
    global _types_cache
    if _types_cache is None:
        _types_cache = _idl_types()
    return _types_cache


def latent_action_qos():
    """Reliable + KeepLast(1): "latest wins" with universal reader matching.

    Reliable (not BestEffort) because DDS reliability must satisfy
    reader-requested <= writer-offered: the gearsonic side subscribes via
    unitree's ChannelSubscriber whose reader QoS we don't control — a
    Reliable writer matches both Reliable and BestEffort readers. KeepLast(1)
    keeps the latest-value semantics; max_blocking_time bounds write() if a
    reliable reader ever stalls.
    """
    from cyclonedds.core import Policy, Qos
    from cyclonedds.util import duration

    return Qos(
        Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=20)),
        Policy.History.KeepLast(1),
    )


def command_qos():
    from cyclonedds.core import Policy, Qos
    from cyclonedds.util import duration

    return Qos(
        Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=100)),
        Policy.History.KeepLast(8),
    )


class DdsActionSink:
    """CycloneDDS publisher toward the gearsonic whole-body controller."""

    def __init__(self, domain_id: int = 0):
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        LatentActionStep, WbcCommand = get_dds_types()
        self._LatentActionStep = LatentActionStep
        self._WbcCommand = WbcCommand

        self._participant = DomainParticipant(domain_id)
        self._action_writer = DataWriter(
            self._participant,
            Topic(self._participant, LATENT_ACTION_TOPIC, LatentActionStep),
            qos=latent_action_qos(),
        )
        self._command_writer = DataWriter(
            self._participant,
            Topic(self._participant, WBC_COMMAND_TOPIC, WbcCommand),
            qos=command_qos(),
        )
        self._action_seq = 0
        self._command_seq = 0
        print(f"[DdsActionSink] Publishing on domain {domain_id}: "
              f"{LATENT_ACTION_TOPIC}, {WBC_COMMAND_TOPIC}")

    def send_latent_action(
        self,
        motion_token: np.ndarray,
        frame_index: int,
        left_hand_joints: np.ndarray,
        right_hand_joints: np.ndarray,
    ) -> None:
        token = np.asarray(motion_token, dtype=np.float32).reshape(-1)
        left = np.asarray(left_hand_joints, dtype=np.float32).reshape(-1)
        right = np.asarray(right_hand_joints, dtype=np.float32).reshape(-1)
        if token.shape != (64,):
            raise ValueError(f"motion_token must have 64 values, got {token.shape}")
        if left.shape != (7,) or right.shape != (7,):
            raise ValueError("hand joints must have 7 values each")

        self._action_seq += 1
        self._action_writer.write(
            self._LatentActionStep(
                seq=self._action_seq,
                stamp_ns=time.time_ns(),
                frame_index=int(frame_index),
                token_state=token.tolist(),
                left_hand_joints=left.tolist(),
                right_hand_joints=right.tolist(),
            )
        )

    def send_command(self, start: bool, planner: bool = False) -> None:
        self._command_seq += 1
        self._command_writer.write(
            self._WbcCommand(
                seq=self._command_seq,
                stamp_ns=time.time_ns(),
                start=start,
                stop=not start,
                planner=planner,
                has_delta_heading=False,
                delta_heading=0.0,
            )
        )

    def close(self) -> None:
        # cyclonedds entities release their resources when garbage collected;
        # drop references deterministically.
        del self._action_writer
        del self._command_writer
        del self._participant
