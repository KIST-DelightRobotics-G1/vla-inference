"""The kist_msgs wire contract: topics, IDL type mirrors, and QoS.

Mirrors the whole `module kist_msgs` of ``idl/kist_latent_action.idl`` —
LatentActionStep (the 50 Hz stream) and WbcCommand (reserved operator
channel) together, because they ship as one IDL module and one contract.
Python counterpart of gearsonic's ``include/vla/vla_latent_action.hpp`` —
the C++ side codegens from the same IDL (``idlc -l cxx``); this side is
hand-mirrored IdlStruct dataclasses (keep them in sync). Topic names ride
with the types because DDS endpoints
match on (topic name, type) together — same convention as ext-sensor-io's
``kCameraColorTopic`` living in ``h264_color_frame.hpp``.

QoS choices:

- latent actions: Reliable + KeepLast(1) — "latest value wins": a
  late-joining or slow reader sees the newest token, never a backlog.
  Reliable (not BestEffort) so the writer matches any reader reliability —
  see ``latent_action_qos``.
- commands: Reliable + KeepLast(8) — lifecycle commands must not drop.

``cyclonedds`` is imported lazily so the module stays importable without
the [dds] extra (topic constants, tests).
"""

from dataclasses import dataclass

# rt/kist/* naming follows the kist-ext-sensor-io convention; must match
# kist-gearsonic-inference include/vla/vla_latent_action.hpp.
LATENT_ACTION_TOPIC = "rt/kist/latent_action"
WBC_COMMAND_TOPIC = "rt/kist/wbc_command"

# Wire dimensions — the array sizes of the IDL struct below (gearsonic
# TokenEncoder::kTokenDim = 64; Dex3 = 7 motors per hand).
TOKEN_DIM = 64
HAND_DIM = 7

# The stream's tick period: gearsonic consumes one step per control tick
# (WholeBodyController::kControlDt = 0.02 s -> the 50 Hz in the IDL comment).
CONTROL_DT_NS = 20_000_000


def _idl_types():
    """Import cyclonedds lazily and build the IDL-mirroring types once."""
    from cyclonedds.idl import IdlStruct
    import cyclonedds.idl.types as t

    @dataclass
    class LatentActionStep(IdlStruct, typename="kist_msgs::LatentActionStep"):
        seq: t.uint64
        stamp_ns: t.int64
        frame_index: t.int64
        token_state: t.array[t.float32, TOKEN_DIM]
        left_hand_joints: t.array[t.float32, HAND_DIM]
        right_hand_joints: t.array[t.float32, HAND_DIM]

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
