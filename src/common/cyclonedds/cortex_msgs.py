"""The cortex_msgs wire contract: topics, IDL type mirrors, and QoS.

Mirrors ``idl/cortex_subtask.idl`` — the orchestrator <-> module subtask
protocol (v0.1 draft, ICD pending). Same shape as ``kist_msgs.py``: topic
names ride with the types, ``cyclonedds`` is imported lazily so the module
stays importable without the [dds] extra.

Topic names are spec-literal (``/cortex/vla/...``) — the orchestrator side
must open exactly these strings; DDS endpoints match on (topic name, type)
together.
"""

from dataclasses import dataclass
from enum import IntEnum

CORTEX_VLA_CMD_TOPIC = "/cortex/vla/cmd"
CORTEX_VLA_STATE_TOPIC = "/cortex/vla/state"

# SubtaskState publish rate (the ICD's "10 Hz 상시").
STATE_PERIOD_S = 0.1


class SubtaskStatus(IntEnum):
    IDLE = 0
    RUNNING = 1
    DONE = 2
    FAILED = 3


def _idl_types():
    """Import cyclonedds lazily and build the IDL-mirroring types once."""
    from cyclonedds.idl import IdlStruct
    import cyclonedds.idl.types as t

    @dataclass
    class SubtaskCmd(IdlStruct, typename="cortex_msgs::SubtaskCmd"):
        plan_id: str
        index: t.uint16
        action: str
        args: t.sequence[str]
        instruction: str
        cancel: bool

    @dataclass
    class SubtaskState(IdlStruct, typename="cortex_msgs::SubtaskState"):
        stamp_ns: t.int64
        plan_id: str
        index: t.uint16
        action: str
        status: t.uint8
        progress: t.float32
        detail: str

    return SubtaskCmd, SubtaskState


_types_cache = None


def get_cortex_types():
    """Return (SubtaskCmd, SubtaskState) IdlStruct classes (cached)."""
    global _types_cache
    if _types_cache is None:
        _types_cache = _idl_types()
    return _types_cache


def subtask_qos():
    """Reliable + KeepLast(10) — the ICD's QoS for both directions.

    Commands must not drop (a lost cancel is a safety problem), and depth 10
    lets a late-joining orchestrator see recent state history.
    """
    from cyclonedds.core import Policy, Qos
    from cyclonedds.util import duration

    return Qos(
        Policy.Reliability.Reliable(max_blocking_time=duration(milliseconds=100)),
        Policy.History.KeepLast(10),
    )
