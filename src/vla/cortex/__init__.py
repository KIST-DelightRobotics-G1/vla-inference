"""Cortex stage: the orchestrator's subtask protocol, module (VLA) side.

The seam where task instructions arrive at runtime instead of a fixed
--prompt: the orchestrator publishes SubtaskCmd on /cortex/vla/cmd, this
stage turns it into "which instruction should the inference loop run right
now" and reports SubtaskState back at 10 Hz (idl/cortex_subtask.idl is the
contract).

    subtask_machine  SubtaskMachine — the protocol's state machine,
                     pure logic (Effect out, StateFields on demand)
    bridge           CortexBridge — DDS Rx/Tx plumbing + the 10 Hz thread,
                     applies cursor effects (freeze = hold last posture)
"""

from .bridge import CortexBridge
from .subtask_machine import Effect, StateFields, SubtaskMachine

__all__ = ["CortexBridge", "Effect", "StateFields", "SubtaskMachine"]
