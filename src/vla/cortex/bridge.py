"""CortexBridge — the DDS plumbing around SubtaskMachine.

Owns one thread and one participant, self-contained like the streamer's Tx
side: a SubtaskCmd reader on /cortex/vla/cmd and a SubtaskState writer on
/cortex/vla/state, both RELIABLE + KeepLast(10) per the ICD. The 10 Hz loop
does three things in order — drain commands into the machine, run its timer
tick, publish the current state — and applies each transition's cursor
effect (FREEZE pins the last posture; the next subtask's first push
unpins).

The inference loop never touches DDS here: it reads
``bridge.instruction()`` (the machine's lock makes that safe) and runs or
idles accordingly.

`reader`/`writer` injection (tests / manual) skips DDS entity creation —
the injected pair only needs ``take()`` and ``write(sample)``.
"""

import threading
import time

from common.cyclonedds.cortex_msgs import (
    CORTEX_VLA_CMD_TOPIC,
    CORTEX_VLA_STATE_TOPIC,
    STATE_PERIOD_S,
    get_cortex_types,
    subtask_qos,
)

from ..chunking import ChunkCursor
from .subtask_machine import Effect, SubtaskMachine


class CortexBridge:
    def __init__(self, machine: SubtaskMachine, cursor: ChunkCursor):
        self._machine = machine
        self._cursor = cursor
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._reader = None
        self._writer = None
        self._participant = None
        self._state_type = None

    def start(self, *, domain_id: int, reader=None, writer=None) -> None:
        if self._thread is not None:
            raise RuntimeError("bridge already started")

        if reader is None or writer is None:
            from cyclonedds.domain import DomainParticipant
            from cyclonedds.pub import DataWriter
            from cyclonedds.sub import DataReader
            from cyclonedds.topic import Topic

            SubtaskCmd, SubtaskState = get_cortex_types()
            self._state_type = SubtaskState
            self._participant = DomainParticipant(domain_id)
            qos = subtask_qos()
            reader = DataReader(
                self._participant,
                Topic(self._participant, CORTEX_VLA_CMD_TOPIC, SubtaskCmd),
                qos=qos,
            )
            writer = DataWriter(
                self._participant,
                Topic(self._participant, CORTEX_VLA_STATE_TOPIC, SubtaskState),
                qos=qos,
            )
            print(
                f"[CortexBridge] {CORTEX_VLA_CMD_TOPIC} -> machine, "
                f"{CORTEX_VLA_STATE_TOPIC} @ 10 Hz on domain {domain_id}"
            )
        else:
            SubtaskCmd, SubtaskState = get_cortex_types()
            self._state_type = SubtaskState
        self._reader = reader
        self._writer = writer

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="cortex-bridge", daemon=True)
        self._thread.start()

    def instruction(self) -> str | None:
        """The inference loop's read: the RUNNING instruction, else None."""
        return self._machine.instruction()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._reader = None
        self._writer = None
        self._participant = None

    # ── 10 Hz thread ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._cycle()
            except Exception as e:
                # Same rule as the Rx sources: one malformed sample must not
                # kill the thread — the orchestrator would see us go stale.
                print(f"[CortexBridge] cycle error ({e}); continuing")
            self._stop_event.wait(STATE_PERIOD_S)

    def _cycle(self) -> None:
        now = time.monotonic()
        for cmd in self._reader.take():
            if not hasattr(cmd, "plan_id"):
                # cyclonedds delivers dataless InvalidSample markers for
                # instance-state changes (e.g. a cmd writer disconnecting) —
                # protocol noise, not commands.
                continue
            effect = self._machine.on_cmd(
                plan_id=cmd.plan_id,
                index=int(cmd.index),
                action=cmd.action,
                instruction=cmd.instruction,
                cancel=bool(cmd.cancel),
                now=now,
            )
            self._apply(effect)
            kind = "cancel" if cmd.cancel else cmd.action
            print(f"[CortexBridge] cmd {kind} ({cmd.plan_id}#{cmd.index})")
        self._apply(self._machine.tick(now))

        f = self._machine.state_fields()
        self._writer.write(
            self._state_type(
                stamp_ns=time.time_ns(),
                plan_id=f.plan_id,
                index=f.index,
                action=f.action,
                status=int(f.status),
                progress=f.progress,
                detail=f.detail,
            )
        )

    def _apply(self, effect: Effect) -> None:
        if effect is Effect.FREEZE:
            self._cursor.freeze()
