"""SubtaskMachine — the VLA module's side of the cortex subtask protocol.

Pure logic (no threads, no clocks, no DDS — the ChunkCursor discipline):
the bridge feeds it commands and a monotonic `now`, it answers with the
cursor effect each transition requires and renders the 10 Hz SubtaskState
fields on demand.

The v0.1 state machine (DONE is deliberately absent — the internal
done-judgment is not implemented yet, so a subtask RUNs until the
orchestrator cancels or preempts it, or the step timeout fails it):

    IDLE ──valid cmd──▶ RUNNING ──cancel (matching id)──▶ IDLE  "cancelled"
                          │  │──new valid cmd────────────▶ RUNNING (preempt)
                          │  └──step timeout─────────────▶ FAILED "timeout <s>s"
                          └────unsupported cmd───────────▶ FAILED "unsupported: …"
    FAILED ──any valid cmd──▶ RUNNING

Appendix-A detail values this module emits: "" (nominal), "cancelled",
"unsupported: <why>", "timeout <s>s". The deferred variants
(cancel_deferred / preempt_deferred / cancelled_at_safe_point) never occur
here: a chunk swap lands within ~0.5 s, so cancel and preempt are always
immediate. "estop" is gearsonic's safety layer, not ours.

Every transition that stops acting (cancel, unsupported, timeout) returns
Effect.FREEZE — the bridge pins the cursor so the robot holds its last
commanded posture instead of blending to safe standing.
"""

import threading
from dataclasses import dataclass
from enum import Enum

from common.cyclonedds.cortex_msgs import SubtaskStatus

# Actions this module cannot perform — the ICD routes move_to to nav.
UNSUPPORTED_ACTIONS = frozenset({"move_to"})


class Effect(Enum):
    """What the transition asks the bridge to do to the token stream."""

    NONE = "none"
    FREEZE = "freeze"  # pin the cursor: hold the last commanded posture


@dataclass(frozen=True)
class StateFields:
    """One SubtaskState message, minus the stamp (the publisher's job)."""

    plan_id: str
    index: int
    action: str
    status: SubtaskStatus
    progress: float
    detail: str


class SubtaskMachine:
    """Thread-safe: the bridge's 10 Hz thread mutates, the inference loop
    reads `instruction()` — one lock covers both."""

    def __init__(self, *, step_timeout_s: float = 0.0):
        """`step_timeout_s` <= 0 disables the timeout (v0.1: tune after
        real-robot measurement, per the ICD note)."""
        self._lock = threading.Lock()
        self._timeout_s = step_timeout_s
        self._status = SubtaskStatus.IDLE
        self._plan_id = ""
        self._index = 0
        self._action = ""
        self._instruction = ""
        self._detail = ""
        self._deadline: float | None = None

    # ── transitions (bridge's Rx side) ────────────────────────────────────────

    def on_cmd(
        self,
        *,
        plan_id: str,
        index: int,
        action: str,
        instruction: str,
        cancel: bool,
        now: float,
    ) -> Effect:
        with self._lock:
            if cancel:
                return self._on_cancel(plan_id, index)
            return self._on_task(plan_id, index, action, instruction, now)

    def tick(self, now: float) -> Effect:
        """Time-based transitions — call once per state-publish cycle."""
        with self._lock:
            if (
                self._status is SubtaskStatus.RUNNING
                and self._deadline is not None
                and now >= self._deadline
            ):
                self._status = SubtaskStatus.FAILED
                self._detail = f"timeout {self._timeout_s:g}s"
                self._deadline = None
                return Effect.FREEZE
            return Effect.NONE

    def _on_cancel(self, plan_id: str, index: int) -> Effect:
        # Only the running (plan_id, index) is cancellable; a cancel for
        # anything else is stale (already replaced or never ours) — ignore.
        if self._status is not SubtaskStatus.RUNNING:
            return Effect.NONE
        if (plan_id, index) != (self._plan_id, self._index):
            return Effect.NONE
        self._status = SubtaskStatus.IDLE
        self._detail = "cancelled"
        self._deadline = None
        return Effect.FREEZE

    def _on_task(
        self, plan_id: str, index: int, action: str, instruction: str, now: float
    ) -> Effect:
        was_running = self._status is SubtaskStatus.RUNNING
        # The reported identity is the received one either way — FAILED must
        # name the cmd it rejects.
        self._plan_id, self._index, self._action = plan_id, index, action

        if action in UNSUPPORTED_ACTIONS or not instruction.strip():
            why = f"action '{action}'" if action in UNSUPPORTED_ACTIONS else "empty instruction"
            self._status = SubtaskStatus.FAILED
            self._detail = f"unsupported: {why}"
            self._deadline = None
            # A running task was preempted by a broken cmd: stop acting.
            return Effect.FREEZE if was_running else Effect.NONE

        self._status = SubtaskStatus.RUNNING
        self._instruction = instruction
        self._detail = ""
        self._deadline = now + self._timeout_s if self._timeout_s > 0 else None
        # Preempt needs no cursor effect: the inference loop picks up the new
        # instruction on its next build and the fresh chunk splices in.
        return Effect.NONE

    # ── reads ─────────────────────────────────────────────────────────────────

    def instruction(self) -> str | None:
        """The prompt the inference loop should run, None unless RUNNING."""
        with self._lock:
            if self._status is SubtaskStatus.RUNNING:
                return self._instruction
            return None

    def state_fields(self) -> StateFields:
        """The current SubtaskState — IDLE reports empty identity per the ICD."""
        with self._lock:
            if self._status is SubtaskStatus.IDLE:
                return StateFields("", 0, "", SubtaskStatus.IDLE, 0.0, self._detail)
            return StateFields(
                self._plan_id,
                self._index,
                self._action,
                self._status,
                0.0,  # progress estimator not landed yet (vla-inference-probe)
                self._detail,
            )
