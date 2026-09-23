"""SubtaskMachine — the VLA module's side of the cortex subtask protocol.

Pure logic (no threads, no clocks, no DDS — the ChunkCursor discipline):
the bridge feeds it commands and a monotonic `now`, the runner feeds it
probe verdicts via on_progress(), and it answers with the cursor effect
each transition requires and renders the 10 Hz SubtaskState fields on
demand.

The v0.2 state machine (DONE landed via the progress probe):

    IDLE ──valid cmd──▶ RUNNING ──cancel (matching id)──▶ IDLE   "cancelled"
                          │  │──new valid cmd─────────────▶ RUNNING (preempt)
                          │  │──probe: DONE───────────────▶ DONE ×3 → IDLE
                          │  │──probe: STALLED────────────▶ FAILED ×3 → IDLE
                          │  │──step timeout──────────────▶ FAILED ×3 → IDLE
                          │  └──unsupported cmd───────────▶ FAILED ×3 → IDLE
    FAILED (during ×3) ──any valid cmd──▶ RUNNING  (preempts the countdown)

DONE and FAILED are published for _FINAL_REPEATS bridge ticks (~300 ms) as
QoS edge-loss insurance, then auto-reset to IDLE per the ICD. A new valid
cmd during that window preempts the countdown immediately.

Appendix-A detail values: "" (nominal), "cancelled", "unsupported: <why>",
"timeout <s>s", "done (progress N.NN)", "stalled at N.NN". Deferred variants
(cancel_deferred / preempt_deferred / cancelled_at_safe_point) never occur:
chunk swap lands within ~0.5 s so cancel and preempt are always immediate.
"estop" is gearsonic's safety layer, not ours.

Every transition that stops acting (cancel, unsupported, timeout, DONE,
STALLED) returns Effect.FREEZE — the bridge pins the cursor so the robot
holds its last commanded posture instead of blending to safe standing.
"""

import threading
from dataclasses import dataclass
from enum import Enum

from common.cyclonedds.cortex_msgs import SubtaskStatus

from ..progress_probe import ProgressState

# Actions this module cannot perform — the ICD routes move_to to nav.
UNSUPPORTED_ACTIONS = frozenset({"move_to"})

# DONE/FAILED are re-published this many bridge ticks (~100 ms each) before
# the machine auto-resets to IDLE. Insurance against QoS edge-message loss.
_FINAL_REPEATS = 3


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
        """`step_timeout_s` <= 0 disables the timeout (tune after real-robot
        measurement, per the ICD note)."""
        self._lock = threading.Lock()
        self._timeout_s = step_timeout_s
        self._status = SubtaskStatus.IDLE
        self._plan_id = ""
        self._index = 0
        self._action = ""
        self._instruction = ""
        self._detail = ""
        self._progress = 0.0            # last probe verdict's progress, published
        self._deadline: float | None = None
        self._final_left = 0            # DONE/FAILED repeat countdown

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

    def on_progress(
        self, state: ProgressState, progress: float, now: float
    ) -> Effect:
        """Runner calls this after each inference with the probe's verdict.

        RUNNING → DONE (state=DONE) or FAILED (state=STALLED) with FREEZE.
        Anything else is a no-op (already terminal, or not our concern)."""
        with self._lock:
            if self._status is not SubtaskStatus.RUNNING:
                return Effect.NONE
            self._progress = progress
            if state is ProgressState.DONE:
                self._status = SubtaskStatus.DONE
                self._detail = f"done (progress {progress:.2f})"
                self._deadline = None
                self._final_left = _FINAL_REPEATS
                return Effect.FREEZE
            if state is ProgressState.STALLED:
                self._status = SubtaskStatus.FAILED
                self._detail = f"stalled at {progress:.2f}"
                self._deadline = None
                self._final_left = _FINAL_REPEATS
                return Effect.FREEZE
            return Effect.NONE

    def tick(self, now: float) -> Effect:
        """Time-based transitions — call once per state-publish cycle."""
        with self._lock:
            # DONE/FAILED countdown → auto IDLE reset after _FINAL_REPEATS ticks.
            if self._final_left > 0:
                self._final_left -= 1
                if self._final_left == 0:
                    self._reset_to_idle()
                return Effect.NONE
            # Step timeout — bridge's own clock, independent of orchestrator.
            if (
                self._status is SubtaskStatus.RUNNING
                and self._deadline is not None
                and now >= self._deadline
            ):
                self._status = SubtaskStatus.FAILED
                self._detail = f"timeout {self._timeout_s:g}s"
                self._deadline = None
                self._final_left = _FINAL_REPEATS
                return Effect.FREEZE
            return Effect.NONE

    def _on_cancel(self, plan_id: str, index: int) -> Effect:
        # Only the running (plan_id, index) is cancellable; a cancel for
        # anything else is stale (already replaced or never ours) — ignore.
        if self._status is not SubtaskStatus.RUNNING:
            return Effect.NONE
        if (plan_id, index) != (self._plan_id, self._index):
            return Effect.NONE
        # Immediate IDLE — chunk swap is fast enough that we don't defer.
        # No _final_left: IDLE with detail="cancelled" is itself stable.
        self._reset_to_idle(detail="cancelled")
        return Effect.FREEZE

    def _on_task(
        self, plan_id: str, index: int, action: str, instruction: str, now: float
    ) -> Effect:
        was_running = self._status is SubtaskStatus.RUNNING
        # A new valid cmd preempts even a DONE/FAILED countdown in progress.
        self._final_left = 0
        # The reported identity is the received one either way — FAILED must
        # name the cmd it rejects.
        self._plan_id, self._index, self._action = plan_id, index, action

        if action in UNSUPPORTED_ACTIONS or not instruction.strip():
            why = f"action '{action}'" if action in UNSUPPORTED_ACTIONS else "empty instruction"
            self._status = SubtaskStatus.FAILED
            self._detail = f"unsupported: {why}"
            self._deadline = None
            self._final_left = _FINAL_REPEATS
            # A running task was preempted by a broken cmd: stop acting.
            return Effect.FREEZE if was_running else Effect.NONE

        self._status = SubtaskStatus.RUNNING
        self._instruction = instruction
        self._detail = ""
        self._progress = 0.0
        self._deadline = now + self._timeout_s if self._timeout_s > 0 else None
        # Preempt needs no cursor effect: the inference loop picks up the new
        # instruction on its next build and the fresh chunk splices in.
        return Effect.NONE

    def _reset_to_idle(self, *, detail: str = "") -> None:
        """Wipe subtask state back to IDLE. detail keeps the last reason
        ('cancelled' after a cancel, '' after natural DONE/FAILED expiry)."""
        self._status = SubtaskStatus.IDLE
        self._plan_id = ""
        self._index = 0
        self._action = ""
        self._instruction = ""
        self._detail = detail
        self._progress = 0.0
        self._deadline = None
        self._final_left = 0

    # ── reads ─────────────────────────────────────────────────────────────────

    def instruction(self) -> str | None:
        """The prompt the inference loop should run, None unless RUNNING."""
        with self._lock:
            if self._status is SubtaskStatus.RUNNING:
                return self._instruction
            return None

    def subtask_id(self) -> tuple[str, int]:
        """(plan_id, index) of the RUNNING subtask, ('', 0) otherwise.

        Runner uses this to detect subtask changes and call monitor.reset() —
        the running max must not carry over between subtasks."""
        with self._lock:
            if self._status is SubtaskStatus.RUNNING:
                return (self._plan_id, self._index)
            return ("", 0)

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
                self._progress,
                self._detail,
            )
