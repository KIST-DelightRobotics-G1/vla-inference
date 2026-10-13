"""SubtaskMachine — the VLA module's side of the cortex subtask protocol.

Pure logic (no threads, no clocks, no DDS — the ChunkCursor discipline):
the bridge feeds it commands and a monotonic `now`, the runner feeds it
probe verdicts via on_progress(), and it answers with the cursor effect
each transition requires and renders the 10 Hz SubtaskState fields on
demand.

The v0.3 state machine (a RUNNING subtask is never replaced, only ended):

    IDLE ──valid cmd──▶ RUNNING ──cancel (matching id)──▶ IDLE   "cancelled"
                          │  │──probe: DONE───────────────▶ DONE ×3 → IDLE
                          │  │──probe: STALLED────────────▶ FAILED ×3 → IDLE
                          │  └──step timeout──────────────▶ FAILED ×3 → IDLE
                          │     (unscored: timeout ───────▶ DONE ×3 → IDLE)
                          │
                          └── any other cmd while RUNNING: IGNORED (no change)
    IDLE ──unsupported cmd──▶ FAILED ×3 → IDLE      "unsupported: <why>"
    DONE/FAILED (during ×3) ──valid cmd──▶ RUNNING  (the task is over; the
                                                     countdown is cut short)

A new prompt therefore always goes cancel → IDLE → cmd: the robot stops
(FREEZE) between subtasks and nothing is ever preempted mid-motion. A cmd
that arrives while RUNNING is dropped on the floor — the orchestrator sees
its (plan_id, index) never appear in SubtaskState; the bridge logs it.

DONE and FAILED are published for _FINAL_REPEATS bridge ticks (~300 ms) as
QoS edge-loss insurance, then auto-reset to IDLE per the ICD.

Scored or unscored is decided per subtask from its instruction, at cmd
time: `scorer(instruction, probe_override)` (the runner passes the probe
bank's lookup — is there a head fitted on this exact sentence?). Scored:
the probe's verdict ends it, the step timeout is the FAILED backstop as
above. Unscored (no head, e.g. approach / step_back): no verdict is taken
and the unscored timeout ends it as DONE, "done (timeout <s>s)" — the
sentence's own number in probes.json (`unscored_timeouts`), else the
runner's --unscored-timeout-s; 0 = no timeout, it runs until cancel. No scorer = every subtask scored
(the --cortex-without---probe run: verdicts never come, timeout = FAILED).

Appendix-A detail values: "" (nominal), "cancelled", "unsupported: <why>",
"timeout <s>s", "done (progress N.NN)", "stalled at N.NN",
"done (timeout <s>s)". Deferred variants
(cancel_deferred / preempt_deferred / cancelled_at_safe_point) never occur:
cancel is immediate and preemption does not exist. "estop" is gearsonic's
safety layer, not ours.

Every transition that stops acting (cancel, unsupported, timeout, DONE,
STALLED) returns Effect.FREEZE — the bridge pins the cursor so the robot
holds its last commanded posture instead of blending to safe standing.
"""

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping

from common.cyclonedds.cortex_msgs import SubtaskStatus

from ..progress_probe import ProgressState, override_from_args

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

    def __init__(
        self,
        *,
        step_timeout_s: float = 0.0,
        scorer: Callable[[str, str | None], bool] | None = None,
        unscored_timeout_s: float = 0.0,
        unscored_timeouts: Mapping[str, float] | None = None,
    ):
        """`step_timeout_s` <= 0 disables the FAILED timeout of a scored
        subtask (tune after real-robot measurement, per the ICD note).
        `scorer(instruction, probe_override) -> bool` says whether a probe
        head scores that sentence; None = always. An unscored subtask ends
        DONE after `unscored_timeouts[instruction]`, else `unscored_timeout_s`
        (<= 0: never — cancel only)."""
        self._lock = threading.Lock()
        self._timeout_s = step_timeout_s
        self._scorer = scorer
        self._unscored_timeout_s = unscored_timeout_s
        self._unscored_timeouts = dict(unscored_timeouts or {})
        self._status = SubtaskStatus.IDLE
        self._plan_id = ""
        self._index = 0
        self._action = ""
        self._instruction = ""
        self._args: tuple[str, ...] = ()
        self._scored = True             # False: no verdict, timeout -> DONE
        self._subtask_timeout_s = 0.0   # this subtask's timeout (0 = none)
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
        args: tuple[str, ...] = (),
    ) -> Effect:
        with self._lock:
            if cancel:
                return self._on_cancel(plan_id, index)
            return self._on_task(plan_id, index, action, instruction, now, args)

    def accepts_cmd(self) -> bool:
        """False while RUNNING: a non-cancel cmd arriving now is ignored."""
        with self._lock:
            return self._status is not SubtaskStatus.RUNNING

    def on_progress(
        self, state: ProgressState, progress: float, now: float
    ) -> Effect:
        """Runner calls this after each inference with the probe's verdict.

        RUNNING → DONE (state=DONE) or FAILED (state=STALLED) with FREEZE.
        Anything else is a no-op (already terminal, or not our concern)."""
        with self._lock:
            if self._status is not SubtaskStatus.RUNNING or not self._scored:
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
            # Scored: the probe never said DONE in time -> FAILED. Unscored:
            # the timeout IS the plan -> DONE.
            if (
                self._status is SubtaskStatus.RUNNING
                and self._deadline is not None
                and now >= self._deadline
            ):
                t = self._subtask_timeout_s
                if self._scored:
                    self._status = SubtaskStatus.FAILED
                    self._detail = f"timeout {t:g}s"
                else:
                    self._status = SubtaskStatus.DONE
                    self._detail = f"done (timeout {t:g}s)"
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
        self, plan_id: str, index: int, action: str, instruction: str, now: float,
        args: tuple[str, ...] = (),
    ) -> Effect:
        if self._status is SubtaskStatus.RUNNING:
            # Only IDLE (or a finished DONE/FAILED) takes a new subtask. The
            # running one ends through cancel, a probe verdict or the step
            # timeout — never by replacement. Nothing changes here.
            return Effect.NONE

        # A DONE/FAILED countdown in progress is cut short: that task is over.
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
            return Effect.NONE  # nothing was running, nothing to freeze

        self._status = SubtaskStatus.RUNNING
        self._instruction = instruction
        self._args = tuple(args)
        # Scored or not is the instruction's call (is a head fitted on it?),
        # and that picks which timeout runs and what it lands as.
        if self._scorer is None:
            self._scored = True
        else:
            self._scored = bool(self._scorer(instruction, override_from_args(self._args)))
        if self._scored:
            self._subtask_timeout_s = self._timeout_s
        else:
            self._subtask_timeout_s = self._unscored_timeouts.get(
                instruction, self._unscored_timeout_s
            )
        self._detail = ""
        self._progress = 0.0
        self._deadline = (
            now + self._subtask_timeout_s if self._subtask_timeout_s > 0 else None
        )
        # Starting needs no cursor effect: the inference loop picks up the
        # instruction on its next build and the first chunk unpins the cursor.
        return Effect.NONE

    def _reset_to_idle(self, *, detail: str = "") -> None:
        """Wipe subtask state back to IDLE. detail keeps the last reason
        ('cancelled' after a cancel, '' after natural DONE/FAILED expiry)."""
        self._status = SubtaskStatus.IDLE
        self._plan_id = ""
        self._index = 0
        self._action = ""
        self._instruction = ""
        self._args = ()
        self._scored = True
        self._subtask_timeout_s = 0.0
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

    def probe_override(self) -> str | None:
        """'probe=<name>' from the RUNNING cmd's args, else None (probe_bank)."""
        with self._lock:
            if self._status is not SubtaskStatus.RUNNING:
                return None
            return override_from_args(self._args)

    def scored(self) -> bool:
        """True while a RUNNING subtask takes probe verdicts; False for an
        unscored one (no head for its instruction: ends DONE by timeout)
        and while nothing runs."""
        with self._lock:
            return self._status is SubtaskStatus.RUNNING and self._scored

    def timeout_s(self) -> float:
        """The RUNNING subtask's timeout in seconds, 0 = none."""
        with self._lock:
            if self._status is not SubtaskStatus.RUNNING:
                return 0.0
            return self._subtask_timeout_s

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
