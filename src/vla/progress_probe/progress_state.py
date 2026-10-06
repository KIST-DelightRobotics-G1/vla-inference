"""ProgressMonitor: probe raw score stream -> subtask verdict.

Stateless observation (raw) in, time-axis statistics out. Running max
(progress is monotonic by fit), 5s sliding slope, and the stalled / done
flags drive one ProgressState verdict that SubtaskMachine.on_progress
consumes. DONE is the absolute threshold only; a high plateau is NOT done
(the `plateaued` flag is logged for the record but carries no verdict —
a stuck-but-high episode stays RUNNING until the step timeout or the
orchestrator decides).

reset() must be called on every new subtask — running max carried over
from the previous subtask would make the new one look near-done from
tick zero.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum


class ProgressState(str, Enum):
    """Verdict fed to SubtaskMachine.on_progress(state, progress, now)."""

    RUNNING = "running"
    DONE = "done"
    STALLED = "stalled"


@dataclass
class Reading:
    """One monitor tick, ready to log or feed into on_progress."""

    raw: float
    progress: float
    slope_per_s: float | None
    stalled: bool
    plateaued: bool
    done: bool
    state: ProgressState


class ProgressMonitor:
    """Stream raw probe scores in, get a Reading with a verdict out.

    Thresholds are the v0.1 defaults from the probe experiments; tune
    against real-robot logs (see progress_log.py output) before locking in.
    """

    def __init__(
        self,
        *,
        done_threshold: float = 0.75,
        stuck_value_threshold: float = 0.55,
        slope_stuck_threshold: float = 0.027,
        window_s: float = 5.0,
        stall_min_elapsed_s: float = 10.0,
        stall_hold_s: float = 3.0,
    ):
        """STALLED needs all of: slope < slope_stuck_threshold over window_s,
        progress < stuck_value_threshold, at least `stall_min_elapsed_s` since
        reset() (the approach phase of a task is low and flat by nature), and
        that condition holding continuously for `stall_hold_s` (a single flat
        window is not a verdict). DONE is unaffected: progress >= done_threshold
        fires at once. slope_stuck_threshold is per second."""
        self._done_th = done_threshold
        self._stuck_val_th = stuck_value_threshold
        self._slope_th = slope_stuck_threshold
        self._window_s = window_s
        self._stall_min_elapsed_s = stall_min_elapsed_s
        self._stall_hold_s = stall_hold_s
        self.reset()

    def reset(self) -> None:
        """Wipe running max and slope history — call on every new subtask."""
        self._running_max = 0.0
        self._history: deque[tuple[float, float]] = deque()
        self._t_start: float | None = None
        self._stall_since: float | None = None

    def update(self, raw: float, now: float) -> Reading:
        if self._t_start is None:
            self._t_start = now
        progress = min(1.0, max(0.0, max(self._running_max, raw)))
        self._running_max = progress

        self._history.append((now, progress))
        while self._history and now - self._history[0][0] > self._window_s:
            self._history.popleft()

        slope: float | None = None
        if len(self._history) >= 2:
            t0, p0 = self._history[0]
            span = now - t0
            if span >= self._window_s * 0.95:
                slope = (progress - p0) / span

        stuck = slope is not None and slope < self._slope_th
        plateaued = stuck and progress >= self._stuck_val_th
        low_flat = stuck and progress < self._stuck_val_th
        if low_flat and now - self._t_start >= self._stall_min_elapsed_s:
            if self._stall_since is None:
                self._stall_since = now
            stalled = now - self._stall_since >= self._stall_hold_s
        else:
            self._stall_since = None
            stalled = False

        done = progress >= self._done_th

        if stalled:
            state = ProgressState.STALLED
        elif done:
            state = ProgressState.DONE
        else:
            state = ProgressState.RUNNING

        return Reading(
            raw=raw,
            progress=progress,
            slope_per_s=slope,
            stalled=stalled,
            plateaued=plateaued,
            done=done,
            state=state,
        )
