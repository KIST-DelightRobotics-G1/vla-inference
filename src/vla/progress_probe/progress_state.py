"""ProgressMonitor: probe raw score stream -> subtask verdict.

Stateless observation (raw) in, time-axis statistics out. Running max
(progress is monotonic by fit), 5s sliding slope, and the stalled /
plateaued / done flags drive one ProgressState verdict that
SubtaskMachine.on_progress consumes.

reset() must be called on every new subtask — running max carried over
from the previous subtask would make the new one look near-done from
tick zero.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum


class ProgressState(str, Enum):
    """Verdict fed to SubtaskMachine.on_progress(state, progress, now)."""

    RUNNING = "running"    # still in progress
    DONE = "done"          # complete: progress >= done_threshold or plateaued high
    STALLED = "stalled"    # failed: stuck at low progress


@dataclass
class Reading:
    """One monitor tick, ready to log or feed into on_progress."""

    raw: float                    # probe's raw score, unclipped
    progress: float               # running max, clipped to [0, 1]
    slope_per_s: float | None     # 5s slope, None until the window fills
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
        done_threshold: float = 0.70,
        stuck_value_threshold: float = 0.55,
        slope_stuck_threshold: float = 0.027,   # per second
        window_s: float = 5.0,
    ):
        self._done_th = done_threshold
        self._stuck_val_th = stuck_value_threshold
        self._slope_th = slope_stuck_threshold
        self._window_s = window_s
        self.reset()

    def reset(self) -> None:
        """Wipe running max and slope history — call on every new subtask."""
        self._running_max = 0.0
        self._history: deque[tuple[float, float]] = deque()

    def update(self, raw: float, now: float) -> Reading:
        # 1) running max, clipped to [0, 1]
        progress = min(1.0, max(0.0, max(self._running_max, raw)))
        self._running_max = progress

        # 2) 5s sliding window of (t, progress)
        self._history.append((now, progress))
        while self._history and now - self._history[0][0] > self._window_s:
            self._history.popleft()

        # 3) slope — only when the window is nearly full (avoid warmup noise)
        slope: float | None = None
        if len(self._history) >= 2:
            t0, p0 = self._history[0]
            span = now - t0
            if span >= self._window_s * 0.95:
                slope = (progress - p0) / span

        # 4) stuck branch: slope low AND which side of stuck_value_threshold
        stuck = slope is not None and slope < self._slope_th
        stalled = stuck and progress < self._stuck_val_th
        plateaued = stuck and progress >= self._stuck_val_th

        # 5) done: absolute value OR plateaued (high but stopped moving)
        done = progress >= self._done_th or plateaued

        # 6) verdict — stalled beats done (they can't both be true anyway)
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
