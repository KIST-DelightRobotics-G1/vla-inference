"""ChunkCursor — ActionChunks in at ~2.5 Hz, one ChunkStep out per 50 Hz tick.

The stage's logic: hold the newest ActionChunk and a play cursor over its 40
steps. `push()` (the inference loop's side) swaps the chunk in, skipping the
steps that went stale while the prediction was being computed; `step()` (the
publish thread's side) hands out one ChunkStep per 20 ms tick. Pure
bookkeeping under one lock — no threads, no clocks, no DDS: the runner
computes `skip_ticks` from its own timestamps and the publisher owns the
50 Hz schedule, so every policy here is testable tick by tick.

Replacement policy — a chunk's step 0 is the action for the moment the
observation was taken; by the time the prediction arrives, that moment is
`skip_ticks` in the past. Starting there would command the robot to rewind,
so the cursor starts at `skip_ticks` instead (adjacent SONIC tokens overlap
~98%, so the splice from the previous chunk is smooth). A chunk arriving
entirely stale (skip >= 40, inference slower than the horizon) still lands
on its last step — the freshest target available beats silence.

Exhaustion policy — when the chunk runs out before the next one arrives,
the last step is held for `hold_ticks` more (rides out an occasional slow
inference), then `step()` returns None: the publisher goes silent, and
500 ms later gearsonic's LOST recovery blends to the safe standing pose.
Holding forever would keep a dead policy in command; the bounded hold makes
a real failure end in the verified recovery path. A later push() resumes
the stream — gearsonic re-claims from the origin, no restart needed.
"""

import threading

from ..policy.action_chunk import ActionChunk
from .chunk_step import ChunkStep

# Ride out one slow inference (~2x the 400 ms budget) but let a dead policy
# hit gearsonic's LOST recovery (0.5 s hold + 0.5 s staleness = 1 s to safe
# standing).
HOLD_TICKS = 25


class ChunkCursor:
    """The seam between the inference rate and the control rate.

    Counters (`stats()`) tell the health story: `held` ticks mean inference
    is occasionally late, `starved` ticks mean it stalled long enough to end
    the stream, `stale_pushes` mean a whole prediction aged past its horizon.
    """

    def __init__(self, *, hold_ticks: int = HOLD_TICKS):
        self._lock = threading.Lock()
        self._chunk: ActionChunk | None = None
        self._cursor = 0
        self._held = 0
        self._hold_ticks = hold_ticks
        self._stats = {
            "pushes": 0,
            "stale_pushes": 0,
            "steps": 0,
            "held": 0,
            "starved": 0,
            "skipped": 0,
        }

    def push(self, chunk: ActionChunk, *, skip_ticks: int = 0) -> None:
        """Swap in a fresh prediction, skipping the ticks it spent in flight."""
        if skip_ticks < 0:
            raise ValueError(f"skip_ticks must be >= 0, got {skip_ticks}")
        horizon = len(chunk.motion_token)
        with self._lock:
            self._stats["pushes"] += 1
            if skip_ticks >= horizon:
                self._stats["stale_pushes"] += 1
                skip_ticks = horizon - 1
            self._stats["skipped"] += skip_ticks
            self._chunk = chunk
            self._cursor = skip_ticks
            self._held = 0

    def step(self) -> ChunkStep | None:
        """The next tick's action — held past the end, None once given up."""
        with self._lock:
            if self._chunk is None:
                self._stats["starved"] += 1
                return None

            horizon = len(self._chunk.motion_token)
            index = self._cursor
            if index >= horizon:
                if self._held >= self._hold_ticks:
                    self._stats["starved"] += 1
                    return None
                self._held += 1
                self._stats["held"] += 1
                index = horizon - 1
            else:
                self._cursor += 1

            self._stats["steps"] += 1
            return ChunkStep(
                motion_token=self._chunk.motion_token[index],
                left_hand_joints=self._chunk.left_hand_joints[index],
                right_hand_joints=self._chunk.right_hand_joints[index],
            )

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
