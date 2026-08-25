"""Resampling recorded ticks onto a uniform 50 Hz timeline.

Two things the recording is *not*, and this package fixes both:

- **Not uniformly sampled.** Rows exist only for ticks that decoded a
  token; `seq`/`stamp_ns` gaps mark periods outside CONTROL (INIT ramp,
  damping, e-stop) — not loss. `build_timeline` resamples onto a strict
  20 ms grid keyed on `stamp_ns` (gearsonic's computation-tick clock) and
  fills gaps by blending across them, capped at `max_hold_ticks` so a long
  e-stop pause does not stretch the replay. Every gap is reported so the
  caller can refuse to play across a big one.
- **Not one stream.** The hand-command rows carry only `recv_ns` (HandCmd_
  has no clock of its own), so they align to the token rows by `recv_ns` —
  newest command at or before each tick, the `merge_asof` rule the storage
  format prescribes.

Layout, one unit per file:

    gap.py                 Gap — one hole in the grid, and how it was covered
    replay_timeline.py     ReplayTimeline — the uniform timeline struct
    blending.py            align_by_recv_ns / blend primitives
    build_timeline.py      recorded rows -> ReplayTimeline
    bracket_timeline.py    ReplayTimeline -> ActionStream (standing bracket)
"""

from .blending import align_by_recv_ns, blend
from .bracket_timeline import bracket_timeline
from .build_timeline import build_timeline
from .gap import Gap
from .replay_timeline import ReplayTimeline

__all__ = [
    "align_by_recv_ns",
    "blend",
    "bracket_timeline",
    "build_timeline",
    "Gap",
    "ReplayTimeline",
]
