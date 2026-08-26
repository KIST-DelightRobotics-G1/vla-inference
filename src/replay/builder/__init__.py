"""Resampling aligned ticks onto a uniform 50 Hz timeline, safely bracketed.

The recording is not uniformly sampled: rows exist only for ticks that
decoded a token; `seq`/`stamp_ns` gaps mark periods outside CONTROL (INIT
ramp, damping, e-stop) — not loss. This package fixes that in two steps:

    timeline_builder.py  AlignedTokens -> Timeline. Resamples onto a
                         strict 20 ms grid keyed on `stamp_ns` (gearsonic's
                         computation-tick clock) and fills gaps by blending
                         across them, capped at `max_hold_ticks` so a long
                         e-stop pause does not stretch the replay. Owns its
                         builder-internal products (Timeline, Gap).
    bracket_timeline.py  Timeline -> ActionStream. The safety gate — a
                         compressed gap needs an explicit force — plus the
                         standing lead-in/out bracket.
    action_stream.py     ActionStream — the stage's contract with the
                         publisher: the finished publish plan, iterable as
                         LatentActionSteps (dataclasses handed to another
                         stage get their own file)
    latent_action_step.py
                         LatentActionStep — one publish tick, the pure-Python
                         twin of the DDS wire struct, yielded by iterating an
                         ActionStream
    blending.py          the blend (linear ramp) primitive both steps share

Cross-stream time never appears here: the input is the align stage's
`AlignedTokens`, where every stream already sits on the token clock. The
only time this package handles is *on the grid* — quantization and gaps.
"""

from .action_stream import ActionStream
from .blending import blend
from .bracket_timeline import CompressedGapError, bracket_timeline
from .timeline_builder import Gap, Timeline, build_timeline

__all__ = [
    "ActionStream",
    "blend",
    "bracket_timeline",
    "build_timeline",
    "CompressedGapError",
    "Gap",
    "Timeline",
]
