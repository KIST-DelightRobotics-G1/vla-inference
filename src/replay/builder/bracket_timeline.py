"""Wrapping a timeline in the standing bracket: Timeline -> ActionStream.

This is the only way to obtain an `ActionStream`, so the compressed-gap
safety gate lives here: a timeline whose gap fill was capped time-compresses
a real pose change, and turning it into a publishable stream must be an
explicit choice (`force=True`), never an accident of a caller forgetting a
check.
"""

import numpy as np

from ..constants import CONTROL_DT_NS, HAND_DIM, TOKEN_DIM
from .action_stream import ActionStream
from .blending import blend
from .timeline_builder import Timeline


class CompressedGapError(ValueError):
    """The timeline has gaps whose fill was capped (`Gap.compressed`).

    Publishing it would ramp a real pose change faster than the recording —
    possibly faster than the robot can follow. Callers that accept the risk
    pass `force=True`.
    """

    def __init__(self, timeline: Timeline):
        self.gaps = timeline.compressed_gaps
        worst = max(self.gaps, key=lambda g: g.ticks)
        super().__init__(
            f"{len(self.gaps)} gap(s) were compressed, worst {worst.ticks} ticks "
            f"({worst.duration_s:.2f}s) after seq {worst.after_seq} filled in "
            f"{worst.filled_ticks} ticks "
            f"({worst.filled_ticks * CONTROL_DT_NS / 1e9:.2f}s) — publishing would "
            f"ramp a real pose change too fast; pass force=True to accept"
        )


def bracket_timeline(
    timeline: Timeline,
    standing_token: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    *,
    lead_in_ticks: int,
    lead_out_ticks: int,
    blend_ticks: int,
    force: bool = False,
) -> ActionStream:
    """Wrap a timeline in the standing lead-in/out and the two crossfades.

    Returns an `ActionStream` whose T ticks are laid out as:

        lead_in   standing token — lets gearsonic's arbiter claim VLA
                  (200 ms freshness) from a known pose before anything moves
        blend     standing -> the timeline's first tick
        replay    the timeline
        blend     the timeline's last tick -> standing
        lead_out  standing token, then the caller stops publishing

    The hand channels are wrapped the same way, resting at the given
    left/right hand pose (open or fist) during the bracket.

    Without the tail, the stream would end mid-pose and gearsonic would run
    its own LOST recovery 500 ms later (blend to standing, planner reseed,
    back to the origin) — safe, but starting from wherever the episode
    happened to stop.

    Raises:
        CompressedGapError: the timeline has compressed gaps and `force` is
            False (see the class docstring).
    """
    if timeline.compressed_gaps and not force:
        raise CompressedGapError(timeline)

    standing = np.asarray(standing_token, dtype=np.float32).reshape(-1)
    left_hand_q = np.asarray(left_hand_q, dtype=np.float32).reshape(-1)
    right_hand_q = np.asarray(right_hand_q, dtype=np.float32).reshape(-1)
    if standing.shape != (TOKEN_DIM,):
        raise ValueError(f"standing_token must have {TOKEN_DIM} values, got {standing.shape}")
    for name, hand in (("left_hand_q", left_hand_q), ("right_hand_q", right_hand_q)):
        if hand.shape != (HAND_DIM,):
            raise ValueError(f"{name} must have {HAND_DIM} values, got {hand.shape}")

    lead_in = max(0, lead_in_ticks)
    lead_out = max(0, lead_out_ticks)
    blend_ticks = max(0, blend_ticks)

    def wrap(body: np.ndarray, rest: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.tile(rest, (lead_in, 1)),
                blend(rest, body[0], blend_ticks),
                body,
                blend(body[-1], rest, blend_ticks),
                np.tile(rest, (lead_out, 1)),
            ]
        ).astype(np.float32)

    return ActionStream(
        tokens=wrap(timeline.tokens, standing),
        left_hand=wrap(timeline.left_hand, left_hand_q),
        right_hand=wrap(timeline.right_hand, right_hand_q),
    )
