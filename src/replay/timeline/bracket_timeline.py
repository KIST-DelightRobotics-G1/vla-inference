"""Wrapping a timeline in the standing lead-in/out bracket."""

import numpy as np

from ..action_stream import ActionStream
from ..constants import HAND_DIM, TOKEN_DIM
from .blending import blend
from .replay_timeline import ReplayTimeline


def bracket_timeline(
    timeline: ReplayTimeline,
    standing_token: np.ndarray,
    open_hand: np.ndarray,
    *,
    lead_in_ticks: int,
    lead_out_ticks: int,
    blend_ticks: int,
) -> ActionStream:
    """Wrap a timeline in the standing lead-in/out and the two crossfades.

    Returns an `ActionStream` whose T ticks are laid out as:

        lead_in   standing token — lets gearsonic's arbiter claim VLA
                  (200 ms freshness) from a known pose before anything moves
        blend     standing -> the timeline's first tick
        replay    the timeline
        blend     the timeline's last tick -> standing
        lead_out  standing token, then the caller stops publishing

    Without the tail, the stream would end mid-pose and gearsonic would run
    its own LOST recovery 500 ms later (blend to standing, planner reseed,
    back to the origin) — safe, but starting from wherever the episode
    happened to stop.
    """
    standing = np.asarray(standing_token, dtype=np.float32).reshape(-1)
    open_hand = np.asarray(open_hand, dtype=np.float32).reshape(-1)
    if standing.shape != (TOKEN_DIM,):
        raise ValueError(f"standing_token must have {TOKEN_DIM} values, got {standing.shape}")
    if open_hand.shape != (HAND_DIM,):
        raise ValueError(f"open_hand must have {HAND_DIM} values, got {open_hand.shape}")

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
        left_hand=wrap(timeline.left_hand, open_hand),
        right_hand=wrap(timeline.right_hand, open_hand),
    )
