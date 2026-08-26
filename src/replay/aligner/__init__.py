"""Joining the recording's side streams onto the token clock.

A recording is not one table but several streams, each on its own clock:
token rows (recv_ns + stamp_ns), hand-command rows (recv_ns only), and the
whole-body joint stream (recv_ns only). Everything downstream wants a single
clock — the token rows' — so cross-stream time dies here, the way the file
format dies in io/:

    time_aligner.py     the logic — align_by_recv_ns (the shared join rule:
                        newest source row at or before each token tick,
                        backward merge_asof on recv_ns), align_tokens, and
                        align_joints
    aligned_tokens.py   AlignedTokens — the raw (recv_ns, q) hand streams
                        joined to one (N, 7) row per token tick
    aligned_joints.py   AlignedJoints — the joint stream resampled 1:1 onto
                        the token rows (the encoder input)

A parquet episode is one table on one clock, so both joins are the identity
there; collector CSV sessions are where they do real work.
"""

from .aligned_joints import AlignedJoints
from .aligned_tokens import AlignedTokens
from .time_aligner import align_by_recv_ns, align_joints, align_tokens

__all__ = [
    "AlignedJoints",
    "AlignedTokens",
    "align_by_recv_ns",
    "align_joints",
    "align_tokens",
]
