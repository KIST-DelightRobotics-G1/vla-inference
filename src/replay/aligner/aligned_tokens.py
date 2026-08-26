"""AlignedTokens — a recording's token content with every stream on one clock."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedTokens:
    """A recording's token content with every stream on the token clock.

    Same rows as the `Tokens` it came from — still non-uniform, still gappy;
    the 20 ms grid is the builder's job — but the hand streams are joined:
    one (N, 7) row per token tick instead of raw (recv_ns, q) side streams.
    Holding one proves the cross-stream alignment already happened.
    """

    recv_ns: np.ndarray  # (N,) int64
    stamp_ns: np.ndarray  # (N,) int64 — the 20 ms grid clock
    seq: np.ndarray  # (N,) int64
    arbiter_mode: np.ndarray  # (N,) int64 (ARBITER_* values)
    encoder_mode: np.ndarray  # (N,) int64
    values: np.ndarray  # (N, 64) float32 — the motion tokens themselves

    left_hand: np.ndarray  # (N, 7) float32 — open-hand zeros when not recorded
    right_hand: np.ndarray  # (N, 7) float32
    hands_from: str  # "cmd", or "none" when no hand stream was recorded
    hand_ticks_before_first: int  # ticks predating the first hand row (clamped to it)

    def __len__(self) -> int:
        return len(self.stamp_ns)
