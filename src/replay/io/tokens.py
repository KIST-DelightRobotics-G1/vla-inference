"""Tokens — a recording's token content, format-independent."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Tokens:
    """Everything token-related a recording carries, as recorded.

    The same dataclass comes out of every reader (csv_io.read_tokens,
    parquet_io.read_tokens) — the file format dies inside io/. Disk-faithful:
    non-uniform ticks with gaps, side streams on their own clocks;
    `replay.aligner` joins the streams onto the token clock, and the builder
    resamples onto the strict 20 ms grid downstream.

    The hand streams ride along because the wire publishes them together
    with the token (`kist_msgs::LatentActionStep` = token[64] + hands[7]x2);
    they may run on their own `recv_ns` clocks (CSV) or share the token grid
    (parquet).
    """

    recv_ns: np.ndarray  # (N,) int64 — cross-stream alignment clock
    stamp_ns: np.ndarray  # (N,) int64 — the 20 ms grid clock
    seq: np.ndarray  # (N,) int64
    arbiter_mode: np.ndarray  # (N,) int64 (ARBITER_* values)
    encoder_mode: np.ndarray  # (N,) int64
    values: np.ndarray  # (N, 64) float32 — the motion tokens themselves

    left_hand: tuple[np.ndarray, np.ndarray] | None  # (recv_ns, q) or None
    right_hand: tuple[np.ndarray, np.ndarray] | None
    hands_from: str  # "cmd" (both readers publish the commanded targets)

    def __len__(self) -> int:
        return len(self.stamp_ns)
