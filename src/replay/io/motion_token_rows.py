"""MotionTokenRows — the readers' output struct (the pipeline's input side)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionTokenRows:
    """Raw recorded ticks, one entry per row the recording actually has.

    Non-uniform: rows exist only for ticks the recording kept, and
    `timeline.build_timeline` resamples them onto the strict 20 ms grid.
    `csv_io.read_motion_token_csv` produces this from `motion_token.csv`;
    `parquet_io.read_episode_parquet` converts a LeRobot episode into the
    same shape, so both sources share the downstream pipeline.
    """

    recv_ns: np.ndarray  # (N,) int64
    stamp_ns: np.ndarray  # (N,) int64
    seq: np.ndarray  # (N,) int64
    arbiter_mode: np.ndarray  # (N,) int64
    encoder_mode: np.ndarray  # (N,) int64
    tokens: np.ndarray  # (N, 64) float32

    def __len__(self) -> int:
        return len(self.stamp_ns)
