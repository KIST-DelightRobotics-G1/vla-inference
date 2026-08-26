"""Joints — a recording's whole-body joint content, format-independent."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Joints:
    """The recorded joint streams, as recorded (own clock, own resolution).

    The same dataclass comes out of every reader (csv_io.read_joints,
    parquet_io.read_joints). Joined onto the token clock (`replay.aligner`)
    and fed to the SONIC encoder (`replay.encoder`), these become tokens —
    that is how a recording survives a decoder-checkpoint change.
    """

    recv_ns: np.ndarray  # (M,) int64 — this stream's own clock
    q: np.ndarray  # (M, 29) MuJoCo/Unitree order (rad)
    base_quat: np.ndarray  # (M, 4) pelvis quaternion, wxyz
    dq: np.ndarray | None  # (M, 29) measured velocities (CSV) or None (parquet)

    def __len__(self) -> int:
        return len(self.recv_ns)
