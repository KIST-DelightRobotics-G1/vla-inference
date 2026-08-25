"""ReplayTimeline — the uniform 50 Hz action timeline, one row per tick."""

from dataclasses import dataclass, field

import numpy as np

from ..constants import CONTROL_DT_NS
from .gap import Gap


@dataclass
class ReplayTimeline:
    """A uniform 50 Hz action stream ready to publish, one row per tick."""

    tokens: np.ndarray  # (T, 64) float32
    left_hand: np.ndarray  # (T, 7) float32
    right_hand: np.ndarray  # (T, 7) float32
    synthetic: np.ndarray  # (T,) bool — True where the tick fills a gap
    gaps: list[Gap] = field(default_factory=list)
    arbiter_modes: dict[int, int] = field(default_factory=dict)  # mode -> recorded ticks
    hands_from: str = "none"  # "cmd" / "state" / "none" (open-hand fallback)
    hand_ticks_before_first: int = 0

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def duration_s(self) -> float:
        return len(self.tokens) * CONTROL_DT_NS / 1e9

    @property
    def recorded_ticks(self) -> int:
        return int((~self.synthetic).sum())

    @property
    def worst_gap(self) -> Gap | None:
        return max(self.gaps, key=lambda g: g.ticks, default=None)

    @property
    def compressed_gaps(self) -> list[Gap]:
        """Gaps whose fill was capped — each one time-compresses a real pose
        change, which is why `bracket_timeline` refuses them by default."""
        return [g for g in self.gaps if g.compressed]
