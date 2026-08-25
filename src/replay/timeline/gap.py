"""Gap — one hole in the recorded 50 Hz grid, and how it was covered."""

from dataclasses import dataclass

from ..constants import CONTROL_DT_NS


@dataclass
class Gap:
    """A hole in the recorded 50 Hz grid, and how the timeline covers it."""

    after_seq: int  # seq of the last recorded tick before the gap
    ticks: int  # missing ticks on the grid
    filled_ticks: int  # ticks actually emitted (== ticks unless compressed)

    @property
    def compressed(self) -> bool:
        return self.filled_ticks < self.ticks

    @property
    def duration_s(self) -> float:
        return self.ticks * CONTROL_DT_NS / 1e9
