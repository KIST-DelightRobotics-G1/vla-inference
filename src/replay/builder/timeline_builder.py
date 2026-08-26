"""Building the uniform 50 Hz timeline: AlignedTokens -> Timeline.

See the package docstring (`replay/builder/__init__.py`) for what the
recording is *not* and why the grid exists.
"""

from dataclasses import dataclass, field

import numpy as np

from ..aligner import AlignedTokens
from ..constants import CONTROL_DT_NS
from .blending import blend


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


@dataclass
class Timeline:
    """A uniform 50 Hz action stream ready to publish, one row per tick.

    Builder-internal: produced here, consumed by `bracket_timeline` — never
    handed to another stage (that is the ActionStream's job).
    """

    tokens: np.ndarray  # (T, 64) float32
    left_hand: np.ndarray  # (T, 7) float32
    right_hand: np.ndarray  # (T, 7) float32
    synthetic: np.ndarray  # (T,) bool — True where the tick fills a gap
    gaps: list[Gap] = field(default_factory=list)

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


def build_timeline(
    tokens: AlignedTokens,
    *,
    max_hold_ticks: int = 25,
) -> Timeline:
    """Resample an aligned recording onto a strict 20 ms grid, gaps blended.

    Args:
        tokens: the align stage's `AlignedTokens` (every stream already on
            the token clock — this function never looks at `recv_ns`).
        max_hold_ticks: longest gap fill emitted; a longer gap is compressed
            to this many ticks (reported via `Gap.compressed`). The fill is a
            blend from the pre-gap token to the post-gap one, so the stream
            never steps discontinuously — but a *compressed* gap ramps across
            a real pose change in that many ticks, which is why
            `bracket_timeline` refuses such a timeline unless forced.

    Raises:
        ValueError: the recording is empty.
    """
    if len(tokens) == 0:
        raise ValueError("the recording has no rows — nothing to replay")

    order = np.argsort(tokens.stamp_ns, kind="stable")
    stamp = tokens.stamp_ns[order]
    seq = tokens.seq[order]
    token_vals = tokens.values[order]
    left_q = tokens.left_hand[order]
    right_q = tokens.right_hand[order]

    # Grid index of each row on the 20 ms tick clock. Rounding absorbs the
    # jitter of the publisher's system_clock stamp; two rows landing on the
    # same index (a stamp hiccup) collapse to the later one.
    grid = np.rint((stamp - stamp[0]) / CONTROL_DT_NS).astype(np.int64)
    grid = np.maximum.accumulate(grid)  # stamps are sorted; keep the grid so too
    unique_mask = np.concatenate([grid[1:] != grid[:-1], [True]])  # keep last of a run
    grid, seq, token_vals, left_q, right_q = (
        grid[unique_mask],
        seq[unique_mask],
        token_vals[unique_mask],
        left_q[unique_mask],
        right_q[unique_mask],
    )

    # Walk the recorded ticks, emitting gap fills in between. Built as a list
    # of segments so a session with an hour-long e-stop pause never
    # materializes an hour-long array.
    out_tokens: list[np.ndarray] = []
    out_left: list[np.ndarray] = []
    out_right: list[np.ndarray] = []
    out_synth: list[np.ndarray] = []
    gaps: list[Gap] = []

    for i in range(len(grid)):
        if i > 0:
            missing = int(grid[i] - grid[i - 1] - 1)
            if missing > 0:
                fill = min(missing, max(0, max_hold_ticks))
                gaps.append(Gap(after_seq=int(seq[i - 1]), ticks=missing, filled_ticks=fill))
                if fill > 0:
                    out_tokens.append(blend(token_vals[i - 1], token_vals[i], fill))
                    out_left.append(blend(left_q[i - 1], left_q[i], fill))
                    out_right.append(blend(right_q[i - 1], right_q[i], fill))
                    out_synth.append(np.ones(fill, dtype=bool))
        out_tokens.append(token_vals[i : i + 1])
        out_left.append(left_q[i : i + 1])
        out_right.append(right_q[i : i + 1])
        out_synth.append(np.zeros(1, dtype=bool))

    return Timeline(
        tokens=np.concatenate(out_tokens).astype(np.float32),
        left_hand=np.concatenate(out_left).astype(np.float32),
        right_hand=np.concatenate(out_right).astype(np.float32),
        synthetic=np.concatenate(out_synth),
        gaps=gaps,
    )
