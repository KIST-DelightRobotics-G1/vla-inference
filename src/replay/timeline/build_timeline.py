"""Building the uniform 50 Hz timeline from recorded rows.

See the package docstring (`replay/timeline/__init__.py`) for what the
recording is *not* and why the grid exists.
"""

import numpy as np

from ..constants import ARBITER_NAMES, CONTROL_DT_NS, HAND_DIM
from ..io.motion_token_rows import MotionTokenRows
from .blending import align_by_recv_ns, blend
from .gap import Gap
from .replay_timeline import ReplayTimeline


def build_timeline(
    rows: MotionTokenRows,
    *,
    left_hand: tuple[np.ndarray, np.ndarray] | None = None,
    right_hand: tuple[np.ndarray, np.ndarray] | None = None,
    arbiter_modes: tuple[int, ...] | None = None,
    max_hold_ticks: int = 25,
    hands_from: str = "none",
) -> ReplayTimeline:
    """Resample recorded tokens onto a strict 20 ms grid, gaps blended over.

    Args:
        rows: parsed `motion_token.csv`.
        left_hand / right_hand: `(recv_ns, q)` from `read_hand_csv`; None
            falls back to the open-hand pose (zeros, Dex motor order).
        arbiter_modes: keep only these `arbiter_mode` values (None = all).
            Restricting to `(ARBITER_TELEOP,)` replays only the operator
            demonstration segments, which is what the training export keeps.
        max_hold_ticks: longest gap fill emitted; a longer gap is compressed
            to this many ticks (reported via `Gap.compressed`). The fill is a
            blend from the pre-gap token to the post-gap one, so the stream
            never steps discontinuously — but a *compressed* gap ramps across
            a real pose change in that many ticks, which is why the caller
            should gate on it.
        hands_from: provenance label for the report ("cmd" / "state").

    Raises:
        ValueError: no rows survive filtering, or the grid is degenerate.
    """
    if len(rows) == 0:
        raise ValueError("motion_token.csv has no rows — nothing to replay")

    keep = np.ones(len(rows), dtype=bool)
    if arbiter_modes is not None:
        keep = np.isin(rows.arbiter_mode, np.asarray(arbiter_modes, dtype=np.int64))
        if not keep.any():
            have = sorted({int(m) for m in rows.arbiter_mode})
            raise ValueError(
                f"no rows with arbiter_mode in {list(arbiter_modes)}; "
                f"the session has modes {have} "
                f"({', '.join(ARBITER_NAMES.get(m, str(m)) for m in have)})"
            )

    order = np.argsort(rows.stamp_ns[keep], kind="stable")
    stamp = rows.stamp_ns[keep][order]
    recv = rows.recv_ns[keep][order]
    seq = rows.seq[keep][order]
    tokens = rows.tokens[keep][order]
    modes = rows.arbiter_mode[keep][order]

    # Grid index of each row on the 20 ms tick clock. Rounding absorbs the
    # jitter of the publisher's system_clock stamp; two rows landing on the
    # same index (a stamp hiccup) collapse to the later one.
    grid = np.rint((stamp - stamp[0]) / CONTROL_DT_NS).astype(np.int64)
    grid = np.maximum.accumulate(grid)  # stamps are sorted; keep the grid so too
    unique_mask = np.concatenate([grid[1:] != grid[:-1], [True]])  # keep last of a run
    grid, recv, seq, tokens, modes = (
        grid[unique_mask],
        recv[unique_mask],
        seq[unique_mask],
        tokens[unique_mask],
        modes[unique_mask],
    )

    # Hand targets for the recorded ticks, aligned by recv_ns.
    hand_before_first = 0
    hands: list[np.ndarray] = []
    for source in (left_hand, right_hand):
        if source is None:
            hands.append(np.zeros((len(grid), HAND_DIM), dtype=np.float32))
            continue
        src_recv, src_q = source
        if len(src_recv) == 0:
            hands.append(np.zeros((len(grid), HAND_DIM), dtype=np.float32))
            continue
        q, before_first = align_by_recv_ns(recv, src_recv, src_q)
        hand_before_first = max(hand_before_first, int(before_first.sum()))
        hands.append(q)
    left_q, right_q = hands

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
                    out_tokens.append(blend(tokens[i - 1], tokens[i], fill))
                    out_left.append(blend(left_q[i - 1], left_q[i], fill))
                    out_right.append(blend(right_q[i - 1], right_q[i], fill))
                    out_synth.append(np.ones(fill, dtype=bool))
        out_tokens.append(tokens[i : i + 1])
        out_left.append(left_q[i : i + 1])
        out_right.append(right_q[i : i + 1])
        out_synth.append(np.zeros(1, dtype=bool))

    mode_counts = {int(m): int((modes == m).sum()) for m in np.unique(modes)}
    return ReplayTimeline(
        tokens=np.concatenate(out_tokens).astype(np.float32),
        left_hand=np.concatenate(out_left).astype(np.float32),
        right_hand=np.concatenate(out_right).astype(np.float32),
        synthetic=np.concatenate(out_synth),
        gaps=gaps,
        arbiter_modes=mode_counts,
        hands_from=hands_from if (left_hand or right_hand) else "none",
        hand_ticks_before_first=hand_before_first,
    )
