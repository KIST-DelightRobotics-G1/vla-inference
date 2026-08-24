"""Session replay: recorded motion tokens -> a uniform 50 Hz action timeline.

Turns a `kist-data-collector` session directory into the exact stream the
runner would publish, so a recorded episode can be played back on the real
robot without a policy:

    motion_token.csv        t00..t63   ->  LatentActionStep.token_state
    hand_cmd_{side}.csv     f0_q..f6_q ->  LatentActionStep.{left,right}_hand_joints

`motion_token.csv` is the ground truth for this: it is a copy of the token
the gearsonic whole-body decoder actually consumed on each CONTROL tick
(`rt/kist/motion_token`, 50 Hz), so replaying it drives the decoder through
the same latent trajectory. The decoder stays closed-loop on live robot
state, so the robot balances itself — this is a latent replay, not an
open-loop joint playback.

Two things the recording is *not*, and this module fixes both:

- **Not uniformly sampled.** Rows exist only for ticks that decoded a
  token; `seq`/`stamp_ns` gaps mark periods outside CONTROL (INIT ramp,
  damping, e-stop) — not loss. `build_timeline` resamples onto a strict
  20 ms grid keyed on `stamp_ns` (gearsonic's computation-tick clock) and
  fills gaps by blending across them, capped at `max_hold_ticks` so a long
  e-stop pause does not stretch the replay. Every gap is reported so the
  caller can refuse to play across a big one.
- **Not one stream.** The hand-command rows carry only `recv_ns` (HandCmd_
  has no clock of its own), so they align to the token rows by `recv_ns` —
  newest command at or before each tick, the `merge_asof` rule the storage
  format prescribes.

Pure data handling: no DDS, no I/O beyond reading the CSVs, so it is
testable without cyclonedds or a robot. The publisher lives in
`scripts/replay_session.py`.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# gearsonic's control period — the motion-token stream's grid spacing
# (WholeBodyController::kControlDt = 0.02 s).
CONTROL_DT_NS = 20_000_000
TOKEN_DIM = 64
HAND_DIM = 7

# ControlArbiter::Mode values, as recorded in motion_token.csv.
ARBITER_NORMAL = 0
ARBITER_TELEOP = 1
ARBITER_VLA = 2
ARBITER_RECOVERING = 3

ARBITER_NAMES = {
    ARBITER_NORMAL: "normal",
    ARBITER_TELEOP: "teleop",
    ARBITER_VLA: "vla",
    ARBITER_RECOVERING: "recovering",
}


# ── CSV reading ──────────────────────────────────────────────────────────────


def _read_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Read the named float columns out of a collector CSV.

    The collector writes a fixed header row and `%.7g` floats, so a plain
    csv.reader is enough (and keeps pandas out of the dependency set).
    Returns float64 arrays; an empty file (header only) yields empty arrays.
    """
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty (no header row)") from None

        index = {name: i for i, name in enumerate(header)}
        missing = [c for c in columns if c not in index]
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s) {missing} — "
                f"header has {len(header)} columns starting {header[:6]}"
            )
        picks = [index[c] for c in columns]
        rows = [[row[i] for i in picks] for row in reader if row]

    if not rows:
        return {c: np.empty(0, dtype=np.float64) for c in columns}

    values = np.asarray(rows, dtype=np.float64)
    return {c: values[:, i] for i, c in enumerate(columns)}


@dataclass
class MotionTokenRows:
    """Raw `motion_token.csv` contents, one entry per recorded tick."""

    recv_ns: np.ndarray  # (N,) int64
    stamp_ns: np.ndarray  # (N,) int64
    seq: np.ndarray  # (N,) int64
    arbiter_mode: np.ndarray  # (N,) int64
    encoder_mode: np.ndarray  # (N,) int64
    tokens: np.ndarray  # (N, 64) float32

    def __len__(self) -> int:
        return len(self.stamp_ns)


def read_motion_token_csv(path: str | Path) -> MotionTokenRows:
    """Load `motion_token.csv` (recv_ns, stamp_ns, seq, modes, t00..t63)."""
    path = Path(path)
    names = ["recv_ns", "stamp_ns", "seq", "arbiter_mode", "encoder_mode"]
    names += [f"t{i:02d}" for i in range(TOKEN_DIM)]
    cols = _read_columns(path, names)

    tokens = np.stack([cols[f"t{i:02d}"] for i in range(TOKEN_DIM)], axis=-1)
    return MotionTokenRows(
        recv_ns=cols["recv_ns"].astype(np.int64),
        stamp_ns=cols["stamp_ns"].astype(np.int64),
        seq=cols["seq"].astype(np.int64),
        arbiter_mode=cols["arbiter_mode"].astype(np.int64),
        encoder_mode=cols["encoder_mode"].astype(np.int64),
        tokens=tokens.astype(np.float32).reshape(-1, TOKEN_DIM),
    )


def read_hand_csv(path: str | Path, *, column: str = "q") -> tuple[np.ndarray, np.ndarray]:
    """Load a Dex3 hand CSV as (recv_ns, q) with q shaped (N, 7).

    `column="q"` reads `f{i}_q` — the target column in `hand_cmd_{side}.csv`
    and the measured column in `hand_{side}.csv`, which is why the same
    reader serves both. Motor order is the Dex order the wire uses (thumb
    x3, index x2, middle x2), identical to `LatentActionStep`'s hand fields,
    so no permutation is needed.
    """
    path = Path(path)
    names = ["recv_ns"] + [f"f{i}_{column}" for i in range(HAND_DIM)]
    cols = _read_columns(path, names)

    q = np.stack([cols[f"f{i}_{column}"] for i in range(HAND_DIM)], axis=-1)
    return cols["recv_ns"].astype(np.int64), q.astype(np.float32).reshape(-1, HAND_DIM)


# ── alignment / blending ─────────────────────────────────────────────────────


def align_by_recv_ns(
    target_recv_ns: np.ndarray, src_recv_ns: np.ndarray, src_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Newest source row at or before each target time (backward merge_asof).

    Returns `(values, before_first)` where `values[i]` is the source row
    aligned to `target_recv_ns[i]`, and `before_first[i]` marks targets that
    precede the first source row — those clamp to the first row (the only
    honest choice without extrapolating) and the caller may warn about them.
    """
    src_recv_ns = np.asarray(src_recv_ns)
    if len(src_recv_ns) == 0:
        raise ValueError("no source rows to align against")

    idx = np.searchsorted(src_recv_ns, np.asarray(target_recv_ns), side="right") - 1
    before_first = idx < 0
    return src_values[np.clip(idx, 0, len(src_recv_ns) - 1)], before_first


def blend(start: np.ndarray, end: np.ndarray, n_ticks: int) -> np.ndarray:
    """Linear ramp from `start` toward `end` over `n_ticks` published ticks.

    Excludes `start` and includes `end` (`alpha = (i+1)/n`), matching the
    runner's initial-pose blend: the first published tick already moves, and
    the last one is exactly `end`.
    """
    if n_ticks <= 0:
        return np.empty((0, *np.shape(start)), dtype=np.float32)

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    alpha = ((np.arange(n_ticks, dtype=np.float32) + 1.0) / n_ticks).reshape(-1, 1)
    return ((1.0 - alpha) * start.reshape(1, -1) + alpha * end.reshape(1, -1)).astype(np.float32)


# ── timeline ─────────────────────────────────────────────────────────────────


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


def load_session(
    session_dir: str | Path,
    *,
    arbiter_modes: tuple[int, ...] | None = None,
    max_hold_ticks: int = 25,
    hand_source: str = "cmd",
) -> ReplayTimeline:
    """Build a replay timeline from a collector session directory.

    Args:
        session_dir: a `sessions/<YYYYMMDD_HHMMSS>/` directory.
        arbiter_modes: see `build_timeline`.
        max_hold_ticks: see `build_timeline`.
        hand_source: `"cmd"` reads `hand_cmd_{side}.csv` (the commanded
            targets — the action twin of the token, and the same quantity
            `LatentActionStep` carries), `"state"` reads `hand_{side}.csv`
            (measured), `"none"` publishes the open-hand pose. A missing or
            empty file degrades to the open-hand pose rather than failing:
            the hands are independent of the whole-body token.

    Raises:
        FileNotFoundError: no `motion_token.csv` in the directory.
        ValueError: the token stream is unusable (see `build_timeline`).
    """
    session_dir = Path(session_dir)
    token_csv = session_dir / "motion_token.csv"
    if not token_csv.exists():
        raise FileNotFoundError(
            f"{token_csv} not found — the session was recorded without the "
            f"motion_token stream (config.yaml `motion_token.enabled`), so it "
            f"carries no latent action to replay"
        )

    rows = read_motion_token_csv(token_csv)

    hands: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"left": None, "right": None}
    if hand_source in ("cmd", "state"):
        stem = "hand_cmd" if hand_source == "cmd" else "hand"
        for side in ("left", "right"):
            path = session_dir / f"{stem}_{side}.csv"
            if path.exists():
                recv, q = read_hand_csv(path)
                if len(recv):
                    hands[side] = (recv, q)
    elif hand_source != "none":
        raise ValueError(f"hand_source must be 'cmd', 'state' or 'none', got {hand_source!r}")

    return build_timeline(
        rows,
        left_hand=hands["left"],
        right_hand=hands["right"],
        arbiter_modes=arbiter_modes,
        max_hold_ticks=max_hold_ticks,
        hands_from=hand_source,
    )


def bracket_timeline(
    timeline: ReplayTimeline,
    standing_token: np.ndarray,
    open_hand: np.ndarray,
    *,
    lead_in_ticks: int,
    lead_out_ticks: int,
    blend_ticks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrap a timeline in the standing lead-in/out and the two crossfades.

    Returns `(tokens, left_hand, right_hand)`, each `(T, D)`, laid out as:

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

    return (
        wrap(timeline.tokens, standing),
        wrap(timeline.left_hand, open_hand),
        wrap(timeline.right_hand, open_hand),
    )
