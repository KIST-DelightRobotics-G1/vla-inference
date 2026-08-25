"""Readers for the kist-data-collector CSV formats.

The header shapes are pinned against the collector's motion_token_rows.hpp /
dex3_cmd_rows.hpp; a schema change on the collector side shows up in
`tests/test_replay.py`.
"""

import csv
from pathlib import Path

import numpy as np

from ..constants import HAND_DIM, TOKEN_DIM
from .motion_token_rows import MotionTokenRows


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
