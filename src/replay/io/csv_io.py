"""Readers for the kist-data-collector CSV formats -> Tokens / Joints.

The header shapes are pinned against the collector's motion_token_rows.hpp /
dex3_cmd_rows.hpp / lowstate_rows.hpp; a schema change on the collector side
shows up in the tests. The format dies here: both readers return the same
dataclasses parquet_io returns.
"""

import csv
from pathlib import Path

import numpy as np

from ..constants import HAND_DIM, TOKEN_DIM
from .joints import Joints
from .tokens import Tokens

# G1 29-DoF body motors; lowstate.csv carries 35 slots (m29..m34 unused).
_NUM_BODY_MOTORS = 29


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


def read_hand_csv(path: str | Path, *, column: str = "q") -> tuple[np.ndarray, np.ndarray]:
    """Load a Dex3 hand CSV as (recv_ns, q) with q shaped (N, 7).

    `column="q"` reads `f{i}_q` — the commanded target column in
    `hand_cmd_{side}.csv`. Motor order is the Dex order the wire uses (thumb
    x3, index x2, middle x2), identical to the wire's hand fields, so no
    permutation is needed.
    """
    path = Path(path)
    names = ["recv_ns"] + [f"f{i}_{column}" for i in range(HAND_DIM)]
    cols = _read_columns(path, names)

    q = np.stack([cols[f"f{i}_{column}"] for i in range(HAND_DIM)], axis=-1)
    return cols["recv_ns"].astype(np.int64), q.astype(np.float32).reshape(-1, HAND_DIM)


def read_tokens(session_dir: str | Path) -> Tokens:
    """A session's token content: `motion_token.csv` + the hand-command CSVs.

    Hands come from `hand_cmd_{side}.csv` — the commanded targets, which is
    what the wire republishes (a command encodes the intended grip force by
    pressing past contact; the measured `hand_{side}.csv` stops at contact
    and would replay a weaker grasp). A missing or empty hand file degrades
    to no stream (open-hand pose downstream) rather than failing.

    Raises:
        FileNotFoundError: no `motion_token.csv` in the directory.
    """
    session_dir = Path(session_dir)
    token_csv = session_dir / "motion_token.csv"
    if not token_csv.exists():
        raise FileNotFoundError(
            f"{token_csv} not found — the session was recorded without the "
            f"motion_token stream (config.yaml `motion_token.enabled`), so it "
            f"carries no latent action to replay"
        )

    names = ["recv_ns", "stamp_ns", "seq", "arbiter_mode", "encoder_mode"]
    names += [f"t{i:02d}" for i in range(TOKEN_DIM)]
    cols = _read_columns(token_csv, names)
    values = np.stack([cols[f"t{i:02d}"] for i in range(TOKEN_DIM)], axis=-1)

    hands: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"left": None, "right": None}
    for side in ("left", "right"):
        path = session_dir / f"hand_cmd_{side}.csv"
        if path.exists():
            recv, q = read_hand_csv(path)
            if len(recv):
                hands[side] = (recv, q)

    return Tokens(
        recv_ns=cols["recv_ns"].astype(np.int64),
        stamp_ns=cols["stamp_ns"].astype(np.int64),
        seq=cols["seq"].astype(np.int64),
        arbiter_mode=cols["arbiter_mode"].astype(np.int64),
        encoder_mode=cols["encoder_mode"].astype(np.int64),
        values=values.astype(np.float32).reshape(-1, TOKEN_DIM),
        left_hand=hands["left"],
        right_hand=hands["right"],
        hands_from="cmd",
    )


def read_joints(session_dir: str | Path) -> Joints:
    """A session's joint content: `lowstate.csv`, on its own (faster) clock.

    The collector records every rt/lowstate message (see its
    lowstate_rows.hpp): `m{i:02d}_q` / `m{i:02d}_dq` are the 29 body motors
    in Unitree/MuJoCo order, and `quat_*` is the pelvis IMU quaternion (wxyz).

    Raises:
        FileNotFoundError: no `lowstate.csv` in the directory.
    """
    lowstate_csv = Path(session_dir) / "lowstate.csv"
    if not lowstate_csv.exists():
        raise FileNotFoundError(
            f"{lowstate_csv} not found — the session carries no joint "
            f"stream to re-encode"
        )

    names = ["recv_ns", "quat_w", "quat_x", "quat_y", "quat_z"]
    names += [f"m{i:02d}_{f}" for i in range(_NUM_BODY_MOTORS) for f in ("q", "dq")]
    cols = _read_columns(lowstate_csv, names)

    q = np.stack([cols[f"m{i:02d}_q"] for i in range(_NUM_BODY_MOTORS)], axis=-1)
    dq = np.stack([cols[f"m{i:02d}_dq"] for i in range(_NUM_BODY_MOTORS)], axis=-1)
    quat = np.stack([cols[f"quat_{c}"] for c in "wxyz"], axis=-1)
    return Joints(
        recv_ns=cols["recv_ns"].astype(np.int64),
        q=q,
        base_quat=quat,
        dq=dq,
    )
