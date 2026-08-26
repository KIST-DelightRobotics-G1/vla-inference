"""Readers for LeRobot-format episode parquets -> Tokens / Joints.

`kist-vision-training`'s export writes each episode as one parquet file; the
per-frame columns carry the same quantities the collector CSVs record, so
both readers return the same dataclasses csv_io returns — the format dies
here:

    action.motion_token             (64,)  ->  Tokens.values
    teleop.{left,right}_hand_joints (7,)   ->  Tokens.left/right_hand
    observation.state               (43,)  ->  Joints.q (29 body joints)
    observation.root_orientation    (4,)   ->  Joints.base_quat

`timestamp` (seconds, already on the export's 20 ms grid) becomes
`stamp_ns`/`recv_ns`; a hole in the grid (frames the export dropped) shows
up as a `Gap` downstream and is blended over like a CSV gap. `frame_index`
becomes `seq`. `arbiter_mode` is filled with ARBITER_TELEOP: the export
keeps only the teleop demonstration segments, so the label is honest.

Requires `pyarrow` (the `[parquet]` extra); imported lazily so the CSV path
keeps working without it.
"""

import json
from pathlib import Path

import numpy as np

from ..constants import ARBITER_TELEOP, HAND_DIM, TOKEN_DIM
from .joints import Joints
from .tokens import Tokens

TOKEN_COLUMN = "action.motion_token"
HAND_COLUMNS = {"left": "teleop.left_hand_joints", "right": "teleop.right_hand_joints"}
TIMESTAMP_COLUMN = "timestamp"
FRAME_INDEX_COLUMN = "frame_index"

# The 29 body joints inside the episode's 43-dim state vector
# (modality order: legs 0:12, waist 12:15, left_arm 15:22, left_hand 22:29,
# right_arm 29:36, right_hand 36:43) — hands excluded, MuJoCo order kept.
BODY_IN_STATE43 = np.array(list(range(0, 22)) + list(range(29, 36)), dtype=np.int64)



def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "reading episode parquet files requires pyarrow — "
            "install the [parquet] extra: uv pip install -e '.[parquet]'"
        ) from e
    return pq


def _column_matrix(table, name: str, width: int, path: Path) -> np.ndarray:
    """One list-typed column as a dense (N, width) float64 matrix."""
    if name not in table.column_names:
        raise ValueError(
            f"{path} has no column {name!r} — not a LeRobot episode of this "
            f"dataset format (columns start {table.column_names[:6]})"
        )
    values = np.asarray(table[name].to_pylist(), dtype=np.float64)
    values = values.reshape(len(table), -1)
    if values.shape[1] != width:
        raise ValueError(f"{path} column {name!r} has width {values.shape[1]}, expected {width}")
    return values


def _read_grid(path: Path, pq, columns: list[str]):
    """Read `columns` (+ the grid columns) and return (table, stamp_ns, seq)."""
    wanted = [TIMESTAMP_COLUMN, FRAME_INDEX_COLUMN, *columns]
    # Select only the columns that exist so a missing one surfaces as
    # _column_matrix's descriptive error, not pyarrow's KeyError.
    present = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in wanted if c in present])

    timestamp = _column_matrix(table, TIMESTAMP_COLUMN, 1, path).ravel()
    frame_index = _column_matrix(table, FRAME_INDEX_COLUMN, 1, path).ravel()
    stamp_ns = np.rint(timestamp * 1e9).astype(np.int64)
    return table, stamp_ns, frame_index.astype(np.int64)


def read_tokens(path: str | Path) -> Tokens:
    """An episode's token content — same shape csv_io.read_tokens returns.

    Hands are always the episode's `teleop.*_hand_joints` (the export
    carries no measured-hand alternative), sharing the token grid's clock —
    so downstream alignment is the identity.
    """
    pq = _require_pyarrow()
    path = Path(path)
    table, stamp_ns, seq = _read_grid(path, pq, [TOKEN_COLUMN, *HAND_COLUMNS.values()])

    values = _column_matrix(table, TOKEN_COLUMN, TOKEN_DIM, path)
    hands = {
        side: (stamp_ns, _column_matrix(table, name, HAND_DIM, path).astype(np.float32))
        for side, name in HAND_COLUMNS.items()
    }
    return Tokens(
        recv_ns=stamp_ns,
        stamp_ns=stamp_ns,
        seq=seq,
        arbiter_mode=np.full(len(table), ARBITER_TELEOP, dtype=np.int64),
        encoder_mode=np.zeros(len(table), dtype=np.int64),
        values=values.astype(np.float32).reshape(-1, TOKEN_DIM),
        left_hand=hands["left"],
        right_hand=hands["right"],
        hands_from="cmd",
    )


def read_joints(path: str | Path) -> Joints:
    """An episode's joint content — same shape csv_io.read_joints returns.

    Reads `observation.state` (the measured joints — the motion that actually
    happened). The episode also carries `action.wbc` (commanded targets), but
    re-encoding those was validated to DIVERGE from the recorded tokens
    (median per-tick cosine 0.56 vs 0.95 for state): WBC commands mix in
    balancing compensation that is not the pose trajectory the encoder
    expects. The episode carries no joint velocities (`dq=None`) — the
    encoding stage falls back to finite differences. The clock is the token
    grid, so downstream alignment is the identity.
    """
    column = "observation.state"
    pq = _require_pyarrow()
    path = Path(path)
    table, stamp_ns, _ = _read_grid(path, pq, [column, "observation.root_orientation"])

    state43 = _column_matrix(table, column, 43, path)
    base_quat = _column_matrix(table, "observation.root_orientation", 4, path)
    return Joints(
        recv_ns=stamp_ns,
        q=state43[:, BODY_IN_STATE43],
        base_quat=base_quat,
        dq=None,
    )


def resolve_episode_path(dataset_dir: str | Path, episode_index: int) -> Path:
    """Episode index -> parquet path, via the dataset's `meta/info.json`.

    Uses the dataset's own `data_path` template and `chunks_size`, so chunked
    layouts (`data/chunk-001/...`) resolve correctly.
    """
    dataset_dir = Path(dataset_dir)
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} not found — {dataset_dir} is not a LeRobot dataset root"
        )
    with open(info_path) as f:
        info = json.load(f)

    total = int(info.get("total_episodes", 0))
    if not 0 <= episode_index < total:
        raise ValueError(f"episode {episode_index} out of range — the dataset has {total}")

    chunk = episode_index // int(info.get("chunks_size", 1000))
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    path = dataset_dir / rel
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (from {info_path}'s data_path template)")
    return path
