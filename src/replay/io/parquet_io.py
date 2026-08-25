"""Readers for LeRobot-format episode parquet files (the training export).

`kist-vision-training`'s export writes each episode as one parquet file whose
per-frame columns carry the same quantities the collector CSVs record:

    action.motion_token             (64,)  ->  LatentActionStep.token_state
    teleop.{left,right}_hand_joints (7,)   ->  LatentActionStep.*_hand_joints

so a training episode can be replayed on the robot exactly like a raw
collector session. The reader reshapes the columns into the `MotionTokenRows`
the CSV path produces, and everything downstream (`build_timeline`, gap
blending, the bracket) is shared:

- `timestamp` (seconds, already on the export's 20 ms grid) becomes
  `stamp_ns`/`recv_ns`; a hole in the grid (frames the export dropped) shows
  up as a `Gap` and is blended over like a CSV gap.
- `frame_index` becomes `seq`.
- `arbiter_mode` is filled with ARBITER_TELEOP: the export keeps only the
  teleop demonstration segments, so the label is honest and `--teleop-only`
  stays a no-op rather than an error.

Requires `pyarrow` (the `[parquet]` extra); imported lazily so the CSV path
keeps working without it.
"""

import json
from pathlib import Path

import numpy as np

from ..constants import ARBITER_TELEOP, HAND_DIM, TOKEN_DIM
from .motion_token_rows import MotionTokenRows

TOKEN_COLUMN = "action.motion_token"
HAND_COLUMNS = {"left": "teleop.left_hand_joints", "right": "teleop.right_hand_joints"}
TIMESTAMP_COLUMN = "timestamp"
FRAME_INDEX_COLUMN = "frame_index"


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


def read_episode_parquet(
    path: str | Path,
) -> tuple[MotionTokenRows, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load one episode parquet as `(rows, hands)` in the CSV readers' shapes.

    `rows` is a `MotionTokenRows` (see the module docstring for the column
    mapping); `hands` maps "left"/"right" to the `(recv_ns, q)` pairs
    `read_hand_csv` would return — sharing the token rows' clock, so the
    downstream merge_asof alignment is the identity.
    """
    pq = _require_pyarrow()
    path = Path(path)
    wanted = [TOKEN_COLUMN, TIMESTAMP_COLUMN, FRAME_INDEX_COLUMN, *HAND_COLUMNS.values()]
    # Select only the columns that exist so a missing one surfaces as
    # _column_matrix's descriptive error, not pyarrow's KeyError.
    present = set(pq.read_schema(path).names)
    table = pq.read_table(path, columns=[c for c in wanted if c in present])

    tokens = _column_matrix(table, TOKEN_COLUMN, TOKEN_DIM, path)
    timestamp = _column_matrix(table, TIMESTAMP_COLUMN, 1, path).ravel()
    frame_index = _column_matrix(table, FRAME_INDEX_COLUMN, 1, path).ravel()

    stamp_ns = np.rint(timestamp * 1e9).astype(np.int64)
    rows = MotionTokenRows(
        recv_ns=stamp_ns,
        stamp_ns=stamp_ns,
        seq=frame_index.astype(np.int64),
        arbiter_mode=np.full(len(table), ARBITER_TELEOP, dtype=np.int64),
        encoder_mode=np.zeros(len(table), dtype=np.int64),
        tokens=tokens.astype(np.float32).reshape(-1, TOKEN_DIM),
    )

    hands = {
        side: (stamp_ns, _column_matrix(table, name, HAND_DIM, path).astype(np.float32))
        for side, name in HAND_COLUMNS.items()
    }
    return rows, hands


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
