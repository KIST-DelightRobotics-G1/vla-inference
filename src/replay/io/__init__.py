"""Recording readers for the replay pipeline.

    csv_io.py              collector session CSVs (motion_token.csv,
                           hand[_cmd]_*.csv)
    parquet_io.py          LeRobot training-export episodes (needs pyarrow,
                           the [parquet] extra)
    motion_token_rows.py   MotionTokenRows — the struct both readers produce,
                           the contract with `timeline.build_timeline`
"""

from .csv_io import read_hand_csv, read_motion_token_csv
from .motion_token_rows import MotionTokenRows
from .parquet_io import read_episode_parquet, resolve_episode_path

__all__ = [
    "read_hand_csv",
    "read_motion_token_csv",
    "read_episode_parquet",
    "resolve_episode_path",
    "MotionTokenRows",
]
