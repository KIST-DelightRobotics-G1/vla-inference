"""Recording readers: the file format dies here.

    csv_io.py       collector session CSVs -> Tokens / Joints
    parquet_io.py   LeRobot export episodes -> Tokens / Joints
                    (needs pyarrow, the [parquet] extra)
    tokens.py       Tokens — a recording's token content (+ hand streams)
    joints.py       Joints — a recording's whole-body joint content

Whatever the format, `read_tokens` returns a `Tokens` and `read_joints` a
`Joints`; everything downstream (the encoding stage in
`replay.encoder`, timeline, publish) knows only these two dataclasses.
"""

from .joints import Joints
from .tokens import Tokens

__all__ = ["Joints", "Tokens"]
