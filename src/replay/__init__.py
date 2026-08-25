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

Layout:

    io/           recording readers (csv_io: collector sessions, parquet_io:
                  LeRobot export episodes — needs pyarrow, the [parquet]
                  extra) and their output struct (MotionTokenRows)
    timeline.py   resampling onto the strict 20 ms grid, gap blending,
                  the standing lead-in/out bracket
    action_stream.py / latent_action_step.py
                  the publish-side structs: the finished plan, and one tick
                  of it (the pure-Python twin of the DDS wire struct)
    session.py    session directory / episode parquet -> ReplayTimeline
    cli.py        the publisher (`scripts/replay_session.py` /
                  `python -m replay`)
"""

from .constants import (
    ARBITER_NAMES,
    ARBITER_NORMAL,
    ARBITER_RECOVERING,
    ARBITER_TELEOP,
    ARBITER_VLA,
    CONTROL_DT_NS,
    HAND_DIM,
    TOKEN_DIM,
)
from .action_stream import ActionStream
from .io import (
    MotionTokenRows,
    read_episode_parquet,
    read_hand_csv,
    read_motion_token_csv,
    resolve_episode_path,
)
from .latent_action_step import LatentActionStep
from .session import load_episode, load_session
from .timeline import (
    Gap,
    ReplayTimeline,
    align_by_recv_ns,
    blend,
    bracket_timeline,
    build_timeline,
)

__all__ = [
    "ARBITER_NAMES",
    "ARBITER_NORMAL",
    "ARBITER_RECOVERING",
    "ARBITER_TELEOP",
    "ARBITER_VLA",
    "CONTROL_DT_NS",
    "HAND_DIM",
    "TOKEN_DIM",
    "MotionTokenRows",
    "read_hand_csv",
    "read_motion_token_csv",
    "read_episode_parquet",
    "resolve_episode_path",
    "load_episode",
    "load_session",
    "ActionStream",
    "Gap",
    "LatentActionStep",
    "ReplayTimeline",
    "align_by_recv_ns",
    "blend",
    "bracket_timeline",
    "build_timeline",
]
