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
    timeline/     20 ms grid resampling, gap blending, the safety gate +
                  standing bracket, and its products (ReplayTimeline,
                  ActionStream)
    latent_action_step.py
                  one publish tick — the pure-Python twin of the DDS wire
                  struct, yielded by iterating an ActionStream
    latent_action_publisher.py
                  the execution layer: owns the DDS channel and the Tx
                  worker thread, puts an ActionStream on
                  rt/kist/latent_action at 50 Hz (main thread = lifecycle)
    session.py    session directory / episode parquet -> ReplayTimeline
    cli.py        the entry point (`scripts/replay_session.py` /
                  `python -m replay`): Config + load -> bracket -> publish
"""

# Public surface = names external code actually references. Internal types
# and primitives (MotionTokenRows, ReplayTimeline, ActionStream, Gap, blend,
# ...) stay importable from their subpackages (replay.io, replay.timeline).
from .constants import ARBITER_TELEOP, ARBITER_VLA, CONTROL_DT_NS, TOKEN_DIM
from .io import (
    read_episode_parquet,
    read_hand_csv,
    read_motion_token_csv,
    resolve_episode_path,
)
from .session import load_episode, load_session
from .timeline import CompressedGapError, bracket_timeline, build_timeline

__all__ = [
    "ARBITER_TELEOP",
    "ARBITER_VLA",
    "CONTROL_DT_NS",
    "TOKEN_DIM",
    "read_hand_csv",
    "read_motion_token_csv",
    "read_episode_parquet",
    "resolve_episode_path",
    "load_episode",
    "load_session",
    "CompressedGapError",
    "bracket_timeline",
    "build_timeline",
]
