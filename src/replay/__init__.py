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

    io/           recording readers — csv_io (collector sessions) and
                  parquet_io (LeRobot episodes) both return the same Tokens
                  / Joints dataclasses: the file format dies in io/
    aligner/      cross-stream joins onto the token clock — raw hand and
                  joint streams become one row per tick (AlignedTokens,
                  AlignedJoints): cross-stream time dies in aligner/
    encoder/      the encoding stage: AlignedJoints -> EncodedTokens through
                  the SONIC encoder ONNX (checkpoint portability)
    builder/      20 ms grid resampling, gap blending, the safety gate +
                  standing bracket; its contract with the publisher is the
                  ActionStream, iterable as LatentActionSteps (the
                  pure-Python twin of the DDS wire struct)
    publisher/    the execution stage: owns the DDS channel and the Tx
                  worker thread, puts an ActionStream on
                  rt/kist/latent_action at 50 Hz (main thread = lifecycle)
    cli.py        the entry point (`scripts/replay_session.py` /
                  `python -m replay`): Config + load -> bracket -> publish
"""

# Public surface = names external code actually references. Internal types
# and primitives (Timeline, ActionStream, Gap, blend, read_* functions,
# ...) stay importable from their subpackages (replay.io, replay.builder).
from .constants import ARBITER_TELEOP, CONTROL_DT_NS, TOKEN_DIM
from .io import Joints, Tokens
from .builder import CompressedGapError, bracket_timeline, build_timeline

__all__ = [
    "ARBITER_TELEOP",
    "CONTROL_DT_NS",
    "TOKEN_DIM",
    "Joints",
    "Tokens",
    "CompressedGapError",
    "bracket_timeline",
    "build_timeline",
]
