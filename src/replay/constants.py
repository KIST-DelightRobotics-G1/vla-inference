"""Replay-only constants, plus re-exports of the wire-contract dimensions.

TOKEN_DIM / HAND_DIM / CONTROL_DT_NS are facts of the kist_msgs wire
contract — their single source is `common/cyclonedds/kist_msgs.py` (change
them there, together with the IDL). They are re-exported here because every
replay module speaks in them; the ARBITER_* values below are genuinely
replay-only (kist-data-collector recording semantics, as written into
motion_token.csv).
"""

from common.cyclonedds.kist_msgs import CONTROL_DT_NS, HAND_DIM, TOKEN_DIM

# ControlArbiter::Mode values, as recorded in motion_token.csv's
# arbiter_mode column (0=normal, 1=teleop, 2=vla, 3=recovering).
ARBITER_NORMAL = 0
ARBITER_TELEOP = 1
ARBITER_VLA = 2
ARBITER_RECOVERING = 3

__all__ = [
    "CONTROL_DT_NS",
    "HAND_DIM",
    "TOKEN_DIM",
    "ARBITER_NORMAL",
    "ARBITER_TELEOP",
    "ARBITER_VLA",
    "ARBITER_RECOVERING",
]
