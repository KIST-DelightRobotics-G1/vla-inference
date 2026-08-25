"""Shared constants for the replay pipeline."""

# gearsonic's control period — the motion-token stream's grid spacing
# (WholeBodyController::kControlDt = 0.02 s).
CONTROL_DT_NS = 20_000_000
TOKEN_DIM = 64
HAND_DIM = 7

# ControlArbiter::Mode values, as recorded in motion_token.csv.
ARBITER_NORMAL = 0
ARBITER_TELEOP = 1
ARBITER_VLA = 2
ARBITER_RECOVERING = 3

ARBITER_NAMES = {
    ARBITER_NORMAL: "normal",
    ARBITER_TELEOP: "teleop",
    ARBITER_VLA: "vla",
    ARBITER_RECOVERING: "recovering",
}
