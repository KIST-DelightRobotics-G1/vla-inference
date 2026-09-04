"""Checkpoint-coupled constants shared by the publishers."""

import numpy as np

# 64-dim SONIC motion token for a stable standing pose — SONIC v1.1 space
# (sonic_v1_1/model_encoder.onnx). Derived from the calmest stationary
# standing window of a recorded episode (episode_000003) held constant over
# the 10 future frames — tests/derive_standing_token.py re-derives it.
# HARDWARE-VERIFIED 2026-08-31 (replay lead-in stood well). Keep identical
# to gearsonic's kVlaSafeStandingToken (include/vla/vla_initial_pose.hpp).
#
# LESSON (2026-09-01): a re-derivation from session 20260831_080750's
# standby stance FAILED live — the robot never moved through the whole
# bracketed replay. That session's lower body was fixed (hung), so its
# stance never balanced; only derive this token from a pose the robot
# actually held ON THE GROUND, and hardware-verify before deploying.
#
# WARNING: this token is specific to the SONIC checkpoint gearsonic decodes
# with. A different SONIC checkpoint encodes a different latent space — when
# the gearsonic-side SONIC checkpoint changes, this value MUST be replaced
# with a known safe standing pose in the new latent space.
DEFAULT_INITIAL_MOTION_TOKEN = np.array(
    [
         0.2500, -0.3125, -0.0625,  0.0000, -0.0625, -0.0625,  0.1250,
         0.0625, -0.2500,  0.0625,  0.0000, -0.1250,  0.4375,  0.0000,
         0.1250,  0.0625,  0.0625,  0.0000,  0.0000, -0.0625,  0.0000,
         0.0000, -0.1250,  0.1250, -0.3750,  0.3125, -0.1250,  0.0000,
         0.2500, -0.4375,  0.1250, -0.0625, -0.0625,  0.1875,  0.3750,
         0.0625, -0.1250,  0.1875,  0.1250, -0.1250, -0.1250,  0.1250,
         0.1250, -0.3125,  0.2500,  0.4375,  0.5000, -0.1875, -0.1250,
        -0.1875,  0.0000,  0.0000, -0.4375, -0.1875,  0.1250, -0.0625,
         0.4375,  0.1250,  0.1250,  0.0000, -0.1250,  0.0000,  0.0625,
         0.0000,
    ],
    dtype=np.float32,
)

# Hand pose held alongside the standing token during the publish bracket
# (lead-in/lead-out) — together they define the safe rest posture: a firm
# fist, near the finger joint limits. Values are the commanded fist targets
# read out of session 20260831_080750's hand_cmd_*.csv (round numbers, so a
# deliberate target pose; the measured hand state tracks them closely, so
# the hardware reaches it). Dex wire order; right hand mirrors by negation.
# The reference "middle gesture" (common.g1_joints.CLOSED_HAND_Q) is only a
# half-closed grip; OPEN_HAND_Q gives flat hands.
_FIRM_FIST_LEFT = np.array(
    [0.0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75], dtype=np.float32
)
DEFAULT_INITIAL_LEFT_HAND_Q = _FIRM_FIST_LEFT
DEFAULT_INITIAL_RIGHT_HAND_Q = -_FIRM_FIST_LEFT
