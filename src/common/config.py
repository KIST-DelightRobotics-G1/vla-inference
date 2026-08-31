"""Checkpoint-coupled constants shared by the publishers."""

import numpy as np

# 64-dim SONIC motion token for a stable standing pose — SONIC v1.1 space
# (sonic_v1_1/model_encoder.onnx). Derived 2026-08-31 from session
# 20260831_080750's measured standby stance (the pose held through the whole
# recording), encoded as a 10-frame hold — tests/derive_standing_token.py
# re-derives it. Cross-checked against gearsonic's own live-encoded standby
# tokens in that session's motion_token.csv (independent C++ packing, same
# leading values). Keep identical to gearsonic's kVlaSafeStandingToken
# (include/vla/vla_initial_pose.hpp).
#
# WARNING: this token is specific to the SONIC checkpoint gearsonic decodes
# with. A different SONIC checkpoint encodes a different latent space — when
# the gearsonic-side SONIC checkpoint changes, this value MUST be replaced
# with a known safe standing pose in the new latent space.
DEFAULT_INITIAL_MOTION_TOKEN = np.array(
    [
         0.0000, -0.3750, -0.1250, -0.1250,  0.1250,  0.1250, -0.0625,
        -0.0625, -0.2500, -0.0625, -0.1875, -0.0625,  0.3750,  0.1250,
         0.1250,  0.0625, -0.1250, -0.1250, -0.1250,  0.1250, -0.0625,
        -0.0625,  0.0000,  0.2500, -0.4375,  0.2500, -0.1250,  0.0625,
         0.1875, -0.2500,  0.0000,  0.1250,  0.0000,  0.0000,  0.2500,
         0.0000, -0.1250, -0.0625,  0.1250, -0.0625, -0.2500,  0.1875,
        -0.0625,  0.1250,  0.0000,  0.4375,  0.3750,  0.0000,  0.2500,
        -0.1250, -0.0625, -0.0625, -0.3125, -0.1250,  0.1250, -0.1875,
         0.4375, -0.0625,  0.0625, -0.0625, -0.2500,  0.0625, -0.1875,
         0.1250,
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
