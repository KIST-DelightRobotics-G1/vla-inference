"""Checkpoint-coupled constants shared by the publishers."""

import numpy as np

# 64-dim SONIC motion token for a stable standing pose — SONIC v1.1 space
# (sonic_v1_1/model_encoder.onnx). Derived 2026-08-28 from the calmest
# stationary standing window of a recorded episode (episode_000003) held
# constant over the 10 future frames — tests/derive_standing_token.py
# re-derives it. Keep identical to gearsonic's kVlaSafeStandingToken
# (include/vla/vla_initial_pose.hpp).
#
# NOT YET hardware-verified: stream it (gearsonic vla_receiver_probe /
# publish path) and confirm the robot stands before trusting it.
#
# WARNING: this token is specific to the SONIC checkpoint gearsonic decodes
# with. A different SONIC checkpoint encodes a different latent space — when
# the gearsonic-side SONIC checkpoint changes, this value MUST be replaced
# with a known safe standing pose in the new latent space. The previous
# (release-checkpoint) value is in git history.
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
