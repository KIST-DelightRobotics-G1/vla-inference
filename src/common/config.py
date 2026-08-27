"""Checkpoint-coupled constants shared by the publishers."""

import numpy as np

# 64-dim SONIC motion token for a stable standing pose, from the reference
# gear_sonic/utils/inference/initial_poses.py.
#
# WARNING (from the reference): this token is specific to the SONIC checkpoint
# used during training. A different SONIC checkpoint encodes a different
# latent space — when the gearsonic-side SONIC checkpoint changes, this value
# MUST be replaced with a known safe standing pose in the new latent space.
DEFAULT_INITIAL_MOTION_TOKEN = np.array(
    [
        -0.0625,  0.0000, -0.0625, -0.1250, -0.1875, -0.0625,  0.1875,
         0.2500,  0.1875, -0.1250,  0.0625, -0.0625, -0.2500, -0.2500,
        -0.3125, -0.0625,  0.0000, -0.0625, -0.1250, -0.1875,  0.0000,
        -0.2500,  0.0000, -0.2500, -0.0625,  0.0625,  0.1250, -0.1250,
         0.2500,  0.1875,  0.2500, -0.1250,  0.1250,  0.1875, -0.0625,
         0.0000, -0.1875, -0.1875,  0.2500,  0.0000,  0.0000, -0.1250,
         0.0625,  0.0000, -0.0625, -0.0625,  0.1875, -0.0625,  0.0000,
         0.0625,  0.1250,  0.0625,  0.1250,  0.0625,  0.1250,  0.0000,
         0.1250,  0.1875,  0.0000,  0.0000,  0.0625,  0.0625,  0.1875,
         0.0625,
    ],
    dtype=np.float32,
)
