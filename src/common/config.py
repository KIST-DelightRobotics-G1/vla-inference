"""Configuration for the VLA inference runner (tyro CLI dataclasses).

All robot-facing I/O rides CycloneDDS (see common/cyclonedds/). The only
non-DDS endpoint is the remote policy server (port 5550, Isaac-GR00T's own
PolicyServer/PolicyClient pair)."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .cyclonedds.kist_msgs import CONTROL_DT_NS

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


@dataclass
class PolicyConfig:
    """Which GR00T N1.7 policy backend to use."""

    mode: Literal["local", "remote"] = "local"
    """'local' loads Gr00tPolicy in-process (needs GPU + gr00t package);
    'remote' connects to a running PolicyServer over ZMQ."""

    model_path: str | None = None
    """Checkpoint directory (local mode). Must be a UNITREE_G1_SONIC finetune."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for the checkpoint (local mode)."""

    device: str = "cuda:0"
    """Device for model inference (local mode)."""

    host: str = "localhost"
    """PolicyServer host (remote mode)."""

    port: int = 5550
    """PolicyServer port (remote mode)."""


@dataclass
class IOConfig:
    """DDS endpoints toward gearsonic, ext-sensor-io, and the robot."""

    dds_domain_id: int = 0
    """DDS domain id (must match the gearsonic receiver)."""

    dds_camera_topic: str = "rt/kist/camera/color/h264"
    """kist-ext-sensor-io color topic mapped to the ego_view observation.
    Per-camera streams use rt/kist/camera/<name>/color/h264."""


@dataclass
class RunnerConfig:
    """Top-level runner configuration."""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    io: IOConfig = field(default_factory=IOConfig)

    prompt: str = "demo"
    """Initial language prompt (changeable at runtime via 'prompt:<text>')."""

    action_publish_rate: int = round(1e9 / CONTROL_DT_NS)
    """Rate at which single action steps are published to gearsonic (Hz).
    Default = gearsonic's control rate (one step per WBC tick — the same
    CONTROL_DT_NS the wire contract documents)."""

    action_horizon: int = 40
    """Steps per action chunk (fixed by the UNITREE_G1_SONIC embodiment)."""

    inference_rate: float = 2.5
    """Target policy inference rate (Hz)."""

    action_bound: float = 1.25
    """Reject chunks whose |motion_token| exceeds this bound."""

    language_key: str = "annotation.human.task_description"
    """Observation language key expected by the checkpoint."""

    initial_pose_blend_duration: float = 1.0
    """Seconds to blend from the last sent token to the initial pose token
    when the operator requests the initial pose (0 = snap instantly)."""

    verbose_timing: bool = False
    """Always print loop timing telemetry (not only when the loop is slow)."""
