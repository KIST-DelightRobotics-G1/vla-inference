"""Observation — one assembled model input, the stage's contract with policy.

Semantic content only: camera images by view, the state groups the
UNITREE_G1_SONIC embodiment defines, and the language prompt. The
gr00t-specific formatting (batch/time axes, the language-key nesting) is
the policy stage's job — this dataclass carries what the sensors said, in
model order, at one moment.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One model input, assembled from fresh sensor snapshots.

    `video` maps the embodiment's view names (e.g. "ego_view") to RGB
    images; `state` maps the embodiment's state keys (the 7 joint groups
    from `common.g1_joints.split_state` + "projected_gravity") to float32
    vectors. Holding one proves every required stream was present and
    fresh when it was built.
    """

    video: dict[str, np.ndarray]  # view -> (H, W, 3) uint8 RGB
    state: dict[str, np.ndarray]  # group -> (D,) float32
    prompt: str
    # Timing of the frames that went in (optional — the builder fills these,
    # synthetic observations in tests leave them empty):
    camera_age_s: dict[str, float] = field(default_factory=dict)
    """view -> age of its frame when the observation was built, seconds on
    THIS machine's monotonic clock from DDS arrival (trustworthy)."""
    camera_stamp_ns: dict[str, int] = field(default_factory=dict)
    """view -> the sensor-side capture stamp (ext-sensor-io's clock; only
    comparable to ours if the two clocks are synced)."""
