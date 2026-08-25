"""Assemble sensor readings into a Gr00tPolicy observation.

Follows ``run_vla_inference.py``'s ``prepare_observation_from_sensors`` from
the reference stack, with the pinocchio robot model replaced by the static
tables in ``g1_joints``.

Target structure (batch size 1, temporal horizon 1, per the UNITREE_G1_SONIC
modality config):

    {
        "video":    {"ego_view": uint8 (1, 1, H, W, 3), [wrist views...]},
        "state":    {group: float32 (1, 1, D) for the 7 joint groups
                     + "projected_gravity": float32 (1, 1, 3)},
        "language": {<language_key>: [["<prompt>"]]},
    }
"""

from typing import Any

import numpy as np

from common.g1_joints import apply_hand_hardware_coupling, assemble_full_q, split_state
from .transforms import compute_projected_gravity


class ObservationBuilder:
    """Builds policy observations from camera + state messages."""

    def __init__(self, language_key: str = "annotation.human.task_description"):
        self.language_key = language_key

    def build(
        self,
        camera_msg: dict[str, Any] | None,
        state_msg: dict[str, Any] | None,
        prompt: str,
        log_errors: bool = False,
    ) -> dict[str, Any] | None:
        """Return an observation dict, or None if sensor data is missing."""
        if camera_msg is None or "ego_view" not in camera_msg.get("images", {}):
            if log_errors:
                print("[ObservationBuilder] waiting for camera message...", flush=True)
            return None
        if state_msg is None:
            if log_errors:
                print("[ObservationBuilder] waiting for state message...", flush=True)
            return None

        for key in ("body_q", "left_hand_q", "right_hand_q", "base_quat"):
            if key not in state_msg:
                if log_errors:
                    print(f"[ObservationBuilder] state message missing '{key}'", flush=True)
                return None

        images = camera_msg["images"]
        video = {"ego_view": np.asarray(images["ego_view"])[np.newaxis, np.newaxis]}
        if "left_wrist" in images:
            video["left_wrist"] = np.asarray(images["left_wrist"])[np.newaxis, np.newaxis]
        if "right_wrist" in images:
            # The embodiment names the right wrist stream "wrist_view".
            video["wrist_view"] = np.asarray(images["right_wrist"])[np.newaxis, np.newaxis]

        left_hand_q = apply_hand_hardware_coupling(state_msg["left_hand_q"])
        full_q = assemble_full_q(
            body_q=state_msg["body_q"],
            left_hand_q=left_hand_q,
            right_hand_q=state_msg["right_hand_q"],
        )

        state = {
            group: values[np.newaxis, np.newaxis].astype(np.float32)
            for group, values in split_state(full_q).items()
        }

        base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
        assert base_quat.shape == (4,), f"base_quat must have shape (4,), got {base_quat.shape}"
        state["projected_gravity"] = compute_projected_gravity(base_quat)[
            np.newaxis, np.newaxis
        ]

        return {
            "video": video,
            "state": state,
            "language": {self.language_key: [[prompt]]},
        }
