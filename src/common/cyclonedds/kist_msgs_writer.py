"""DDS (CycloneDDS) implementation of ActionWriter.

Publishes the kist_msgs wire contract (see ``kist_msgs.py`` for the
topics, IDL type mirrors, and QoS) toward the gearsonic whole-body
controller. ``cyclonedds`` is imported lazily so the module stays importable
without the [dds] extra.
"""

import time

import numpy as np

from .kist_msgs import (
    HAND_DIM,
    LATENT_ACTION_TOPIC,
    TOKEN_DIM,
    WBC_COMMAND_TOPIC,
    command_qos,
    get_dds_types,
    latent_action_qos,
)


class KistMsgsWriter:
    """CycloneDDS publisher toward the gearsonic whole-body controller."""

    def __init__(self, domain_id: int = 0, *, action_topic: str = LATENT_ACTION_TOPIC):
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.pub import DataWriter
        from cyclonedds.topic import Topic

        LatentActionStep, WbcCommand = get_dds_types()
        self._LatentActionStep = LatentActionStep
        self._WbcCommand = WbcCommand

        self._participant = DomainParticipant(domain_id)
        self._action_writer = DataWriter(
            self._participant,
            Topic(self._participant, action_topic, LatentActionStep),
            qos=latent_action_qos(),
        )
        self._command_writer = DataWriter(
            self._participant,
            Topic(self._participant, WBC_COMMAND_TOPIC, WbcCommand),
            qos=command_qos(),
        )
        self._action_seq = 0
        self._command_seq = 0
        print(f"[KistMsgsWriter] Publishing on domain {domain_id}: "
              f"{action_topic}, {WBC_COMMAND_TOPIC}")

    def send_latent_action(
        self,
        motion_token: np.ndarray,
        frame_index: int,
        left_hand_joints: np.ndarray,
        right_hand_joints: np.ndarray,
    ) -> None:
        token = np.asarray(motion_token, dtype=np.float32).reshape(-1)
        left = np.asarray(left_hand_joints, dtype=np.float32).reshape(-1)
        right = np.asarray(right_hand_joints, dtype=np.float32).reshape(-1)
        if token.shape != (TOKEN_DIM,):
            raise ValueError(f"motion_token must have {TOKEN_DIM} values, got {token.shape}")
        if left.shape != (HAND_DIM,) or right.shape != (HAND_DIM,):
            raise ValueError(f"hand joints must have {HAND_DIM} values each")

        self._action_seq += 1
        self._action_writer.write(
            self._LatentActionStep(
                seq=self._action_seq,
                stamp_ns=time.time_ns(),
                frame_index=int(frame_index),
                token_state=token.tolist(),
                left_hand_joints=left.tolist(),
                right_hand_joints=right.tolist(),
            )
        )

    def send_command(self, start: bool, planner: bool = False) -> None:
        self._command_seq += 1
        self._command_writer.write(
            self._WbcCommand(
                seq=self._command_seq,
                stamp_ns=time.time_ns(),
                start=start,
                stop=not start,
                planner=planner,
                has_delta_heading=False,
                delta_heading=0.0,
            )
        )

    def close(self) -> None:
        # cyclonedds entities release their resources when garbage collected;
        # drop references deterministically.
        del self._action_writer
        del self._command_writer
        del self._participant
