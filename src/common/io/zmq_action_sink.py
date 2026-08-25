"""ZMQ implementation of ActionSink (latent protocol v4, reference-compatible)."""

import time

import numpy as np
import zmq

from ..protocol import build_command_message, pack_latent_action_message


class ZmqActionSink:
    """PUB socket publishing latent actions + control commands to gearsonic."""

    def __init__(self, host: str = "*", port: int = 5556):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        time.sleep(0.1)  # let subscribers attach before the first message
        print(f"[ZmqActionSink] Latent action PUB bound to tcp://{host}:{port}")

    def send_latent_action(
        self,
        motion_token: np.ndarray,
        frame_index: int,
        left_hand_joints: np.ndarray,
        right_hand_joints: np.ndarray,
    ) -> None:
        self._socket.send(
            pack_latent_action_message(
                motion_token=motion_token,
                frame_index=np.array([frame_index], dtype=np.int64),
                left_hand_joints=left_hand_joints,
                right_hand_joints=right_hand_joints,
            )
        )

    def send_command(self, start: bool, planner: bool = False) -> None:
        self._socket.send(build_command_message(start=start, stop=not start, planner=planner))
        time.sleep(0.01)

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
