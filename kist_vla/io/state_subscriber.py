"""Robot-state subscriber (``g1_debug`` topic from the C++ deploy process).

Vendored from GR00T-WholeBodyControl
``gear_sonic/utils/data_collection/zmq_state_subscriber.py``. The publisher
is gearsonic's ZMQ output handler; each message is the topic string followed
by a msgpack map containing at least ``body_q`` (29,), ``left_hand_q`` (7,),
``right_hand_q`` (7,), and ``base_quat`` (4, wxyz).
"""

import msgpack
import numpy as np
import zmq

STATE_TOPIC = "g1_debug"
CONFIG_TOPIC = "robot_config"
DEFAULT_STATE_PORT = 5557


def _unpack_topic_msgpack(raw: bytes, topic: str) -> dict:
    """Strip the ZMQ topic prefix and decode the msgpack payload."""
    payload = raw[len(topic):]
    return msgpack.unpackb(payload, raw=False)


def _lists_to_numpy(data):
    if not isinstance(data, dict):
        return data
    result = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            result[key] = np.array(value)
        elif isinstance(value, dict):
            result[key] = _lists_to_numpy(value)
        else:
            result[key] = value
    return result


class StateSubscriber:
    """Non-blocking SUB on the robot-state topic; keeps only the latest message."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_STATE_PORT,
        topic: str = STATE_TOPIC,
    ):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._topic = topic
        self._msg = None
        print(f"[StateSubscriber] Connected to tcp://{host}:{port} (topic: {topic})")

    def _poll(self) -> None:
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        self._msg = _lists_to_numpy(_unpack_topic_msgpack(raw, self._topic))

    def get_msg(self, clear: bool = True):
        """Return the latest state message dict (or ``None``)."""
        self._poll()
        msg = self._msg
        if clear:
            self._msg = None
        return msg

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
