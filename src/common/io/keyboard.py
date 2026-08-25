"""Keyboard command channel (ZMQ PUB/SUB, single keystrokes or ``prompt:``).

Subscriber vendored from GR00T-WholeBodyControl
``gear_sonic/utils/data_collection/keyboard_subscriber.py``; the publisher is
ours (used by ``scripts/keyboard_publisher.py``). Decoupling the keyboard
from the runner keeps the runner headless — it can run unattended (e.g. as
a systemd service) with the operator console attached separately.
"""

import zmq

DEFAULT_KEYBOARD_PORT = 5580
PROMPT_PREFIX = "prompt:"


class KeyboardSubscriber:
    """Receives keyboard events on a SUB socket (non-blocking, latest only)."""

    def __init__(self, port: int = DEFAULT_KEYBOARD_PORT, host: str = "localhost"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{host}:{port}")
        self._data = None
        print(f"[KeyboardSubscriber] Connected to tcp://{host}:{port}")

    def read_msg(self) -> str | None:
        """Return the latest key press / prompt message (or ``None``)."""
        try:
            self._data = self._socket.recv_string(zmq.NOBLOCK)
        except zmq.Again:
            pass
        data = self._data
        self._data = None
        return data

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class KeyboardPublisher:
    """PUB side of the keyboard channel."""

    def __init__(self, port: int = DEFAULT_KEYBOARD_PORT, host: str = "*"):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.bind(f"tcp://{host}:{port}")
        print(f"[KeyboardPublisher] Bound to tcp://{host}:{port}")

    def send_key(self, key: str) -> None:
        self._socket.send_string(key)

    def send_prompt(self, prompt: str) -> None:
        self._socket.send_string(f"{PROMPT_PREFIX}{prompt}")

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()
