"""Camera frame client (gear_sonic sensor-server wire format).

Vendored (trimmed) from GR00T-WholeBodyControl
``gear_sonic/camera/sensor_server.py``/``composed_camera.py``
(SensorClient + ImageMessageSchema + ComposedCameraClientSensor). The server
publishes msgpack maps ``{"timestamps": {name: t}, "images": {name: jpeg}}``
where each image is raw JPEG bytes, a legacy base64 JPEG string, or an
msgpack_numpy-encoded array. Decoded images are RGB uint8 (H, W, 3).
"""

import base64
import time
from collections import deque
from typing import Any

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

DEFAULT_CAMERA_PORT = 5555


def _decode_image(value) -> np.ndarray:
    if isinstance(value, (bytes, bytearray)):
        mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        return mat[..., ::-1]  # BGR -> RGB
    if isinstance(value, str):
        return _decode_image(base64.b64decode(value))
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict) and (b"nd" in value or "nd" in value):
        return mnp.decode(value)
    return value


class CameraClient:
    """ZMQ SUB client for merged camera frames; keeps only the latest frame."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_CAMERA_PORT,
        staleness_warn_after: float = 0.1,
        staleness_warn_interval: float = 2.0,
    ):
        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, True)
        self._socket.setsockopt(zmq.RCVHWM, 3)
        self._socket.connect(f"tcp://{host}:{port}")

        self._latest: dict[str, Any] | None = None
        self._last_new_message_time: float | None = None
        self._last_staleness_warning_time = 0.0
        self._staleness_warn_after = staleness_warn_after
        self._staleness_warn_interval = staleness_warn_interval
        self._frame_intervals: deque = deque(maxlen=20)
        print(f"[CameraClient] Connected to tcp://{host}:{port}")

    def _receive_nonblocking(self, timeout_ms: int = 0):
        if self._socket.poll(timeout_ms):
            packed = self._socket.recv()
            return msgpack.unpackb(packed, object_hook=mnp.decode)
        return None

    def read(self) -> dict[str, Any] | None:
        """Return ``{"timestamps": {...}, "images": {name: RGB uint8}}`` or None.

        Non-blocking; reuses the previous frame (with a rate-limited warning)
        when no new message has arrived.
        """
        now = time.time()
        message = self._receive_nonblocking()

        if message is not None:
            images = {
                key: _decode_image(value)
                for key, value in message.get("images", {}).items()
            }
            self._latest = {
                "timestamps": message.get("timestamps", {}),
                "images": images,
            }
            if self._last_new_message_time is not None:
                self._frame_intervals.append(now - self._last_new_message_time)
            self._last_new_message_time = now
        elif self._latest is not None and self._last_new_message_time is not None:
            stale_for = now - self._last_new_message_time
            if (
                stale_for > self._staleness_warn_after
                and now - self._last_staleness_warning_time >= self._staleness_warn_interval
            ):
                print(
                    f"[CameraClient][WARNING] No new frame for {stale_for * 1000:.1f}ms; "
                    "reusing stale image. Check the camera server."
                )
                self._last_staleness_warning_time = now

        return self._latest

    def close(self) -> None:
        self._socket.close()
        self._ctx.term()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
