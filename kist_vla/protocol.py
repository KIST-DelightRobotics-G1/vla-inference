"""Latent protocol v4 — the ZMQ wire format between this service and the
kist-gearsonic-inference C++ control loop.

Message layout: ``[topic bytes][1280-byte JSON header][concatenated binary]``.
The header is ``{"v": version, "endian": "le", "count": 1, "fields": [...]}``
padded with NUL bytes to exactly HEADER_SIZE.

Vendored from GR00T-WholeBodyControl
``gear_sonic/utils/teleop/zmq/zmq_planner_sender.py`` (pack_pose_message,
build_command_message) and ``gear_sonic/scripts/run_vla_inference.py``
(pack_latent_action_message). Byte-compatible with the reference — the C++
side parses this exact format. ``unpack_message`` is our own addition for
tests and debugging.
"""

import json
import struct

import numpy as np

HEADER_SIZE = 1280

POSE_TOPIC = "pose"
COMMAND_TOPIC = "command"

_DTYPE_STR_TO_NP = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "bool": np.bool_,
}


def _build_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header = {
        "v": version,
        "endian": "le",
        "count": count,
        "fields": fields,
    }
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    return header_json.ljust(HEADER_SIZE, b"\x00")


def pack_pose_message(pose_data: dict, topic: str = POSE_TOPIC, version: int = 4) -> bytes:
    """Pack a dict of numpy arrays into a latent-protocol message.

    Args:
        pose_data: Field name -> numpy array. Unsupported float dtypes are
            cast to float32.
        topic: Topic prefix string.
        version: Protocol version (gearsonic latent actions use v4).

    Returns:
        Packed message bytes ready for ``socket.send()``.
    """
    fields = []
    binary_data = []

    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)

        fields.append({"name": key, "dtype": dtype_str, "shape": list(value.shape)})

        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))

        binary_data.append(value.tobytes())

    header_bytes = _build_header(fields, version=version, count=1)
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray | None = None,
    right_hand_joints: np.ndarray | None = None,
) -> bytes:
    """Pack one motion-token action step for the C++ control loop (v4).

    Args:
        motion_token: (64,) or (1, 64) SONIC latent token.
        frame_index: scalar or (1,) int64 playback frame counter.
        left_hand_joints: (7,) or (1, 7) Dex motor order, optional.
        right_hand_joints: (7,) or (1, 7) Dex motor order, optional.

    Returns:
        Packed message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    for name, joints in (
        ("left_hand_joints", left_hand_joints),
        ("right_hand_joints", right_hand_joints),
    ):
        if joints is None:
            continue
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim == 1:
            if joints.shape[0] != 7:
                raise ValueError(f"{name} must have shape [7], got {joints.shape}")
            joints = joints.reshape(1, 7)
        pose_data[name] = joints

    return pack_pose_message(pose_data, topic=POSE_TOPIC, version=4)


def build_command_message(
    start: bool, stop: bool, planner: bool, delta_heading: float | None = None
) -> bytes:
    """Build a 'command' topic message controlling the C++ loop lifecycle."""
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("B", 1 if start else 0),
            struct.pack("B", 1 if stop else 0),
            struct.pack("B", 1 if planner else 0),
        )
    )

    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))

    header = _build_header(fields, version=1, count=1)
    return COMMAND_TOPIC.encode("utf-8") + header + payload


def unpack_message(message: bytes, topic: str) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse a latent-protocol message back into (header, field arrays).

    Inverse of ``pack_pose_message``/``build_command_message`` for the given
    topic. Used by tests and debugging tools; the C++ side has its own parser.
    """
    topic_bytes = topic.encode("utf-8")
    if not message.startswith(topic_bytes):
        raise ValueError(f"Message does not start with topic {topic!r}")

    header_start = len(topic_bytes)
    header_raw = message[header_start : header_start + HEADER_SIZE]
    header = json.loads(header_raw.rstrip(b"\x00").decode("utf-8"))

    fields = {}
    offset = header_start + HEADER_SIZE
    for field in header["fields"]:
        dtype = np.dtype(_DTYPE_STR_TO_NP[field["dtype"]]).newbyteorder("<")
        shape = tuple(field["shape"])
        nbytes = dtype.itemsize * int(np.prod(shape))
        fields[field["name"]] = np.frombuffer(
            message[offset : offset + nbytes], dtype=dtype
        ).reshape(shape)
        offset += nbytes

    return header, fields
