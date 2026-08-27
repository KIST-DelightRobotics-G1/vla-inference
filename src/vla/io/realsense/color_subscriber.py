"""CameraSubscriber — one view: DDS reader + H.264 decode thread -> latest Frame.

The decode thread exists because the data format forces it: H.264 delta
frames need their predecessors, so every sample must be decoded in arrival
order regardless of how often the consumer looks. The thread keeps the
newest decoded frame in a latest-value slot (a single tuple reference —
atomic to swap under the GIL, no lock needed), and `latest()` returns a
snapshot with its age so staleness stays the consumer's decision.

Resync rule (from the validated vla_old implementation): wait for the
first keyframe; a seq gap breaks the delta chain, so drop until the next
keyframe (ext-sensor-io sends periodic keyframes).
"""

import threading
import time
from dataclasses import dataclass, field

from common.cyclonedds.config import apply_network_interface

from .frame import Frame

DEFAULT_COLOR_TOPIC = "rt/kist/camera/color/h264"


def color_topic_for(name: str) -> str:
    """Per-camera topic, mirroring ext-sensor-io's kCameraColorTopicFor."""
    return f"rt/kist/camera/{name}/color/h264"


_frame_type_cache = None


def _compressed_color_frame_type():
    """CompressedColorFrame IdlStruct mirroring kist-ext-sensor-io's
    idl/kist_camera_frames.idl — keep in sync with that repo. Lazy so this
    module imports without cyclonedds."""
    global _frame_type_cache
    if _frame_type_cache is None:
        from cyclonedds.idl import IdlStruct
        import cyclonedds.idl.types as t

        @dataclass
        class CompressedColorFrame(IdlStruct, typename="kist_msgs::CompressedColorFrame"):
            width: t.uint32
            height: t.uint32
            seq: t.uint64
            stamp_ns: t.int64
            is_keyframe: bool
            frame_id: str
            data: t.sequence[t.uint8] = field(default_factory=list)

        _frame_type_cache = CompressedColorFrame
    return _frame_type_cache


class CameraSubscriber:
    """Latest decoded frame of one ext-sensor-io color stream.

    Usage:
        sub = CameraSubscriber("ego_view", topic=DEFAULT_COLOR_TOPIC)
        sub.start(domain_id=0)
        frame, age_s = sub.latest()   # (None, inf) until the first frame
        sub.stop()
    """

    def __init__(self, view: str, *, topic: str = DEFAULT_COLOR_TOPIC, history_depth: int = 32):
        self.view = view
        self.topic = topic
        self._history_depth = history_depth
        self._latest: tuple[Frame, float] | None = None  # (frame, monotonic recv time)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._participant = None
        self._reader = None
        self._codec = None
        self._synced = False  # waiting for the first keyframe
        self._last_seq: int | None = None

    def start(self, *, domain_id: int, network_interface: str = "") -> None:
        import av
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        apply_network_interface(network_interface)
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(self._history_depth))
        self._participant = DomainParticipant(domain_id)
        self._reader = DataReader(
            self._participant,
            Topic(self._participant, self.topic, _compressed_color_frame_type()),
            qos=qos,
        )
        self._codec = av.CodecContext.create("h264", "r")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._decode_loop, name=f"camera-rx-{self.view}", daemon=True
        )
        self._thread.start()
        print(f"[CameraSubscriber] domain {domain_id}: {self.topic} -> '{self.view}'")

    def latest(self) -> tuple[Frame | None, float]:
        """The newest decoded frame and its age in seconds ((None, inf) before
        the first frame). Snapshot semantics — never blocks, never decodes."""
        snapshot = self._latest
        if snapshot is None:
            return None, float("inf")
        frame, received = snapshot
        return frame, time.monotonic() - received

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._reader = None
        self._participant = None

    # ── decode thread ─────────────────────────────────────────────────────────

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            samples = self._reader.take(N=self._history_depth)
            if not samples:
                self._stop_event.wait(0.002)
                continue
            for sample in samples:
                rgb = self._decode_sample(sample)
                if rgb is not None:
                    self._latest = (
                        Frame(rgb=rgb, stamp_ns=int(sample.stamp_ns)),
                        time.monotonic(),
                    )

    def _decode_sample(self, sample):
        """Feed one CompressedColorFrame into the decoder; return newest image."""
        if not self._synced:
            if not sample.is_keyframe:
                return None
            self._synced = True
        elif self._last_seq is not None and sample.seq != self._last_seq + 1:
            # Lost frames -> the delta chain is broken; resync at a keyframe.
            if not sample.is_keyframe:
                self._synced = False
                return None
        self._last_seq = sample.seq

        newest = None
        try:
            for packet in self._codec.parse(bytes(sample.data)):
                for frame in self._codec.decode(packet):
                    newest = frame.to_ndarray(format="rgb24")
        except Exception as e:
            print(f"[CameraSubscriber:{self.view}] decode error ({e}); waiting for keyframe")
            self._synced = False
            return None
        return newest
