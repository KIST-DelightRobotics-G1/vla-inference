"""Camera source subscribing to kist-ext-sensor-io's DDS color stream.

kist-ext-sensor-io owns the RealSense (a RealSense can only be opened by
one process) and publishes H.264 Annex-B NAL units as
``kist_msgs::CompressedColorFrame`` on ``rt/kist/camera[/<name>]/color/h264``.
This source subscribes, decodes with PyAV, and exposes the newest frame
under the observation key the embodiment expects (``ego_view``).

The IdlStruct below mirrors kist-ext-sensor-io ``idl/kist_camera_frames.idl``
— keep in sync with that repo.

Decode notes: H.264 delta frames need their predecessors, so the reader
keeps a short history (KeepLast) and every taken sample is fed to the
decoder in order; ``read()`` only *returns* the newest decoded frame.
After a subscription gap the decoder resynchronizes at the next keyframe
(``is_keyframe`` flag; ext-sensor-io sends periodic keyframes). The
decoder-side parser holds the newest NAL until the next packet's boundary
arrives, so decoded frames trail the wire by one frame (~33 ms at 30 fps)
— well inside the VLA observation latency budget.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_COLOR_TOPIC = "rt/kist/camera/color/h264"


def color_topic_for(name: str) -> str:
    """Per-camera topic, mirroring ext-sensor-io's kCameraColorTopicFor."""
    return f"rt/kist/camera/{name}/color/h264"


_types_cache = None


def get_camera_types():
    """CompressedColorFrame IdlStruct mirroring kist-ext-sensor-io (cached)."""
    global _types_cache
    if _types_cache is None:
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

        _types_cache = CompressedColorFrame
    return _types_cache


class DdsCameraSource:
    """Latest-frame camera source over DDS (H.264 decode via PyAV)."""

    def __init__(
        self,
        domain_id: int = 0,
        topic: str = DEFAULT_COLOR_TOPIC,
        image_key: str = "ego_view",
        history_depth: int = 32,
        staleness_warn_after: float = 0.5,
        staleness_warn_interval: float = 2.0,
    ):
        import av
        from cyclonedds.core import Policy, Qos
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        frame_type = get_camera_types()
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(history_depth))
        self._participant = DomainParticipant(domain_id)
        self._reader = DataReader(
            self._participant, Topic(self._participant, topic, frame_type), qos=qos
        )

        self._codec = av.CodecContext.create("h264", "r")
        self._synced = False  # waiting for the first keyframe
        self._last_seq: int | None = None

        self.image_key = image_key
        self._latest: dict[str, Any] | None = None
        self._last_new_frame_time: float | None = None
        self._last_staleness_warning_time = 0.0
        self._staleness_warn_after = staleness_warn_after
        self._staleness_warn_interval = staleness_warn_interval
        print(f"[DdsCameraSource] domain {domain_id}: {topic} -> '{image_key}'")

    def _decode_sample(self, sample) -> np.ndarray | None:
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
            print(f"[DdsCameraSource] decode error ({e}); waiting for next keyframe")
            self._synced = False
            return None
        return newest

    def _drain(self):
        """Yield every queued sample in arrival order (take() caps at N)."""
        while True:
            samples = self._reader.take(N=32)
            if not samples:
                return
            yield from samples

    def read(self) -> dict[str, Any] | None:
        """Return ``{"timestamps": {...}, "images": {key: RGB uint8}}`` or None."""
        now = time.time()
        newest_image = None
        newest_stamp = None

        for sample in self._drain():  # in-order batch since last poll
            image = self._decode_sample(sample)
            if image is not None:
                newest_image = image
                newest_stamp = sample.stamp_ns / 1e9

        if newest_image is not None:
            self._latest = {
                "timestamps": {self.image_key: newest_stamp},
                "images": {self.image_key: newest_image},
            }
            self._last_new_frame_time = now
        elif self._latest is not None and self._last_new_frame_time is not None:
            stale_for = now - self._last_new_frame_time
            if (
                stale_for > self._staleness_warn_after
                and now - self._last_staleness_warning_time >= self._staleness_warn_interval
            ):
                print(
                    f"[DdsCameraSource][WARNING] No new frame for {stale_for * 1000:.0f}ms; "
                    "reusing stale image. Check kist-ext-sensor-io."
                )
                self._last_staleness_warning_time = now

        return self._latest

    def close(self) -> None:
        del self._reader
        del self._participant
