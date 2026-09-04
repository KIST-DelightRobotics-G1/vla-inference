"""ColorSubscriber — one view: DDS reader + H.264 decode thread -> latest ColorFrame.

The decode thread exists because the data format forces it: H.264 delta
frames need their predecessors, so every sample must be decoded in arrival
order regardless of how often the consumer looks. The thread keeps the
newest decoded frame in a latest-value slot (a single tuple reference —
atomic to swap under the GIL, no lock needed), and `latest()` returns a
snapshot with its age so staleness stays the consumer's decision.

Resync rule (validated on the real cameras): wait for the
first keyframe; a seq gap breaks the delta chain, so drop until the next
keyframe (ext-sensor-io sends periodic keyframes).
"""

import threading
import time
from dataclasses import dataclass, field

from .color_frame import ColorFrame

DEFAULT_COLOR_TOPIC = "rt/kist/camera/color/h264"

# Reader queue depth (KeepLast N), same value and rationale as the C++
# receiver (ext-sensor-io color_subscriber.cpp): deep enough (~1s at 30fps)
# to absorb arrival bursts without dropping at the reader; the downstream
# slot is latest-wins, so a deeper queue adds no consumer latency.
_HISTORY_DEPTH = 30


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


class ColorSubscriber:
    """Latest decoded frame of one ext-sensor-io color stream.

    Usage:
        participant = DomainParticipant(domain_id)   # ONE per process
        sub = ColorSubscriber("ego_view", topic=DEFAULT_COLOR_TOPIC)
        sub.start(participant=participant)
        frame, age_s = sub.latest()   # (None, inf) until the first frame
        sub.stop()

    The participant is injected, not owned: a process opens ONE
    DomainParticipant and every source attaches its readers to it (the
    ChannelFactory convention on the C++ side) — one participant on the
    bus instead of one per source.
    """

    def __init__(self, view: str, *, topic: str = DEFAULT_COLOR_TOPIC):
        self.view = view
        self.topic = topic
        # Wire diagnostics (monotonic counters, read from any thread):
        # received = samples taken off DDS; lost = frames a seq jump skipped
        # over (receive-side loss); resyncs = delta-chain breaks that forced
        # a wait for the next keyframe.
        self.received = 0
        self.lost = 0
        self.resyncs = 0
        self._latest: tuple[ColorFrame, float] | None = None  # (frame, monotonic recv time)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._reader = None
        self._codec = None
        self._synced = False  # waiting for the first keyframe
        self._last_seq: int | None = None

    def start(self, *, participant) -> None:
        import av
        from cyclonedds.core import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic

        # BestEffort: matches any writer reliability, latest-value semantics —
        # the same stance as every state reader on this bus. A lost sample
        # costs decode until the next keyframe (see the counters); if that
        # ever needs retransmission instead, revisit together with the tx
        # writer's QoS.
        qos = Qos(Policy.Reliability.BestEffort, Policy.History.KeepLast(_HISTORY_DEPTH))
        self._reader = DataReader(
            participant,
            Topic(participant, self.topic, _compressed_color_frame_type()),
            qos=qos,
        )
        self._codec = av.CodecContext.create("h264", "r")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._decode_loop, name=f"color-rx-{self.view}", daemon=True
        )
        self._thread.start()
        print(f"[ColorSubscriber] {self.topic} -> '{self.view}'")

    def latest(self) -> tuple[ColorFrame | None, float]:
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

    # ── decode thread ─────────────────────────────────────────────────────────

    def _decode_loop(self) -> None:
        while not self._stop_event.is_set():
            samples = self._reader.take(N=_HISTORY_DEPTH)
            if not samples:
                self._stop_event.wait(0.002)
                continue
            for sample in samples:
                rgb = self._decode_sample(sample)
                if rgb is not None:
                    self._latest = (
                        ColorFrame(rgb=rgb, stamp_ns=int(sample.stamp_ns)),
                        time.monotonic(),
                    )

    def _decode_sample(self, sample):
        """Feed one CompressedColorFrame into the decoder; return newest image."""
        self.received += 1
        if self._last_seq is not None and sample.seq > self._last_seq + 1:
            self.lost += int(sample.seq - self._last_seq - 1)
        if not self._synced:
            if not sample.is_keyframe:
                self._last_seq = sample.seq
                return None
            self._synced = True
        elif self._last_seq is not None and sample.seq != self._last_seq + 1:
            # Lost frames -> the delta chain is broken; resync at a keyframe.
            if not sample.is_keyframe:
                self._synced = False
                self.resyncs += 1
                self._last_seq = sample.seq
                return None
        self._last_seq = sample.seq

        newest = None
        try:
            for packet in self._codec.parse(bytes(sample.data)):
                for frame in self._codec.decode(packet):
                    newest = frame.to_ndarray(format="rgb24")
        except Exception as e:
            print(f"[ColorSubscriber:{self.view}] decode error ({e}); waiting for keyframe")
            self._synced = False
            return None
        return newest
