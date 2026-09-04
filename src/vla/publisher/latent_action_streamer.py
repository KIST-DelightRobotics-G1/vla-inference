"""LatentActionStreamer: put a live ChunkCursor on rt/kist/latent_action at 50 Hz.

The vla counterpart of replay's LatentActionPublisher: the same Tx assembly
(owns the DDS channel and its thread, ext-sensor-io transmitter pattern) and
the same absolute 20 ms deadlines, but pulling from a live ChunkCursor
instead of iterating a finite ActionStream — the stream has no end, only
`stop()`.

A tick where the cursor yields None publishes nothing: silence IS the
protocol — gearsonic marks the stream LOST after 500 ms and runs its
verified recovery (blend to safe standing). frame_index counts published
ticks, matching what gearsonic's arbiter sees as stream continuity.
"""

import threading
import time

from common.cyclonedds.kist_msgs import LATENT_ACTION_TOPIC

from ..chunking import ChunkCursor

CONTROL_DT_NS = 20_000_000  # 50 Hz, the SONIC control tick


class LatentActionStreamer:
    """Tx assembly for the live latent-action stream.

    Lifecycle:

        streamer = LatentActionStreamer()
        streamer.start(cursor, domain_id=0)
        ...                       # inference loop keeps cursor.push()-ing
        streamer.stop()           # joins the thread, closes the channel

    `writer` injects a ready channel (tests / manual writers) — then no DDS
    entity is created and `stop()` still closes it.
    """

    def __init__(self) -> None:
        self._writer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._published = 0
        self._late = 0

    def start(
        self,
        cursor: ChunkCursor,
        *,
        domain_id: int,
        topic: str = LATENT_ACTION_TOPIC,
        writer=None,
    ) -> None:
        if self._thread is not None:
            raise RuntimeError("streamer already started")

        # Discovery settle is only needed for a channel we just opened; an
        # injected writer (tests / manual) is assumed ready.
        self._discovery_wait = writer is None
        if writer is None:
            from common.cyclonedds.kist_msgs_writer import KistMsgsWriter

            writer = KistMsgsWriter(domain_id=domain_id, action_topic=topic)
            print(f"Streaming {topic} @ 50 Hz on DDS domain {domain_id}")
        self._writer = writer

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(cursor,), name="latent-action-tx", daemon=True
        )
        self._thread.start()

    def _run(self, cursor: ChunkCursor) -> None:
        # Discovery settle happens on the Tx thread — start() stays instant.
        if self._discovery_wait:
            print("Waiting 1s for DDS discovery...")
            time.sleep(1.0)

        # Absolute deadlines (not sleep(period - elapsed)) so a slow tick
        # does not push the schedule late; a tick near gearsonic's 500 ms
        # staleness threshold would end the VLA session.
        period = CONTROL_DT_NS / 1e9
        start = time.monotonic()
        tick = 0
        while not self._stop_event.is_set():
            step = cursor.step()
            if step is not None:
                self._writer.send_latent_action(
                    motion_token=step.motion_token,
                    frame_index=self._published,
                    left_hand_joints=step.left_hand_joints,
                    right_hand_joints=step.right_hand_joints,
                )
                self._published += 1

            tick += 1
            remaining = start + tick * period - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                self._late += 1

    @property
    def published(self) -> int:
        return self._published

    @property
    def late(self) -> int:
        return self._late

    def stop(self) -> None:
        """End the worker and close the channel (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._late:
            print(f"WARNING: {self._late} tick(s) missed their 20 ms deadline")
