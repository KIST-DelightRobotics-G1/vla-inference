"""LatentAction publisher: put an ActionStream on rt/kist/latent_action at 50 Hz.

Follows the kist-ext-sensor-io transmitter pattern (ColorPublisher): the
publisher owns its DDS channel *and* its Tx thread — `start()` opens the
channel and spawns the worker, `stop()` ends it and closes the channel, and
the caller's main thread does nothing but lifecycle (`wait()` until the
finite stream is done). `publish_via` is the channel-injected loop core for
tests and manual writers — the `start_channel` + one-shot analogue.

Takes a finished, gate-approved publish plan (`ActionStream` — the type is
the proof it went through `bracket_timeline`) and does nothing but timing:
one `LatentActionStep` per 20 ms tick.
"""

import threading
import time

# The topic constant is a plain string in the wire-contract module —
# importing it does not pull in cyclonedds (that stays lazy inside start()).
from common.cyclonedds.kist_msgs import LATENT_ACTION_TOPIC

from ..constants import CONTROL_DT_NS
from ..builder import ActionStream


def publish_via(
    writer, stream: ActionStream, *, stop_event: threading.Event | None = None
) -> int:
    """Publish the stream over a given writer on an absolute 50 Hz schedule.

    Absolute deadlines (not sleep(period - elapsed)) so a slow tick does not
    push the whole trajectory late: the replay's timing IS the recorded
    motion's timing. Late ticks are counted and reported — a tick beyond
    gearsonic's 500 ms staleness threshold would end the VLA session.

    Returns the number of ticks actually published (< len(stream) when
    `stop_event` was set mid-stream).
    """
    period = CONTROL_DT_NS / 1e9
    total = len(stream)
    published = 0
    late = 0
    worst_late = 0.0
    start = time.monotonic()

    for i, step in enumerate(stream):
        if stop_event is not None and stop_event.is_set():
            print(f"Publish stopped at tick {published}/{total}")
            break

        deadline = start + (i + 1) * period
        writer.send_latent_action(
            motion_token=step.token_state,
            frame_index=step.frame_index,
            left_hand_joints=step.left_hand_joints,
            right_hand_joints=step.right_hand_joints,
        )
        published += 1
        if i % 250 == 0:
            print(f"  tick {i}/{total}  ({i * period:.1f}s / {total * period:.1f}s)")

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            late += 1
            worst_late = max(worst_late, -remaining)

    elapsed = time.monotonic() - start
    print(f"Published {published} ticks in {elapsed:.2f}s (expected {total * period:.2f}s)")
    if late:
        print(f"WARNING: {late} tick(s) missed their deadline, worst overrun {worst_late * 1e3:.1f}ms")
    return published


class LatentActionPublisher:
    """Tx assembly for the latent-action stream: owns the channel + thread.

    Lifecycle (mirroring ext-sensor-io's ColorPublisher):

        pub = LatentActionPublisher()
        pub.start(stream, domain_id=0)
        pub.wait()      # main thread just waits; Ctrl+C interrupts here
        pub.stop()      # idempotent: joins the thread, closes the channel

    The stream is finite, so the worker ends on its own after the last tick;
    `stop()` also ends it early (the mid-stream stop is safe: gearsonic sees
    the stream go stale after 500 ms and runs its LOST recovery).
    """

    def __init__(self) -> None:
        self._writer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(
        self,
        stream: ActionStream,
        *,
        domain_id: int,
        topic: str = LATENT_ACTION_TOPIC,
        writer=None,
    ) -> None:
        """Open the DDS channel and run the Tx worker off `stream`.

        `writer` injects a ready channel (tests / manual writers) — then no DDS
        entity is created and `stop()` still closes it.
        """
        if self._thread is not None:
            raise RuntimeError("publisher already started")

        # Discovery settle is only needed for a channel we just opened; an
        # injected writer (tests / manual) is assumed ready.
        self._discovery_wait = writer is None
        if writer is None:
            from common.cyclonedds.kist_msgs_writer import KistMsgsWriter

            writer = KistMsgsWriter(domain_id=domain_id, action_topic=topic)
            print(f"Publishing {topic} @ 50 Hz on DDS domain {domain_id}")
        self._writer = writer

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(stream,), name="latent-action-tx", daemon=True
        )
        self._thread.start()

    def _run(self, stream: ActionStream) -> None:
        # Discovery settle happens on the Tx thread — start() stays instant.
        if self._discovery_wait:
            print("Waiting 1s for DDS discovery...")
            time.sleep(1.0)
        publish_via(self._writer, stream, stop_event=self._stop_event)

    def wait(self) -> None:
        """Block until the worker finishes the stream (Ctrl+C-interruptible)."""
        while self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)

    def stop(self) -> None:
        """End the worker (early if mid-stream) and close the channel."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
