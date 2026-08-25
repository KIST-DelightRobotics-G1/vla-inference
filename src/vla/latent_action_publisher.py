"""LatentAction publisher for the VLA runner: owns the DDS channel + Tx thread.

Same transmitter pattern as replay's publisher (ext-sensor-io ColorPublisher
style): `start()` opens the channel and spawns the Tx worker, `stop()` ends
it and closes the channel, and the caller's main thread does lifecycle only.

Unlike replay there is no finite stream to iterate — what to send is decided
live (chunk playback, operator state). So the Tx thread drives a `tick`
callback once per period, and the callback sends zero or more messages back
through `send` / `send_command`. The publisher owns the cadence; the
callback owns the content.
"""

import threading
import time

from common.cyclonedds.config import apply_network_interface
from common.cyclonedds.kist_msgs import LATENT_ACTION_TOPIC


class LatentActionPublisher:
    """Tx assembly for the runner's latent-action stream: channel + thread."""

    def __init__(self) -> None:
        self._writer = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(
        self,
        tick,
        *,
        rate_hz: int,
        domain_id: int,
        network_interface: str = "",
        topic: str = LATENT_ACTION_TOPIC,
        verbose_timing: bool = False,
        writer=None,
    ) -> None:
        """Open the DDS channel and drive `tick` at `rate_hz` on the Tx thread.

        `writer` injects a ready channel (tests) — then no DDS entity is
        created and `stop()` still closes it.
        """
        if self._thread is not None:
            raise RuntimeError("publisher already started")

        if writer is None:
            from common.cyclonedds.kist_msgs_writer import KistMsgsWriter

            apply_network_interface(network_interface)
            writer = KistMsgsWriter(domain_id=domain_id, action_topic=topic)
            print(f"Publishing {topic} @ {rate_hz} Hz on DDS domain {domain_id}")
        self._writer = writer

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(tick, rate_hz, verbose_timing),
            name="latent-action-tx", daemon=True,
        )
        self._thread.start()

    # The callback publishes through these — the writer never leaks out.
    def send(self, motion_token, frame_index, left_hand_joints, right_hand_joints) -> None:
        self._writer.send_latent_action(
            motion_token=motion_token,
            frame_index=frame_index,
            left_hand_joints=left_hand_joints,
            right_hand_joints=right_hand_joints,
        )

    def send_command(self, start: bool, planner: bool = False) -> None:
        self._writer.send_command(start=start, planner=planner)

    def _run(self, tick, rate_hz: int, verbose_timing: bool) -> None:
        period = 1.0 / rate_hz
        while not self._stop_event.is_set():
            t_start = time.monotonic()
            try:
                tick()
            except Exception as e:
                print(f"Error in Tx tick: {e}")
                import traceback

                traceback.print_exc()

            elapsed = time.monotonic() - t_start
            if verbose_timing or elapsed > period:
                print(f"[timing] tick took {elapsed * 1000:.1f}ms (budget {period * 1000:.0f}ms)")
            remaining = period - elapsed
            if remaining > 0:
                # Event.wait doubles as an interruptible sleep: stop() takes
                # effect within one tick even mid-sleep.
                self._stop_event.wait(remaining)

    def stop(self) -> None:
        """End the Tx thread and close the channel (idempotent)."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
