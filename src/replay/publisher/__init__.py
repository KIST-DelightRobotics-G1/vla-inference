"""The execution stage: put a finished ActionStream on the wire at 50 Hz.

    latent_action_publisher.py
        LatentActionPublisher — owns the DDS channel and the Tx worker
        thread (kist-ext-sensor-io transmitter pattern; the caller's main
        thread does lifecycle only), publishing one LatentActionStep per
        20 ms tick on rt/kist/latent_action. `publish_via` is the
        channel-injected loop core for tests and manual writers.

The only stage that touches the wire: everything upstream (io, aligner,
encoder, builder) is pure data handling, and the input `ActionStream` is
the builder's gate-approved publish plan — holding one is the proof it
went through `bracket_timeline`.
"""

from .latent_action_publisher import LatentActionPublisher, publish_via

__all__ = ["LatentActionPublisher", "publish_via"]
