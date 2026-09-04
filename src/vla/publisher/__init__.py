"""The publisher stage: a live ChunkCursor -> rt/kist/latent_action at 50 Hz.

    latent_action_streamer.py  LatentActionStreamer — owns the DDS channel
                               and the 50 Hz Tx thread (absolute deadlines);
                               a None tick publishes nothing, which is the
                               protocol for handing gearsonic its LOST
                               recovery
"""

from .latent_action_streamer import LatentActionStreamer

__all__ = ["LatentActionStreamer"]
