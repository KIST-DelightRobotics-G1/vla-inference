"""The chunking stage: ~2.5 Hz ActionChunks -> one action per 50 Hz tick.

    chunk_cursor.py  ChunkCursor — the newest chunk + a play cursor:
                     push() swaps a fresh prediction in (skipping its
                     in-flight staleness), step() hands out one tick,
                     holds briefly past the end, then goes silent so
                     gearsonic's LOST recovery takes over
    chunk_step.py    ChunkStep — the contract with the publisher: one
                     tick's token + hand targets
"""

from .chunk_cursor import ChunkCursor
from .chunk_step import ChunkStep

__all__ = ["ChunkCursor", "ChunkStep"]
