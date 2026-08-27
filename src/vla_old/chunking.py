"""Action-chunk playback: latency compensation and inference scheduling.

The policy produces a chunk of ``action_horizon`` future steps every
``1/inference_rate`` seconds; the publish loop plays it back one step per
tick at ``action_publish_rate``. Because inference takes a substantial
fraction of the chunk duration, playback starts partway into each new chunk
(latency compensation) and holds the final step if a chunk runs dry before
the next one lands.

``calculate_latency_compensated_index`` and ``should_trigger_new_inference``
are vendored from GR00T-WholeBodyControl
``gear_sonic/utils/inference/vla_utils.py``; ``ActionChunkPlayer`` wraps them
into the state machine used by the runner.
"""

from typing import Any

import numpy as np


def calculate_latency_compensated_index(
    inference_delay: float, control_freq: float, action_horizon: int
) -> int:
    """Starting chunk index that skips steps gone stale during inference."""
    raw_index = np.round(inference_delay * control_freq)
    return int(np.clip(raw_index, 0, action_horizon - 1))


def should_trigger_new_inference(
    cached_chunk_exists: bool,
    inference_thread_running: bool,
    time_since_last_inference: float,
    inference_interval: float,
) -> bool:
    """Whether the publish loop should request a new inference this tick."""
    if not cached_chunk_exists:
        return True
    if inference_thread_running:
        return False
    return time_since_last_inference >= inference_interval


class ActionChunkPlayer:
    """Plays back the latest action chunk one step at a time.

    Holds at most one chunk (a dict of ``(T, D)`` — or ``(B, T, D)``, batch
    squeezed — arrays, all sharing horizon T). ``step()`` returns the per-key
    slice at the current index and advances, clamping at the final step so a
    stale chunk degrades to holding the last action.
    """

    def __init__(self, action_horizon: int):
        self.action_horizon = action_horizon
        self._chunk: dict[str, np.ndarray] | None = None
        self._index = 0

    @property
    def has_chunk(self) -> bool:
        return self._chunk is not None

    def clear(self) -> None:
        self._chunk = None
        self._index = 0

    def update(
        self,
        chunk: dict[str, Any],
        inference_delay: float,
        publish_rate: float,
    ) -> None:
        """Adopt a fresh chunk, starting at the latency-compensated index."""
        normalized = {}
        for key, value in chunk.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim == 3:  # (B, T, D) -> (T, D)
                arr = arr[0]
            normalized[key] = arr
        self._chunk = normalized
        self._index = calculate_latency_compensated_index(
            inference_delay, publish_rate, self.action_horizon
        )

    def step(self) -> dict[str, np.ndarray] | None:
        """Return the current step's per-key arrays and advance the cursor."""
        if self._chunk is None:
            return None

        out = {}
        for key, arr in self._chunk.items():
            if arr.ndim == 2:
                idx = min(self._index, arr.shape[0] - 1)
                out[key] = arr[idx]
            else:
                out[key] = arr

        self._index = min(self._index + 1, self.action_horizon - 1)
        return out
