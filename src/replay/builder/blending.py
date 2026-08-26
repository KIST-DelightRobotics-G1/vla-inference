"""The linear-ramp primitive shared by build and bracket."""

import numpy as np


def blend(start: np.ndarray, end: np.ndarray, n_ticks: int) -> np.ndarray:
    """Linear ramp from `start` toward `end` over `n_ticks` published ticks.

    Excludes `start` and includes `end` (`alpha = (i+1)/n`), matching the
    runner's initial-pose blend: the first published tick already moves, and
    the last one is exactly `end`.
    """
    if n_ticks <= 0:
        return np.empty((0, *np.shape(start)), dtype=np.float32)

    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    alpha = ((np.arange(n_ticks, dtype=np.float32) + 1.0) / n_ticks).reshape(-1, 1)
    return ((1.0 - alpha) * start.reshape(1, -1) + alpha * end.reshape(1, -1)).astype(np.float32)
