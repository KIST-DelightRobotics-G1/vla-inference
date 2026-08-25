"""Alignment and linear-ramp primitives shared by build and bracket."""

import numpy as np


def align_by_recv_ns(
    target_recv_ns: np.ndarray, src_recv_ns: np.ndarray, src_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Newest source row at or before each target time (backward merge_asof).

    Returns `(values, before_first)` where `values[i]` is the source row
    aligned to `target_recv_ns[i]`, and `before_first[i]` marks targets that
    precede the first source row — those clamp to the first row (the only
    honest choice without extrapolating) and the caller may warn about them.
    """
    src_recv_ns = np.asarray(src_recv_ns)
    if len(src_recv_ns) == 0:
        raise ValueError("no source rows to align against")

    idx = np.searchsorted(src_recv_ns, np.asarray(target_recv_ns), side="right") - 1
    before_first = idx < 0
    return src_values[np.clip(idx, 0, len(src_recv_ns) - 1)], before_first


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
