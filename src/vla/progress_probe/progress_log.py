"""ProgressLog: one JSONL file per rollout under shared/probe.

Line-buffered append — every sample is one complete line the moment it is
written, so a Ctrl+C (or a crash) keeps everything up to that point. The
file lands in shared/, which docker/run.sh mounts from the host: the record
outlives the container.
"""

import json
import os
import time
from datetime import datetime

from .progress_state import Reading


class ProgressLog:
    """Append one JSONL row per prediction.

    Minimal schema (probe alone, no monitor): {"t", "progress", "latency_ms"}.
    Extended schema (probe + monitor): + raw, slope_per_s, stalled, plateaued,
    done, state — one row is enough to redraw a full progress timeline offline.

    Args:
        directory: where the file goes (created if missing).
    """

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"progress_{stamp}.jsonl")
        self._file = open(self.path, "a", buffering=1)  # line-buffered
        print(f"[ProgressLog] -> {self.path}")

    def append(
        self,
        progress: float | None,
        latency_ms: float,
        *,
        reading: Reading | None = None,
    ) -> None:
        entry: dict = {
            "t": round(time.time(), 3),
            "progress": None if progress is None else round(progress, 4),
            "latency_ms": round(latency_ms, 1),
        }
        if reading is not None:
            entry["raw"] = round(reading.raw, 4)
            entry["slope_per_s"] = (
                None if reading.slope_per_s is None else round(reading.slope_per_s, 4)
            )
            entry["stalled"] = reading.stalled
            entry["plateaued"] = reading.plateaued
            entry["done"] = reading.done
            entry["state"] = reading.state.value
        self._file.write(json.dumps(entry) + "\n")

    def close(self) -> None:
        self._file.close()
