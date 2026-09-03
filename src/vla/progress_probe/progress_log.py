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


class ProgressLog:
    """Append {"t", "progress", "latency_ms"} per prediction.

    Args:
        directory: where the file goes (created if missing).
    """

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"progress_{stamp}.jsonl")
        self._file = open(self.path, "a", buffering=1)  # line-buffered
        print(f"[ProgressLog] -> {self.path}")

    def append(self, progress: float | None, latency_ms: float) -> None:
        self._file.write(
            json.dumps(
                {
                    "t": round(time.time(), 3),
                    "progress": None if progress is None else round(progress, 4),
                    "latency_ms": round(latency_ms, 1),
                }
            )
            + "\n"
        )

    def close(self) -> None:
        self._file.close()
