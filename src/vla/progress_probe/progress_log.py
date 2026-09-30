"""ProgressLog: one JSONL file per rollout under shared/probe.

Line-buffered append — every sample is one complete line the moment it is
written, so a Ctrl+C (or a crash) keeps everything up to that point. The
file lands in shared/, which docker/run.sh mounts from the host: the record
outlives the container.

File name: <task>_<checkpoint>_<probe>_<NN>_<YYYYMMDD_HHMM>.jsonl — the run's
identity first (what was picked, which model, which probe), a per-identity
run counter NN so successive runs of the same setup sort as 01, 02, ..., and
the start time last. tests/view_progress.py names its marks file from the
same stem (<stem>.marks.jsonl).
"""

import glob
import json
import os
import re
import time
from datetime import datetime

from .progress_state import Reading

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def slug(text: str, *, max_words: int = 5) -> str:
    """A filename-safe token from free text: lowercase, first words, [a-z0-9_-].

    "Reach out with the right hand and pick up the cucumber inside"
    -> "reach_out_with_the_right"
    """
    words = _UNSAFE.sub(" ", text).lower().split()
    return "_".join(words[:max_words]) or "run"


def run_stem(directory: str, task: str, checkpoint: str, probe: str) -> str:
    """<task>_<checkpoint>_<probe>_<NN>_<YYYYMMDD_HHMM>, NN = 1 + the runs of
    the same identity already in `directory`."""
    identity = f"{slug(task)}_{slug(checkpoint, max_words=8)}_{slug(probe, max_words=8)}"
    existing = glob.glob(os.path.join(directory, f"{identity}_[0-9][0-9]_*.jsonl"))
    runs = {
        m.group(1)
        for m in (re.search(r"_(\d\d)_\d{8}_\d{4}(?:\.marks)?\.jsonl$", f) for f in existing)
        if m
    }
    number = max((int(r) for r in runs), default=0) + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{identity}_{number:02d}_{stamp}"


class ProgressLog:
    """Append one JSONL row per prediction.

    Minimal schema (probe alone, no monitor): {"t", "progress", "latency_ms"}.
    Extended schema (probe + monitor): + raw, slope_per_s, stalled, plateaued,
    done, state — one row is enough to redraw a full progress timeline offline.

    Args:
        directory: where the file goes (created if missing).
        stem: file name without extension (see run_stem); None keeps the old
            progress_<YYYYMMDD_HHMMSS> name.
    """

    def __init__(self, directory: str, stem: str | None = None):
        os.makedirs(directory, exist_ok=True)
        if stem is None:
            stem = "progress_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"{stem}.jsonl")
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
