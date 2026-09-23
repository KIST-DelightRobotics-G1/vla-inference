"""The progress-probe stage: DiT latent -> task progress in [0, 1] -> verdict.

    progress_probe.py  ProgressProbe — loads probe_succ3v.pt, hooks
                       action_head.vl_self_attention, scores each
                       prediction (fused normalize + linear on GPU)
    progress_state.py  ProgressMonitor — turns the raw score stream into
                       a Reading (running max, 5s slope, stalled/plateaued/
                       done) and a ProgressState verdict for SubtaskMachine
    progress_log.py    ProgressLog — one JSONL per rollout in shared/,
                       line-buffered so an interrupted run keeps its record

Optional: the runner only builds these when --probe is given.
"""

from .progress_log import ProgressLog
from .progress_probe import ProgressProbe
from .progress_state import ProgressMonitor, ProgressState, Reading

__all__ = [
    "ProgressLog",
    "ProgressMonitor",
    "ProgressProbe",
    "ProgressState",
    "Reading",
]
