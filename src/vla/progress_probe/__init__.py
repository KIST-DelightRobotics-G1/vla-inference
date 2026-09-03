"""The progress-probe stage: DiT latent -> task progress in [0, 1].

    progress_probe.py  ProgressProbe — loads probe_succ3v.pt, hooks
                       action_head.vl_self_attention, scores each
                       prediction (one dot product; the VLA forward
                       that runs anyway is the feature extractor)
    progress_log.py    ProgressLog — one JSONL per rollout in shared/,
                       line-buffered so an interrupted run keeps its record

Optional: the runner only builds these when --probe is given.
"""

from .progress_log import ProgressLog
from .progress_probe import ProgressProbe

__all__ = ["ProgressLog", "ProgressProbe"]
