"""The progress-probe stage: DiT latent -> task progress in [0, 1] -> verdict.

    progress_probe.py  ProgressProbe — one hook on action_head.vl_self_attention,
                       a bank of ProbeHead (.pt per fitted prompt), scores each
                       prediction with the head selected for the live prompt
    probe_bank.py      pick() — which head for this instruction (torch-free)
    progress_state.py  ProgressMonitor — turns the raw score stream into
                       a Reading (running max, 5s slope, stalled/plateaued/
                       done) and a ProgressState verdict for SubtaskMachine
    progress_log.py    ProgressLog — one JSONL per rollout in shared/,
                       line-buffered so an interrupted run keeps its record

Optional: the runner only builds these when --probe is given.

ProgressProbe is resolved lazily: it imports torch, and SubtaskMachine
imports ProgressState from here — the pure-logic side (state machine,
monitor, tests, tests/drive_progress_probe.py) must load without torch.
"""

from .probe_bank import override_from_args, pick
from .progress_log import ProgressLog
from .progress_state import ProgressMonitor, ProgressState, Reading

__all__ = [
    "ProbeHead",
    "ProgressLog",
    "override_from_args",
    "pick",
    "ProgressMonitor",
    "ProgressProbe",
    "ProgressState",
    "Reading",
]


def __getattr__(name: str):
    if name in ("ProgressProbe", "ProbeHead"):
        from . import progress_probe

        return getattr(progress_probe, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
