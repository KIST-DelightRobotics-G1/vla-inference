"""SubtaskCmd.args: the per-subtask knobs the orchestrator may set.

The ICD calls `args` signature-order arguments, informational for the VLA
module. We read three `key=value` entries out of them; anything else is
left alone.

    probe=<name>     score with this probe .pt (file stem) instead of the
                     one the index picks for the instruction (probe_bank)
    verdict=off      do not score this subtask at all: no probe head is
                     selected, no DONE/STALLED verdict — the subtask ends
                     by its timeout, and that end is DONE (not FAILED).
                     For tasks without a fitted probe, or whose end is a
                     fixed duration rather than a visible state change.
    timeout=<s>      this subtask's step timeout in seconds (overrides the
                     runner's --step-timeout-s; 0 disables for this one)

Unknown keys and malformed values are ignored, never fatal — a typo in
the orchestrator's plan must not drop the subtask.
"""

from dataclasses import dataclass
from typing import Sequence

from ..progress_probe.probe_bank import override_from_args

VERDICT_KEY = "verdict="
TIMEOUT_KEY = "timeout="


@dataclass(frozen=True)
class CmdArgs:
    probe: str | None = None     # probe=<name>
    verdict: bool = True         # False on verdict=off
    timeout_s: float | None = None  # timeout=<s>, None = runner default


def parse(args: Sequence[str]) -> CmdArgs:
    verdict = True
    timeout_s: float | None = None
    for a in args:
        if a.startswith(VERDICT_KEY):
            verdict = a[len(VERDICT_KEY):].strip().lower() not in ("off", "false", "0", "no")
        elif a.startswith(TIMEOUT_KEY):
            try:
                timeout_s = float(a[len(TIMEOUT_KEY):].strip())
            except ValueError:
                pass
    return CmdArgs(probe=override_from_args(args), verdict=verdict, timeout_s=timeout_s)
