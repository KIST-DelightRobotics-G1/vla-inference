"""Which probe .pt scores this subtask? Pure selection logic, no torch, no I/O.

A probe is fitted on one (checkpoint, prompt) pair, so the head that scores
a subtask must be the one fitted on that subtask's instruction. The runner
keeps an index {fit prompt -> .pt path} (shared/probe/probes.json) and
`pick()` turns it plus the live instruction into one path or None — None
means "no probe for this prompt": no score, no verdict (RUNNING until
timeout / cortex), which beats scoring with a mismatched head.

Override: an orchestrator cmd may carry ``args=["probe=<name>"]`` to force a
.pt by file stem — the experiment hook for comparing two fits of the same
prompt (scripts/send_subtask.py --args probe=probe_pick_v2).
"""

import os
from dataclasses import dataclass
from typing import Sequence

OVERRIDE_KEY = "probe="


@dataclass(frozen=True)
class Pick:
    path: str | None      # the chosen .pt path, or None
    reason: str           # one line for the log: why this one (or none)


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def override_from_args(args: Sequence[str]) -> str | None:
    """'probe=<name>' in a SubtaskCmd's args -> <name> (file stem, '.pt' ok)."""
    for a in args:
        if a.startswith(OVERRIDE_KEY):
            name = a[len(OVERRIDE_KEY):].strip()
            return name[:-3] if name.endswith(".pt") else name
    return None


def pick(index: Sequence[tuple[str, str]], instruction: str, override: str | None = None) -> Pick:
    """Choose the .pt for `instruction` from `index` = [(fit prompt, path), ...].

    1. override by file stem wins (or fails loudly: None, never a fallback);
    2. else the entry whose prompt equals the instruction exactly;
    3. else None.
    """
    names = [_stem(p) for _, p in index]
    if override:
        for _, path in index:
            if _stem(path) == override:
                return Pick(path, f"override probe={override}")
        return Pick(None, f"override probe={override!r} not in index {names}")
    for prompt, path in index:
        if prompt == instruction:
            return Pick(path, f"prompt match -> {_stem(path)}")
    return Pick(None, f"no probe fitted on {instruction!r} (index: {names})")
