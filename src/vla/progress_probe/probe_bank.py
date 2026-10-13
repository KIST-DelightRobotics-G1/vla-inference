"""Which probe .pt scores this subtask? Pure selection logic, no torch.

A probe is fitted on one (checkpoint, prompt) pair, so the head that scores
a subtask must be the one fitted on that subtask's instruction. The runner
keeps an index (shared/probe/<checkpoint>/probes.json) — one entry per
sentence the orchestrator may send, saying how a subtask with that
instruction ENDS:

    "<fit prompt>": "probe_open.pt"    a head scores it: DONE / STALLED by verdict
    "<fit prompt>": {"pt": "probe_open.pt", "done_threshold": 0.8}
                                       same, with this sentence's own
                                       ProgressMonitor thresholds (any of
                                       MONITOR_KEYS; the rest = the CLI values)
    "<fit prompt>": 8                  no head: unscored, DONE 8 s after start

`load_index()` splits the file into those maps; `pick()` turns the heads
plus the live instruction into one path or None — None means "no probe
for this prompt": no score, no verdict, the subtask ends by its unscored
timeout (its number here, else the runner's --unscored-timeout-s). That
beats scoring with a mismatched head.

Override: an orchestrator cmd may carry ``args=["probe=<name>"]`` to force a
.pt by file stem — the experiment hook for comparing two fits of the same
prompt (scripts/send_subtask.py --args probe=probe_pick_v2).
"""

import json
import os
from dataclasses import dataclass
from typing import Sequence

OVERRIDE_KEY = "probe="

# ProgressMonitor keyword arguments a probes.json object entry may set per
# sentence — the same names as the runner's CLI flags.
MONITOR_KEYS = frozenset({
    "done_threshold", "stuck_value_threshold", "slope_stuck_threshold",
    "window_s", "stall_min_elapsed_s", "stall_hold_s",
})


@dataclass(frozen=True)
class Index:
    heads: dict[str, str]                 # prompt -> absolute .pt path
    timeouts: dict[str, float]            # prompt -> unscored timeout (s)
    thresholds: dict[str, dict[str, float]]  # prompt -> ProgressMonitor kwargs


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def load_index(path: str) -> Index:
    """probes.json -> Index. A string value is a .pt relative to the json's
    directory; a number is the unscored timeout in seconds; an object is
    {"pt": <.pt>, <monitor key>: <number>, ...}. Anything else is a config
    error, raised here so a typo fails at startup, not mid-run."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object {{prompt: .pt | seconds | {{pt, thresholds}}}}")
    base = os.path.dirname(os.path.abspath(path))
    heads: dict[str, str] = {}
    timeouts: dict[str, float] = {}
    thresholds: dict[str, dict[str, float]] = {}
    for prompt, value in raw.items():
        if isinstance(value, str):
            heads[prompt] = os.path.join(base, value)
        elif _number(value):
            timeouts[prompt] = float(value)
        elif isinstance(value, dict):
            pt = value.get("pt")
            if not isinstance(pt, str):
                raise ValueError(f"{path}: {prompt!r}: object entry needs \"pt\": \"<name>.pt\"")
            bad = sorted(set(value) - MONITOR_KEYS - {"pt"})
            if bad:
                raise ValueError(f"{path}: {prompt!r}: unknown keys {bad}; allowed: pt, "
                                 f"{', '.join(sorted(MONITOR_KEYS))}")
            heads[prompt] = os.path.join(base, pt)
            over = {k: v for k, v in value.items() if k != "pt"}
            for k, v in over.items():
                if not _number(v):
                    raise ValueError(f"{path}: {prompt!r}: {k} must be a number, got {v!r}")
            if over:
                thresholds[prompt] = {k: float(v) for k, v in over.items()}
        else:
            raise ValueError(f"{path}: {prompt!r}: value must be a .pt name, seconds, or "
                             f"{{\"pt\": ..., thresholds}}, got {value!r}")
    return Index(heads, timeouts, thresholds)


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
