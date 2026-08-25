#!/usr/bin/env python3
"""Replay a recorded kist-data-collector session on the robot (no model).

Thin entry point — the implementation lives in `replay.cli` (see its
docstring for the stream layout and safety notes). Equivalent to
`python -m replay`.

Usage:
    # Inspect a session — no DDS, no robot:
    python scripts/replay_session.py sessions/20260824_141530 --dry-run

    # Link check against the gearsonic probe (./build/vla_receiver_probe 42):
    python scripts/replay_session.py sessions/20260824_141530 --domain 42

    # On the real robot (ROBOT MOVES — hang it first, VR e-stop in reach):
    python scripts/replay_session.py sessions/20260824_141530 --domain 0
"""

import tyro

from replay.cli import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
