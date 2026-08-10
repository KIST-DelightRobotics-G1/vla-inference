#!/usr/bin/env python3
"""Entry point for the KIST VLA inference runner.

Examples:
    # Single-process (model in-process, default):
    python scripts/run_vla.py --policy.model-path /path/to/checkpoint-XXXX

    # Remote policy server (GPU on another machine):
    python scripts/run_vla.py --policy.mode remote --policy.host <gpu-box>
"""

import tyro

from kist_vla.config import RunnerConfig
from kist_vla.runner import main

if __name__ == "__main__":
    main(tyro.cli(RunnerConfig))
