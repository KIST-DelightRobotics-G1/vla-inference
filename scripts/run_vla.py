#!/usr/bin/env python3
"""Run GR00T VLA inference on the robot: cameras + state -> 50 Hz latents.

Thin entry point — the implementation lives in `vla.runner` (see its
docstring for the pipeline and safety behavior). Inference container only
(torch + the baked backbone).

Usage (inside the vla container, gearsonic + ext-sensor-io running):
    python scripts/run_vla.py
    python scripts/run_vla.py --checkpoint /workspace/checkpoints/checkpoint-28000 \\
        --embodiment-tag unitree_g1_sonic --prompt "<its training prompt>"

ROBOT MOVES — hang it first, VR e-stop in reach. Ctrl+C ends the stream;
gearsonic blends to the safe standing pose via its LOST recovery.
"""

import tyro

from vla.runner import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
