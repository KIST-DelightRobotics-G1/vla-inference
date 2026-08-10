#!/usr/bin/env python3
"""Launch an Isaac-GR00T PolicyServer for the remote policy mode.

Thin wrapper over ``gr00t/eval/run_gr00t_server.py`` with KIST defaults
(UNITREE_G1_SONIC embodiment, port 5550). Run this on the GPU machine, then
start the runner with ``--policy.mode remote``.

Example:
    python scripts/run_policy_server.py \
        --model-path /path/to/checkpoint-XXXX --device cuda:0
"""

import tyro

from gr00t.eval.run_gr00t_server import ServerConfig, main

if __name__ == "__main__":
    config = tyro.cli(
        ServerConfig,
        default=ServerConfig(embodiment_tag="unitree_g1_sonic", port=5550),
    )
    main(config)
