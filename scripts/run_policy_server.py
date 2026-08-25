#!/usr/bin/env python3
"""Launch a GR00T PolicyServer for the remote policy mode.

Serves the vendored ``Gr00tPolicy`` over ZMQ (REQ/REP) with KIST defaults
(UNITREE_G1_SONIC embodiment, port 5550). Run this on the GPU machine, then
start the runner with ``--policy.mode remote --policy.host <this-box>``.

Replaces upstream ``gr00t/eval/run_gr00t_server.py``, which is not vendored
(its ReplayPolicy / dataset paths belong to the training stack).

Example:
    python scripts/run_policy_server.py \
        --model-path /path/to/checkpoint-XXXX --device cuda:0
"""

import os
from dataclasses import dataclass

import tyro


@dataclass
class ServerConfig:
    model_path: str
    """Checkpoint directory (a UNITREE_G1_SONIC finetune)."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for the checkpoint."""

    device: str = "cuda:0"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Bind address."""

    port: int = 5550
    """Port (the runner's --policy.port default)."""

    strict: bool = True
    """Enforce strict observation/action validation in Gr00tPolicy."""


def main(config: ServerConfig) -> None:
    from thirdparty.gr00t.policy.gr00t_policy import Gr00tPolicy
    from thirdparty.gr00t.policy.server_client import PolicyServer

    if not os.path.isdir(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    print("Starting GR00T inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Listening on: {config.host}:{config.port}")

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
    )
    with PolicyServer(policy=policy, host=config.host, port=config.port) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
