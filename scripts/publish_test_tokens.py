#!/usr/bin/env python3
"""Publish test latent-action tokens over DDS (no model needed).

Link-check tool for the gearsonic receiver: streams the known-safe standing
pose token (DEFAULT_INITIAL_MOTION_TOKEN, hands open) at the control rate,
exactly like the runner would.

Usage:
    # Against the gearsonic probe (any domain, no robot):
    python scripts/publish_test_tokens.py --domain 42

    # Against the real control loop (ROBOT MOVES — air-hang first!):
    python scripts/publish_test_tokens.py --domain 0 --duration 10

WARNING: the standing-pose token is specific to the SONIC checkpoint used in
training (see src/common/config.py). With a different SONIC checkpoint on the
gearsonic side it produces a different, possibly unsafe pose.
"""

import time
from dataclasses import dataclass

import numpy as np
import tyro

from common.config import DEFAULT_INITIAL_MOTION_TOKEN
from common.g1_joints import OPEN_HAND_Q
from common.cyclonedds.kist_msgs_writer import KistMsgsWriter


@dataclass
class Config:
    domain: int = 0
    """DDS domain id (must match the receiver)."""

    rate: float = 50.0
    """Publish rate (Hz)."""

    duration: float = 30.0
    """How long to stream (seconds)."""

    send_start_command: bool = False
    """Also send a WbcCommand(start) before streaming."""


def main(config: Config) -> None:
    writer = KistMsgsWriter(domain_id=config.domain)
    time.sleep(1.0)  # discovery

    if config.send_start_command:
        writer.send_command(start=True, planner=False)
        print("Sent WbcCommand(start)")

    token = DEFAULT_INITIAL_MOTION_TOKEN
    period = 1.0 / config.rate
    n_steps = int(config.duration * config.rate)
    print(f"Streaming standing-pose token: {config.rate:.0f} Hz x {config.duration:.0f}s "
          f"on domain {config.domain} (Ctrl+C to stop)")

    try:
        for i in range(n_steps):
            t0 = time.monotonic()
            writer.send_latent_action(
                motion_token=token,
                frame_index=i,
                left_hand_joints=OPEN_HAND_Q,
                right_hand_joints=OPEN_HAND_Q,
            )
            if i % 100 == 0:
                print(f"  frame {i}/{n_steps}")
            remaining = period - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        writer.close()
        print("Done — the receiver will see the stream go stale "
              "(gearsonic drops to damping after 500ms in VLA mode).")


if __name__ == "__main__":
    main(tyro.cli(Config))
