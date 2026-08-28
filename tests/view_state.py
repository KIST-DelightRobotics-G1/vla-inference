#!/usr/bin/env python3
"""Manually watch the unitree state streams (not a pytest test).

Subscribes with the real UnitreeStateReader (rt/lowstate +
rt/dex3/{left,right}/state) and prints one line per second — the same
quantities and per-stream ages the observation assembly will key off.
Needs unitree_sdk2py (and a robot / simulator publishing on the domain).

Usage:
    python tests/view_state.py              # domain from config/config.yaml
    python tests/view_state.py --domain 0
"""

import time
from dataclasses import dataclass

import numpy as np
import tyro

from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config
from vla.io.unitree import UnitreeStateReader


@dataclass
class Config:
    config: str = "config/config.yaml"
    """Network settings (dds.domain_id + the CycloneDDS transport XML)."""

    domain: int | None = None
    """DDS domain id override. Default: the config file's dds.domain_id."""


def _fmt(arr: np.ndarray, n: int = 4) -> str:
    head = ", ".join(f"{v:+.3f}" for v in arr[:n])
    return f"[{head}, ...]" if len(arr) > n else f"[{head}]"


def _age(age_s: float) -> str:
    return "  -  " if age_s == float("inf") else f"{age_s * 1000:4.0f}ms"


def main(config: Config) -> None:
    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    reader = UnitreeStateReader()
    reader.start(domain_id=domain)

    try:
        while True:
            state, state_age = reader.latest_state()
            left, left_age = reader.latest_left_hand()
            right, right_age = reader.latest_right_hand()

            if state is None:
                print("waiting for rt/lowstate ... (is the robot / simulator up?)")
            else:
                quat = state.imu_pelvis.quaternion
                print(
                    f"lowstate {_age(state_age)}  tick {state.tick:>8}  "
                    f"mode {state.mode_machine}  q {_fmt(state.q)}  "
                    f"quat [{quat[0]:+.3f}, {quat[1]:+.3f}, {quat[2]:+.3f}, {quat[3]:+.3f}]"
                )
                print(
                    f"  left  {_age(left_age)}  "
                    f"q {_fmt(left.q, 7) if left is not None else '-'}"
                )
                print(
                    f"  right {_age(right_age)}  "
                    f"q {_fmt(right.q, 7) if right is not None else '-'}"
                )
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()


if __name__ == "__main__":
    main(tyro.cli(Config))
