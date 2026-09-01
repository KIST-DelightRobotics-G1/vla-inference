#!/usr/bin/env python3
"""Derive the safe standing motion token from recorded episodes (not pytest).

The safe standing token is CHECKPOINT-SPECIFIC: gearsonic holds it when the
VLA stream is lost (kVlaSafeStandingToken) and replay brackets timelines
with it (DEFAULT_INITIAL_MOTION_TOKEN). Whenever the SONIC checkpoint
changes, re-derive it with this script and update BOTH constants.

Method: find the calmest stationary standing window in each recorded
episode, hold its mean pose constant over the encoder's 10 future frames
(zero velocities, identity heading-relative anchor), and encode with the
CURRENT models/model_encoder.onnx via the replay encoder's own packing.
Structural checks printed per episode: the token must land exactly on the
FSQ 1/16 grid, and nearby windows must agree to ~1 grid step.

The tokens differ between episodes (each session's stance differs slightly);
pick the one from the calmest window — and HARDWARE-VERIFY it (stream the
token, watch the robot stand) before trusting the recovery path.

Usage (host venv or replay container):
    python tests/derive_standing_token.py
    python tests/derive_standing_token.py --data shared/data/chunk-000 --episodes 8
"""

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from replay.encoder import encoder as enc
from replay.io import csv_io, parquet_io


@dataclass
class Config:
    data: str = "shared/data/chunk-000"
    """A collector session directory (contains lowstate.csv), or a directory
    of LeRobot episode parquets."""

    encoder: str = "models/model_encoder.onnx"
    """SONIC encoder ONNX — must be the checkpoint gearsonic decodes with."""

    episodes: int = 8
    """How many episodes to derive from (for cross-checking)."""

    window: int = 60
    """Stationarity window length in 50 Hz ticks."""


def main(config: Config) -> None:
    encoder = enc.load_onnx_encoder(config.encoder)
    if (Path(config.data) / "lowstate.csv").exists():
        paths = [config.data]  # one collector session
    else:
        paths = sorted(glob.glob(f"{config.data}/episode_*.parquet"))[: config.episodes]
    if not paths:
        raise SystemExit(f"no session/episodes under {config.data}")

    results = []
    for path in paths:
        reader = csv_io if (Path(path) / "lowstate.csv").exists() else parquet_io
        joints = reader.read_joints(path)
        q, base_quat = joints.q, joints.base_quat
        W = config.window
        # The calmest window near the episode start (a balancing humanoid
        # always sways a few hundredths of a radian — that IS standing).
        sway, start = min(
            (np.abs(q[i : i + W] - q[i : i + W].mean(0)).max(), i)
            for i in range(0, len(q) - W, 10)
        )
        q_bar = q[start : start + W].mean(0)
        quat_bar = base_quat[start : start + W].mean(0)
        quat_bar /= np.linalg.norm(quat_bar)

        # Constant pose over the 10 future frames = "hold this stance".
        T = (enc.NUM_FRAMES - 1) * enc.FRAME_STEP + 1
        token = enc.encode_joints(
            encoder, np.tile(q_bar, (T, 1)), np.tile(quat_bar, (T, 1))
        )[0]
        grid_err = float(np.abs(token * 16 - np.round(token * 16)).max())
        results.append((sway, path, token))
        print(f"{path.split('/')[-1]}: sway={sway:.3f} rad  FSQ grid err={grid_err:.1e}")
        if grid_err > 1e-4:
            print("  ^ WARNING: off the 1/16 grid — packing/layout is wrong for this encoder")

    sway, path, token = min(results, key=lambda r: r[0])
    print(f"\ncalmest stance: {path.split('/')[-1]} (sway {sway:.3f} rad)")
    print("token (numpy):")
    print(np.array2string(token, separator=", ", max_line_width=79))
    print("\nUpdate BOTH: src/common/config.py DEFAULT_INITIAL_MOTION_TOKEN and")
    print("gearsonic include/vla/vla_initial_pose.hpp kVlaSafeStandingToken,")
    print("then hardware-verify before trusting the recovery path.")


if __name__ == "__main__":
    main(tyro.cli(Config))
