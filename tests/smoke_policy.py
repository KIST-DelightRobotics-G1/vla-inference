#!/usr/bin/env python3
"""Load the checkpoint and run inference on a synthetic Observation (not pytest).

The policy-stage smoke test for the inference container: no robot, no
cameras — random images and a neutral standing state, the training prompt.
Proves the whole chain (vendored gr00t registration -> checkpoint load ->
processor -> backbone -> DiT -> decode -> ActionChunk) and measures the
inference latency the runner will have to live with.

Usage (inside the vla container; HF auth needed once for the gated
Cosmos-Reason2-2B backbone download):
    python tests/smoke_policy.py
    python tests/smoke_policy.py --checkpoint /workspace/checkpoints/checkpoint-4500 --runs 10

Prints load time, per-run latency (first run includes CUDA warmup), and the
decoded action ranges — motion tokens should land in a roughly [-1, 1]-ish
physical range (FSQ grid), hands within joint limits.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from common.g1_joints import JOINT_GROUP_INDICES
from vla.observation import Observation

# The training prompt — the checkpoint is single-task and conditions on this
# exact string (UNITREE_G1_SONIC_3VIEWS.md §5).
TRAINING_PROMPT = (
    "Open the right door of the refrigerator. Hook the yellow tip attached "
    "to your right hand under the door handle and pull."
)

VIEWS = ("ego_view", "left_wrist", "right_wrist")


@dataclass
class Config:
    checkpoint: str = "/workspace/checkpoints/checkpoint-4500"
    """Finetuned checkpoint directory (mounted by docker/run.sh)."""

    device: str = "cuda:0"
    """CUDA device for the model."""

    runs: int = 5
    """Timed inference runs after the warmup run."""

    seed: int = 0
    """Seed for the synthetic images."""

    baseline: str | None = None
    """Regression baseline .npz: saved if missing, compared against if
    present. Seeds torch too, so the flow-matching noise is fixed — the same
    environment must reproduce the prediction bit-for-bit; any refactor of
    the vendored gr00t core is checked against this."""


def make_observation(rng: np.random.Generator) -> Observation:
    video = {
        view: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for view in VIEWS
    }
    state = {
        group: np.zeros(len(idx), dtype=np.float32)
        for group, idx in JOINT_GROUP_INDICES.items()
    }
    state["projected_gravity"] = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return Observation(video=video, state=state, prompt=TRAINING_PROMPT)


def main(config: Config) -> None:
    rng = np.random.default_rng(config.seed)

    t0 = time.monotonic()
    from vla.policy import SonicPolicy

    policy = SonicPolicy(config.checkpoint, device=config.device)
    print(f"checkpoint loaded in {time.monotonic() - t0:.1f}s")

    chunk = None
    for run in range(config.runs + 1):
        observation = make_observation(rng)
        t0 = time.monotonic()
        chunk = policy.predict(observation)
        latency = time.monotonic() - t0
        label = "warmup" if run == 0 else f"run {run}"
        print(f"{label}: {latency * 1000:.0f} ms")

    print(f"\nmotion_token     {chunk.motion_token.shape}  "
          f"[{chunk.motion_token.min():+.3f}, {chunk.motion_token.max():+.3f}]")
    print(f"left_hand_joints {chunk.left_hand_joints.shape}  "
          f"[{chunk.left_hand_joints.min():+.3f}, {chunk.left_hand_joints.max():+.3f}]")
    print(f"right_hand_joints{chunk.right_hand_joints.shape}  "
          f"[{chunk.right_hand_joints.min():+.3f}, {chunk.right_hand_joints.max():+.3f}]")
    print(f"\nfirst motion_token step, dims 0..7: {np.round(chunk.motion_token[0, :8], 3)}")

    if config.baseline:
        import torch

        rng = np.random.default_rng(config.seed)
        torch.manual_seed(config.seed)
        chunk = policy.predict(make_observation(rng))
        arrays = {
            "motion_token": chunk.motion_token,
            "left_hand_joints": chunk.left_hand_joints,
            "right_hand_joints": chunk.right_hand_joints,
        }
        path = Path(config.baseline)
        if path.exists():
            reference = np.load(path)
            print()
            for name, value in arrays.items():
                diff = float(np.abs(reference[name] - value).max())
                verdict = "OK" if diff == 0.0 else "DIFFERS"
                print(f"baseline {name}: max|delta| = {diff:.3e}  {verdict}")
        else:
            np.savez(path, **arrays)
            print(f"\nbaseline saved: {path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
