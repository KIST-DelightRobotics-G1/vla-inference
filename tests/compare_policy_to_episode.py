#!/usr/bin/env python3
"""Offline eval: does the policy reproduce a recorded episode's actions? (not pytest)

The decisive test between "the model is fine, the deployment scene is off"
and "the model (or our observation assembly) is broken" — no robot needed:

    episode frame t: (images from the mp4s, state from the parquet)
        -> SonicPolicy -> predicted motion_token[40]
                            vs
    episode joints t..t+39 -> the v1.1 SONIC encoder -> ground-truth tokens

On in-distribution inputs a healthy policy should track the ground truth
far better than the "hold the first token" baseline. If it doesn't, the
problem is on our side of the robot (model weights, prompt, or observation
formatting), not the scene.

Usage (inside the vla container):
    python tests/compare_policy_to_episode.py --episode 50
    python tests/compare_policy_to_episode.py --checkpoint /workspace/checkpoints/checkpoint-28000 \\
        --embodiment-tag unitree_g1_sonic --episode 50
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from replay.encoder import encoder as enc
from replay.io.parquet_io import read_joints
from vla.observation import Observation


@dataclass
class Config:
    checkpoint: str = "/workspace/checkpoints/checkpoint-4500"
    """Finetuned checkpoint directory."""

    embodiment_tag: str = "unitree_g1_sonic_3views"
    """Which of the checkpoint's embodiments to run."""

    data: str = "shared"
    """LeRobot dataset root (data/, videos/, meta/)."""

    episode: int = 50
    """Episode index to evaluate against."""

    prompt: str | None = None
    """Task instruction; default reads the episode's own task from
    meta/tasks.jsonl — the string the model was trained on."""

    samples: int = 8
    """How many timesteps to evaluate, spread over the episode."""

    encoder: str = "models/model_encoder.onnx"
    """SONIC encoder for the ground-truth tokens (must match the decoder
    generation the checkpoint was trained against — v1.1)."""

    device: str = "cuda:0"


def episode_paths(root: Path, episode: int) -> tuple[Path, dict[str, Path]]:
    name = f"episode_{episode:06d}"
    parquet = root / "data" / "chunk-000" / f"{name}.parquet"
    videos = {}
    for d in sorted((root / "videos" / "chunk-000").glob("observation.images.*")):
        view = d.name.split("observation.images.")[-1]
        videos[view] = d / f"{name}.mp4"
    return parquet, videos


def decode_frames(path: Path, ticks: list[int]) -> dict[int, np.ndarray]:
    """Decode the requested frame indices from one mp4 (single pass)."""
    import av

    wanted = set(ticks)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in wanted:
                frames[i] = frame.to_ndarray(format="rgb24")
            if len(frames) == len(wanted):
                break
    missing = wanted - frames.keys()
    if missing:
        raise SystemExit(f"{path}: frames {sorted(missing)} beyond the video")
    return frames


def main(config: Config) -> None:
    root = Path(config.data)
    parquet, videos = episode_paths(root, config.episode)

    prompt = config.prompt
    if prompt is None:
        tasks = [json.loads(l) for l in (root / "meta" / "tasks.jsonl").read_text().splitlines()]
        prompt = tasks[0]["task"]
    print(f"episode {config.episode}  prompt: {prompt!r}")

    # Ground truth: the episode's own joints through the SONIC encoder —
    # the tokens a perfect policy would predict.
    joints = read_joints(parquet)
    gt = enc.encode_joints(enc.load_onnx_encoder(config.encoder), joints.q, joints.base_quat)
    T = len(gt)

    # States for the model observation, exactly as the dataset stores them:
    # observation.state is already the 43-dim model slot layout, and
    # projected_gravity is its own column.
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["observation.state", "observation.projected_gravity"])
    states = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    gravity = np.stack(table["observation.projected_gravity"].to_numpy()).astype(np.float32)

    from common.g1_joints import split_state

    from vla.policy import SonicPolicy

    policy = SonicPolicy(
        config.checkpoint, device=config.device, embodiment_tag=config.embodiment_tag
    )
    views = policy.video_views
    for view in views:
        if view not in videos:
            raise SystemExit(f"checkpoint wants view {view!r} but the episode has {list(videos)}")

    horizon = 40
    ticks = np.linspace(0, T - horizon - 1, config.samples, dtype=int).tolist()
    frames = {view: decode_frames(videos[view], ticks) for view in views}

    print(f"\n{'tick':>6}  {'policy |Δ|':>12}  {'hold |Δ|':>10}   (mean abs token error over 40 steps; grid=0.0625)")
    policy_errs, hold_errs = [], []
    for t in ticks:
        state = {g: v.astype(np.float32) for g, v in split_state(states[t]).items()}
        state["projected_gravity"] = gravity[t]
        observation = Observation(
            video={view: frames[view][t] for view in views}, state=state, prompt=prompt
        )
        predicted = policy.predict(observation).motion_token  # (40, 64)
        truth = gt[t : t + horizon]

        policy_err = float(np.abs(predicted - truth).mean())
        hold_err = float(np.abs(np.tile(gt[t], (horizon, 1)) - truth).mean())
        policy_errs.append(policy_err)
        hold_errs.append(hold_err)
        print(f"{t:>6}  {policy_err:>12.4f}  {hold_err:>10.4f}")

    p, h = np.mean(policy_errs), np.mean(hold_errs)
    print(f"\nmean: policy {p:.4f} vs hold-baseline {h:.4f}  (ratio {p / max(h, 1e-9):.2f})")
    print(
        "verdict: "
        + (
            "policy tracks the demonstration (model + observation path healthy; "
            "if the robot stands still live, suspect the scene/prompt at deployment)"
            if p < h
            else "policy does NOT beat the hold baseline on in-distribution inputs — "
            "suspect the model weights, the prompt, or our observation formatting"
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
