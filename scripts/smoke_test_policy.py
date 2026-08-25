#!/usr/bin/env python3
"""Load a checkpoint and run one inference — no robot, no camera, no gearsonic.

The first thing to run against a new checkpoint. It answers, in order, the
questions a live run answers too late:

1. Did every parameter in the model actually get a weight from the checkpoint?
   ``AutoModel.from_pretrained`` is HF-default non-strict, so a renamed module
   leaves its parameters randomly initialized behind a warning — the policy
   loads, runs, and only the *tokens* are wrong. This is the failure this
   script exists to catch.
2. Does the checkpoint accept the observation this repo builds, and answer
   with the chunk shapes the runner and gearsonic expect?
3. Is ``|motion_token|`` inside ``action_bound``? The runner drops whole
   chunks above it, which on the robot looks like "the policy stopped".

Usage:

    python scripts/smoke_test_policy.py --model-path ~/vla_data/checkpoint-18000

Exits non-zero on any failed check, so it works as a preflight gate.

This box sources ROS 2 Humble globally, which puts Python 3.10 site-packages
on ``PYTHONPATH`` and breaks imports inside the 3.12 venv. Prefix with
``env -u PYTHONPATH`` if imports fail oddly.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from kist_vla.config import RunnerConfig
from kist_vla.observation import ObservationBuilder


@dataclass
class Config:
    model_path: str
    """Checkpoint directory (a UNITREE_G1_SONIC finetune)."""

    embodiment_tag: str = "unitree_g1_sonic"
    """Embodiment tag for the checkpoint."""

    device: str = "cuda:0"
    """Device to load the model on."""

    prompt: str = "demo"
    """Language prompt. Match the training instruction exactly — the
    checkpoint was finetuned on one phrasing and another is out of
    distribution."""

    image_size: tuple[int, int] = (480, 640)
    """Synthetic ego_view size as (height, width). Keep it at the native
    resolution used during data collection; resizing puts the model on a
    different preprocessing path than training."""

    repeats: int = 10
    """Timed inference calls after a discarded warmup call."""

    latency_ceiling: float = 0.8
    """Hard ceiling on max inference time (s). Above this the system does not
    hold together: latency compensation skips more steps than the horizon has,
    so a finished chunk has nothing left to play."""


class Checks:
    """Collects results so one run reports every problem it found."""

    def __init__(self) -> None:
        self.failed = 0

    def ok(self, label: str, detail: str = "") -> None:
        print(f"  [ OK ] {label}{': ' + detail if detail else ''}", flush=True)

    def fail(self, label: str, detail: str = "") -> None:
        self.failed += 1
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}", flush=True)

    def warn(self, label: str, detail: str = "") -> None:
        print(f"  [WARN] {label}{': ' + detail if detail else ''}", flush=True)

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        (self.ok if condition else self.fail)(label, detail)
        return condition


def _preview(keys: list[str], limit: int = 5) -> str:
    if not keys:
        return ""
    head = ", ".join(sorted(keys)[:limit])
    return f"{len(keys)} — {head}{', ...' if len(keys) > limit else ''}"


def checkpoint_weight_names(model_dir: Path) -> set[str] | None:
    """Parameter names stored in the checkpoint, or None if unreadable.

    Read from the safetensors shard index when present (header only, no
    tensor data), otherwise from the shard headers directly.
    """
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        try:
            return set(json.loads(index.read_text())["weight_map"])
        except (OSError, ValueError, KeyError):
            return None

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        return None
    try:
        from safetensors import safe_open

        names: set[str] = set()
        for shard in shards:
            with safe_open(shard, framework="pt") as handle:
                names.update(handle.keys())
        return names
    except Exception:
        return None


def load_policy(config: Config, checks: Checks) -> Any:
    print("\n1. checkpoint load")
    import torch

    from thirdparty.gr00t.policy.gr00t_policy import Gr00tPolicy

    checks.check(torch.cuda.is_available(), "CUDA available", f"torch {torch.__version__}")

    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )
    model = policy.model
    params = sum(p.numel() for p in model.parameters())
    checks.ok("model loaded", f"{params / 1e9:.2f}B params")

    # Record VRAM: the 3090's 24 GB is the constraint that decides whether this
    # box can host the policy at all.
    if torch.cuda.is_available():
        index = torch.device(config.device).index or 0
        free, total = torch.cuda.mem_get_info(index)
        checks.ok(
            "VRAM after load",
            f"{(total - free) / 2**30:.1f} / {total / 2**30:.1f} GiB used "
            f"(torch reserved {torch.cuda.memory_reserved(index) / 2**30:.1f} GiB)",
        )

    # Compare the checkpoint's weight names against the assembled model's
    # rather than trusting transformers' warnings. Parameters with no stored
    # weight are the dangerous case: silently random, and the model still runs.
    stored = checkpoint_weight_names(Path(config.model_path).expanduser())
    if stored is None:
        checks.warn(
            "weight coverage unverified",
            "no readable safetensors index — is the checkpoint fully downloaded?",
        )
        return policy

    parameters = {name for name, _ in model.named_parameters()}
    slots = set(model.state_dict())
    randomly_initialized = sorted(parameters - stored)
    unplaced = sorted(stored - slots)

    checks.check(
        not randomly_initialized,
        "every parameter got a checkpoint weight",
        _preview(randomly_initialized),
    )
    checks.check(
        not unplaced,
        "every checkpoint weight found a slot",
        _preview(unplaced),
    )
    if randomly_initialized or unplaced:
        checks.warn(
            "this is the code/checkpoint version mismatch signature",
            "the vendored thirdparty/gr00t must match the commit the checkpoint "
            "was finetuned with — see thirdparty/gr00t/VENDORED_FROM.md",
        )
    return policy


def build_observation(config: Config, runner_config: RunnerConfig) -> dict[str, Any]:
    """A neutral observation: mid-gray image, zero joints, upright pelvis."""
    height, width = config.image_size
    camera_msg = {
        "timestamps": {"ego_view": 0.0},
        "images": {"ego_view": np.full((height, width, 3), 128, dtype=np.uint8)},
    }
    state_msg = {
        "body_q": np.zeros(29, dtype=np.float32),
        "left_hand_q": np.zeros(7, dtype=np.float32),
        "right_hand_q": np.zeros(7, dtype=np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz, upright
    }
    observation = ObservationBuilder(language_key=runner_config.language_key).build(
        camera_msg, state_msg, config.prompt, log_errors=True
    )
    if observation is None:
        raise RuntimeError("ObservationBuilder returned None for a complete message pair")
    return observation


def check_inference(
    policy: Any, config: Config, runner_config: RunnerConfig, checks: Checks
) -> None:
    print("\n2. inference")
    observation = build_observation(config, runner_config)
    horizon = runner_config.action_horizon

    # First call pays CUDA/cuDNN autotuning; it is not representative.
    warmup_start = time.perf_counter()
    action, _info = policy.get_action(observation)
    warmup = time.perf_counter() - warmup_start

    durations: list[float] = []
    for _ in range(max(1, config.repeats)):
        start = time.perf_counter()
        action, _info = policy.get_action(observation)
        durations.append(time.perf_counter() - start)

    # The runner strips "action." prefixes; mirror that so both conventions work.
    chunk = {key.replace("action.", ""): value for key, value in action.items()}
    print(f"  keys: {sorted(chunk)}")

    expected_shapes = {
        "motion_token": (1, horizon, 64),
        "left_hand_joints": (1, horizon, 7),
        "right_hand_joints": (1, horizon, 7),
    }
    for name, shape in expected_shapes.items():
        if name not in chunk:
            checks.fail(f"{name} present")
            continue
        array = np.asarray(chunk[name])
        checks.check(array.shape == shape, f"{name} shape", f"{array.shape} vs {shape}")
        checks.check(bool(np.isfinite(array).all()), f"{name} finite")

    if "motion_token" in chunk:
        absmax = float(np.abs(np.asarray(chunk["motion_token"])).max())
        checks.check(
            absmax <= runner_config.action_bound,
            "|motion_token| within action bound",
            f"{absmax:.4f} vs {runner_config.action_bound}",
        )

    mean = sum(durations) / len(durations)
    worst = max(durations)
    budget = 1.0 / runner_config.inference_rate
    print(
        f"  latency over {len(durations)} calls: "
        f"mean {mean * 1000:.0f} ms, max {worst * 1000:.0f} ms, "
        f"min {min(durations) * 1000:.0f} ms  (warmup {warmup * 1000:.0f} ms, discarded)"
    )

    # Hard ceiling first: past it the runner has no steps left to play.
    checks.check(
        worst <= config.latency_ceiling,
        "max latency under the hard ceiling",
        f"{worst * 1000:.0f} ms vs {config.latency_ceiling * 1000:.0f} ms",
    )
    if worst > budget:
        checks.warn(
            "max latency over the chunk budget",
            f"{worst * 1000:.0f} ms > {budget * 1000:.0f} ms "
            f"({runner_config.inference_rate} Hz) — the runner replays stale chunks",
        )
    else:
        checks.ok("within the chunk budget", f"max {worst * 1000:.0f} ms <= {budget * 1000:.0f} ms")


def main(config: Config) -> None:
    runner_config = RunnerConfig()
    print(f"checkpoint: {config.model_path}")
    print(f"embodiment: {config.embodiment_tag}   prompt: {config.prompt!r}")

    checks = Checks()
    policy = load_policy(config, checks)
    check_inference(policy, config, runner_config, checks)

    print(f"\n{'FAILED' if checks.failed else 'PASSED'} — {checks.failed} failed check(s)")
    raise SystemExit(1 if checks.failed else 0)


if __name__ == "__main__":
    main(tyro.cli(Config))
