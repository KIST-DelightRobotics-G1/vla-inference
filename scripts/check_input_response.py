#!/usr/bin/env python3
"""Does the policy actually look at its inputs, or is it emitting boilerplate?

``smoke_test_policy.py`` feeds one synthetic observation and checks the reply is
well-formed. A policy whose vision path is silently broken — wrong image key,
wrong preprocessing, image never reaching the backbone — passes that test, and
passes the DDS and runner stages too. It only shows up on the robot, as "why is
it moving without regard to the scene", by which point the suspect list is huge.

This script perturbs one input at a time and asks whether the action follows.
The catch is that the action head is a diffusion model: identical inputs give
different samples every call. So a changed output proves nothing on its own —
it has to be compared against how far the output moves when nothing changes.

The statistic is the distance between the means of two independent sample sets.
The null distribution for it is built the same way, from repeated independent
sample sets of the SAME baseline observation. That keeps both sides of the
comparison on one scale — averaging N samples shrinks noise by sqrt(N), so
comparing a distance-between-means against a raw sample-to-sample spread would
be biased toward declaring every input ignored.

Usage:

    python scripts/check_input_response.py --model-path ~/vla_data/checkpoint-18000
    python scripts/check_input_response.py --model-path ... --image /path/to/photo.jpg

Exits non-zero when any perturbation fails to move the action.
"""

import itertools
import sys
from dataclasses import dataclass
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
    device: str = "cuda:0"

    prompt: str = "raise your right arm if you see a banana"
    """Baseline prompt. Use the training instruction."""

    alt_prompt: str = "stand still and do nothing"
    """Contrasting prompt for the language-path probe."""

    images: tuple[str, ...] = ()
    """Real photos (anything cv2 reads), one variant each. Synthetic gray /
    white / noise are all equally out of distribution, so they cannot tell a
    dead vision path from a model that correctly sees nothing in any of them.
    Two real scenes can: the model has to separate them or the path is suspect."""

    image_size: tuple[int, int] = (480, 640)
    """(height, width) of the synthetic images — the collection resolution."""

    samples: int = 8
    """Diffusion samples per variant. More samples tighten the noise floor."""

    margin: float = 3.0
    """A perturbation counts as felt when its distance from the baseline mean
    exceeds the largest null distance by this factor."""

    null_replicates: int = 4
    """Extra independent baseline sample sets used to build the null. Each one
    costs `samples` inference calls."""

    seed: int = 0


def _gray(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def build_variants(config: Config) -> dict[str, tuple[str, dict, dict, str]]:
    """Named (channel, camera_msg, state_msg, prompt) tuples, each one
    perturbation off baseline so a failure names its own input path."""
    height, width = config.image_size
    rng = np.random.default_rng(config.seed)

    zero_state = {
        "body_q": np.zeros(29, dtype=np.float32),
        "left_hand_q": np.zeros(7, dtype=np.float32),
        "right_hand_q": np.zeros(7, dtype=np.float32),
        "base_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }
    # A plausible non-zero posture: arms and waist off neutral, pelvis pitched.
    posed_state = {
        "body_q": np.linspace(-0.4, 0.4, 29).astype(np.float32),
        "left_hand_q": np.full(7, 0.3, dtype=np.float32),
        "right_hand_q": np.full(7, -0.3, dtype=np.float32),
        "base_quat": np.array([0.966, 0.0, 0.259, 0.0]),  # ~30 deg about y
    }

    def cam(image: np.ndarray) -> dict:
        return {"timestamps": {"ego_view": 0.0}, "images": {"ego_view": image}}

    flat = cam(_gray(height, width, 128))
    variants = {
        "baseline (gray, zero state)": ("baseline", flat, zero_state, config.prompt),
        "prompt changed": ("language", flat, zero_state, config.alt_prompt),
        # Synthetic images are all equally out of distribution, so a policy that
        # sees nothing in any of them is behaving correctly. They are kept as
        # diagnostics; only a real scene can vindicate the vision path.
        "image -> white": ("vision", cam(_gray(height, width, 250)), zero_state, config.prompt),
        "image -> noise": (
            "vision",
            cam(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)),
            zero_state,
            config.prompt,
        ),
        "state -> posed": ("state", flat, posed_state, config.prompt),
    }

    for path in config.images:
        import cv2

        raw = cv2.imread(path)
        if raw is None:
            raise SystemExit(f"could not read --images entry {path}")
        # cv2 gives BGR; the sensor path delivers RGB.
        resized = cv2.resize(raw, (width, height))[:, :, ::-1].copy()
        variants[f"image -> {path.split('/')[-1]}"] = (
            "vision",
            cam(resized),
            zero_state,
            config.prompt,
        )

    return variants


def sample_tokens(policy: Any, observation: dict, count: int) -> np.ndarray:
    """(count, 40*64) motion tokens for one fixed observation."""
    rows = []
    for _ in range(count):
        action, _ = policy.get_action(observation)
        chunk = {k.replace("action.", ""): v for k, v in action.items()}
        rows.append(np.asarray(chunk["motion_token"], dtype=np.float64).reshape(-1))
    return np.stack(rows)


def mean_pairwise_distance(rows: np.ndarray) -> float:
    if len(rows) < 2:
        return 0.0
    return float(
        np.mean([np.linalg.norm(a - b) for a, b in itertools.combinations(rows, 2)])
    )


def main(config: Config) -> None:
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    runner_config = RunnerConfig()
    builder = ObservationBuilder(language_key=runner_config.language_key)
    variants = build_variants(config)

    print(f"checkpoint: {config.model_path}")
    print(f"variants: {len(variants)}   samples each: {config.samples}")
    print("loading...", flush=True)
    policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
    )

    baseline_name = next(iter(variants))
    _, baseline_camera, baseline_state, baseline_prompt = variants[baseline_name]
    baseline_obs = builder.build(baseline_camera, baseline_state, baseline_prompt)
    if baseline_obs is None:
        raise SystemExit("ObservationBuilder returned None for the baseline")

    # Reference set, then independent replicates of the SAME observation. The
    # spread among those replicate means is exactly the null for the statistic
    # applied to the perturbations below.
    reference = sample_tokens(policy, baseline_obs, config.samples).mean(axis=0)
    print(f"  sampled {baseline_name} (reference)", flush=True)

    null_distances = []
    for i in range(config.null_replicates):
        replicate = sample_tokens(policy, baseline_obs, config.samples).mean(axis=0)
        null_distances.append(float(np.linalg.norm(replicate - reference)))
        print(f"  sampled baseline replicate {i + 1}", flush=True)

    null_max = max(null_distances)
    threshold = null_max * config.margin

    print("\nnull — same observation, independent sample sets")
    for i, distance in enumerate(null_distances, 1):
        print(f"  {distance:8.4f}   replicate {i}")
    print(f"  {null_max:8.4f}   << worst null")
    print(f"  {threshold:8.4f}   << threshold ({config.margin}x worst null)")

    print("\ndistance from baseline")
    best: dict[str, float] = {}
    for name, (channel, camera_msg, state_msg, prompt) in variants.items():
        if name == baseline_name:
            continue
        observation = builder.build(camera_msg, state_msg, prompt)
        if observation is None:
            raise SystemExit(f"ObservationBuilder returned None for variant {name!r}")
        rows = sample_tokens(policy, observation, config.samples)
        distance = float(np.linalg.norm(rows.mean(axis=0) - reference))
        ratio = distance / null_max if null_max else float("inf")
        best[channel] = max(best.get(channel, 0.0), ratio)
        print(
            f"  {distance:8.4f}  ({ratio:6.2f}x null)  "
            f"[{'FELT' if distance >= threshold else 'NOT FELT':8s}]  "
            f"{channel:8s}  {name}"
        )

    # Verdict per input path, not per variant: one perturbation of a channel
    # landing above threshold vindicates that channel. A synthetic image the
    # policy ignores says nothing, so it must not fail the vision path on its own.
    print(f"\nverdict per input path (best variant, threshold {config.margin}x null)")
    unproven = []
    for channel in ("vision", "language", "state"):
        if channel not in best:
            continue
        ratio = best[channel]
        proven = ratio >= config.margin
        if not proven:
            unproven.append(channel)
        print(f"  {ratio:6.2f}x  [{'REACHES MODEL' if proven else 'UNPROVEN':13s}]  {channel}")

    if "vision" in best and not any(
        name.startswith("image -> ") and variants[name][0] == "vision"
        and not name.endswith(("white", "noise"))
        for name in variants
    ):
        print(
            "\nnote: no real photo was supplied (--images). Synthetic gray/white/noise "
            "are all out of distribution, so they cannot vindicate the vision path."
        )

    if unproven:
        print(f"\nUNPROVEN — {', '.join(unproven)}: the action did not move beyond resampling noise.")
        print("Either that input does not reach the model, or the perturbation was too weak.")
    else:
        print("\nPASSED — every input path measurably moves the action.")
    sys.exit(1 if unproven else 0)


if __name__ == "__main__":
    main(tyro.cli(Config))
