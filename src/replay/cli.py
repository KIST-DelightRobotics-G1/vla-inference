"""Replay a recorded kist-data-collector session on the robot (no model).

Publishes the session's recorded motion tokens as `kist_msgs::LatentActionStep`
on `rt/kist/latent_action` at 50 Hz — the same stream the VLA runner produces,
so gearsonic's arbiter claims the robot for VLA and its PolicyDecoder drives
the whole body through the recorded latent trajectory. The decoder stays
closed-loop on live robot state: the robot balances itself, this is not an
open-loop joint playback.

The published stream is bracketed for safety:

    [lead-in]  safe standing token, so gearsonic claims VLA from a known pose
    [blend]    standing -> the session's first token
    [replay]   the recorded tokens, gaps blended over (see replay/builder/)
    [blend]    the session's last token -> standing
    [lead-out] safe standing token, then the publisher stops

Without the lead-out the stream would simply end mid-pose and gearsonic would
run its own LOST recovery 500 ms later (blend to standing, planner reseed,
back to the origin) — safe, but from wherever the episode happened to stop.

Usage:
    # Link check against the gearsonic probe (./build/vla_receiver_probe 42):
    python scripts/replay_session.py --path sessions/20260824_141530 --domain 42

    # On the real robot (ROBOT MOVES — hang it first, VR e-stop in reach):
    python scripts/replay_session.py --path sessions/20260824_141530 --domain 0

WARNING: the recorded tokens are latents of the SONIC checkpoint that was
running when the session was collected. Replaying them against a gearsonic
built on a *different* SONIC decoder checkpoint produces a different, possibly
unsafe motion — the latent spaces are not comparable.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

from common.config import (
    DEFAULT_INITIAL_LEFT_HAND_Q,
    DEFAULT_INITIAL_MOTION_TOKEN,
    DEFAULT_INITIAL_RIGHT_HAND_Q,
)
from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config

from .aligner import align_joints, align_tokens
from .constants import CONTROL_DT_NS
from .publisher import LatentActionPublisher
from .io import csv_io, parquet_io
from .builder import (
    ActionStream,
    CompressedGapError,
    Timeline,
    bracket_timeline,
    build_timeline,
)

# Fixed path, gearsonic-style: the GEAR-SONIC encoder paired with the decoder
# gearsonic runs (swap them together). The docker image downloads it at build.
ENCODER_ONNX = "models/model_encoder.onnx"


@dataclass
class Config:
    path: str
    """What to replay: a collector session directory (contains
    motion_token.csv), a LeRobot training-export episode parquet, or a
    LeRobot dataset root (contains meta/info.json — pass --episode too)."""

    episode: int | None = None
    """Episode index, when --path is a LeRobot dataset root."""

    joints: bool = False
    """Default publishes the episode's recorded motion tokens (latents of
    the collection-time SONIC checkpoint). --joints RE-ENCODES the recorded
    whole-body joints through the SONIC encoder at models/model_encoder.onnx
    (g1 mode) — checkpoint-portable. Parquet episodes encode observation.state;
    collector CSV sessions encode lowstate.csv q+dq on the token grid."""

    config: str = "config/config.yaml"
    """Network settings (dds: domain_id, cyclonedds_xml) — gearsonic-style.
    A missing file at this default path falls back to built-in defaults."""

    domain: int | None = None
    """DDS domain id override (must match the gearsonic receiver). Default:
    the config file's dds.domain_id."""

    max_gap_ticks: int = 25
    """Longest gap (in 20 ms ticks) to blend across. A larger gap is
    compressed to this many ticks, which ramps a real pose change over
    0.5 s — the run refuses to start unless --force is given."""

    lead_in_s: float = 1.5
    """Seconds of safe standing token before the replay, so gearsonic's
    arbiter claims VLA (200 ms freshness) from a known pose first."""

    lead_out_s: float = 1.5
    """Seconds of safe standing token after the replay, before the stream
    ends."""

    blend_s: float = 0.7
    """Seconds to ramp standing <-> the session's first/last token. Matches
    gearsonic's own handoff crossfade (ControlArbiter::kHandoffBlendTicks)."""

    force: bool = False
    """Publish even when a gap had to be compressed."""


def _require_encoder() -> None:
    if not Path(ENCODER_ONNX).exists():
        raise SystemExit(
            f"{ENCODER_ONNX} not found — the docker image bakes it in; on a "
            f"host checkout: wget -P models "
            f"https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1/model_encoder.onnx"
        )


def build_stream(timeline: Timeline, config: Config) -> ActionStream:
    """Bracket the replay with the standing lead-in/out and the two blends."""
    rate = 1e9 / CONTROL_DT_NS
    return bracket_timeline(
        timeline,
        DEFAULT_INITIAL_MOTION_TOKEN,
        DEFAULT_INITIAL_LEFT_HAND_Q,
        DEFAULT_INITIAL_RIGHT_HAND_Q,
        lead_in_ticks=round(config.lead_in_s * rate),
        lead_out_ticks=round(config.lead_out_s * rate),
        blend_ticks=round(config.blend_s * rate),
        force=config.force,
    )


def main(config: Config) -> None:
    path = Path(config.path)
    is_parquet = path.suffix == ".parquet" or (path / "meta" / "info.json").exists()

    if is_parquet:
        print(f"Episode: {path}" + (f" [{config.episode}]" if config.episode is not None else ""))
        if path.is_dir():
            if config.episode is None:
                raise SystemExit(f"{path} is a dataset root — pass --episode to pick one")
            path = parquet_io.resolve_episode_path(path, config.episode)
        tokens = parquet_io.read_tokens(path)
        joints = parquet_io.read_joints(path) if config.joints else None
    else:
        if config.episode is not None:
            raise SystemExit(
                f"--episode is a LeRobot-dataset option; {path} is a collector "
                f"session directory (motion_token.csv) and has no episodes"
            )
        print(f"Session: {path}")
        tokens = csv_io.read_tokens(path)
        joints = csv_io.read_joints(path) if config.joints else None

    # The align stage: every recorded side stream onto the token clock.
    aligned = align_tokens(tokens)

    if joints is not None:
        # The encoding stage: replace the recorded token values with ones
        # re-encoded from the joints (checkpoint portability).
        from .encoder import encode_tokens_from_joints

        _require_encoder()
        print(f"Re-encoding joints via {ENCODER_ONNX}")
        aligned = encode_tokens_from_joints(aligned, align_joints(tokens, joints), ENCODER_ONNX)

    timeline = build_timeline(aligned, max_hold_ticks=config.max_gap_ticks)
    try:
        stream = build_stream(timeline, config)
    except CompressedGapError as e:
        print(
            f"\nRefusing to publish: {e}.\n"
            f"Either raise --max-gap-ticks (now {config.max_gap_ticks}) so the transition "
            f"is spread over the whole gap instead, or pass --force if the faster "
            f"ramp is acceptable."
        )
        sys.exit(1)

    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    # Main thread = lifecycle only: the publisher owns the channel and the
    # Tx worker thread (ext-sensor-io transmitter pattern).
    print("THE ROBOT MOVES NOW — VR e-stop is A+B+X+Y held 1s.")
    publisher = LatentActionPublisher()
    publisher.start(stream, domain_id=domain)
    try:
        publisher.wait()
    except KeyboardInterrupt:
        print(
            "\nStopped by user mid-stream — gearsonic sees the stream go stale after 500ms "
            "and runs its LOST recovery (blend to standing, planner reseed, back to origin)."
        )
    finally:
        publisher.stop()
