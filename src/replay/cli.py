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
    [replay]   the recorded tokens, gaps blended over (see replay/timeline/)
    [blend]    the session's last token -> standing
    [lead-out] safe standing token, then the publisher stops

Without the lead-out the stream would simply end mid-pose and gearsonic would
run its own LOST recovery 500 ms later (blend to standing, planner reseed,
back to the origin) — safe, but from wherever the episode happened to stop.

Usage:
    # Link check against the gearsonic probe (./build/vla_receiver_probe 42):
    python scripts/replay_session.py sessions/20260824_141530 --domain 42

    # On the real robot (ROBOT MOVES — hang it first, VR e-stop in reach):
    python scripts/replay_session.py sessions/20260824_141530 --domain 0

WARNING: the recorded tokens are latents of the SONIC checkpoint that was
running when the session was collected. Replaying them against a gearsonic
built on a *different* SONIC decoder checkpoint produces a different, possibly
unsafe motion — the latent spaces are not comparable.
"""

import sys
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

from common.config import DEFAULT_INITIAL_MOTION_TOKEN
from common.cyclonedds.config import load_dds_config
from common.g1_joints import OPEN_HAND_Q

from .constants import ARBITER_TELEOP, CONTROL_DT_NS
from .latent_action_publisher import LatentActionPublisher
from .session import load_episode, load_reencoded_episode, load_session
from .timeline import ActionStream, CompressedGapError, ReplayTimeline, bracket_timeline

# Fixed path, gearsonic-style: the GEAR-SONIC encoder paired with the decoder
# gearsonic runs (swap them together). The docker image downloads it at build.
ENCODER_ONNX = "models/model_encoder.onnx"


@dataclass
class Config:
    session: str
    """What to replay: a collector session directory (contains
    motion_token.csv), a LeRobot training-export episode parquet, or a
    LeRobot dataset root (contains meta/info.json — pass --episode too)."""

    episode: int | None = None
    """Episode index, when --session is a LeRobot dataset root."""

    tokens_from: Literal["recorded", "joints"] = "recorded"
    """'recorded' publishes the episode's recorded motion tokens (latents of
    the collection-time SONIC checkpoint). 'joints' RE-ENCODES the recorded
    whole-body joints through the SONIC encoder at models/model_encoder.onnx
    (g1 mode) — checkpoint-portable, parquet episodes only."""

    joint_source: Literal["state", "wbc"] = "state"
    """--tokens-from joints only: 'state' encodes the measured joints (the
    motion that actually happened), 'wbc' the commanded targets."""

    config: str = "config/config.yaml"
    """Network settings (dds: domain_id, network_interface) — gearsonic-style.
    A missing file at this default path falls back to built-in defaults."""

    domain: int | None = None
    """DDS domain id override (must match the gearsonic receiver). Default:
    the config file's dds.domain_id."""

    teleop_only: bool = False
    """Replay only arbiter_mode==1 (teleop demonstration) ticks — what the
    training export keeps. Default replays every recorded tick."""

    hand_source: Literal["cmd", "state", "none"] = "cmd"
    """Hand targets: 'cmd' (hand_cmd_{side}.csv, the commanded targets),
    'state' (hand_{side}.csv, measured), or 'none' (open hands)."""

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


def build_stream(timeline: ReplayTimeline, config: Config) -> ActionStream:
    """Bracket the replay with the standing lead-in/out and the two blends."""
    rate = 1e9 / CONTROL_DT_NS
    return bracket_timeline(
        timeline,
        DEFAULT_INITIAL_MOTION_TOKEN,
        OPEN_HAND_Q,
        lead_in_ticks=round(config.lead_in_s * rate),
        lead_out_ticks=round(config.lead_out_s * rate),
        blend_ticks=round(config.blend_s * rate),
        force=config.force,
    )


def main(config: Config) -> None:
    session = Path(config.session)
    is_parquet = session.suffix == ".parquet" or (session / "meta" / "info.json").exists()

    if is_parquet:
        # A training-export episode: already teleop-only, hands always from
        # the recorded teleop targets (there is no measured-hand column).
        if config.hand_source not in ("cmd",):
            raise SystemExit(
                f"--hand-source {config.hand_source} is a collector-session option; "
                f"a LeRobot episode only carries the commanded teleop hand targets"
            )
        print(f"Episode: {session}" + (f" [{config.episode}]" if config.episode is not None else ""))
        if config.tokens_from == "joints":
            if not Path(ENCODER_ONNX).exists():
                raise SystemExit(
                    f"{ENCODER_ONNX} not found — the docker image bakes it in; on a "
                    f"host checkout: wget -P models "
                    f"https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_encoder.onnx"
                )
            print(f"Re-encoding joints ({config.joint_source}) via {ENCODER_ONNX}")
            timeline = load_reencoded_episode(
                session,
                ENCODER_ONNX,
                episode_index=config.episode,
                joint_source=config.joint_source,
                max_hold_ticks=config.max_gap_ticks,
            )
        else:
            timeline = load_episode(
                session, episode_index=config.episode, max_hold_ticks=config.max_gap_ticks
            )
    else:
        if config.tokens_from == "joints":
            raise SystemExit(
                "--tokens-from joints needs a LeRobot parquet episode (the collector "
                "CSV sessions carry no aligned whole-body joint stream)"
            )
        print(f"Session: {session}")
        timeline = load_session(
            session,
            arbiter_modes=(ARBITER_TELEOP,) if config.teleop_only else None,
            max_hold_ticks=config.max_gap_ticks,
            hand_source=config.hand_source,
        )
    try:
        stream = build_stream(timeline, config)
    except CompressedGapError as e:
        print(
            f"\nRefusing to publish: {e}.\n"
            f"Either raise --max-gap-ticks (now {config.max_gap_ticks}) so the transition "
            f"is spread over the whole gap instead, restrict the replay (--teleop-only), "
            f"or pass --force if the faster ramp is acceptable."
        )
        sys.exit(1)

    dds_cfg = load_dds_config(config.config)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    # Main thread = lifecycle only: the publisher owns the channel and the
    # Tx worker thread (ext-sensor-io transmitter pattern).
    print("THE ROBOT MOVES NOW — VR e-stop is A+B+X+Y held 1s.")
    publisher = LatentActionPublisher()
    publisher.start(
        stream, domain_id=domain, network_interface=dds_cfg.network_interface
    )
    try:
        publisher.wait()
    except KeyboardInterrupt:
        print(
            "\nStopped by user mid-stream — gearsonic sees the stream go stale after 500ms "
            "and runs its LOST recovery (blend to standing, planner reseed, back to origin)."
        )
    finally:
        publisher.stop()
