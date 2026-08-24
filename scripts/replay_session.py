#!/usr/bin/env python3
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
    [replay]   the recorded tokens, gaps blended over (see kist_vla/replay.py)
    [blend]    the session's last token -> standing
    [lead-out] safe standing token, then the publisher stops

Without the lead-out the stream would simply end mid-pose and gearsonic would
run its own LOST recovery 500 ms later (blend to standing, planner reseed,
back to the origin) — safe, but from wherever the episode happened to stop.

Usage:
    # Inspect a session — no DDS, no robot:
    python scripts/replay_session.py sessions/20260824_141530 --dry-run

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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from kist_vla.config import DEFAULT_INITIAL_MOTION_TOKEN
from kist_vla.g1_joints import OPEN_HAND_Q
from kist_vla.replay import (
    ARBITER_NAMES,
    ARBITER_TELEOP,
    CONTROL_DT_NS,
    ReplayTimeline,
    bracket_timeline,
    load_session,
)


@dataclass
class Config:
    session: str
    """Collector session directory (contains motion_token.csv)."""

    domain: int = 0
    """DDS domain id (must match the gearsonic receiver)."""

    dry_run: bool = False
    """Analyze the session and print the plan without publishing (no DDS)."""

    teleop_only: bool = False
    """Replay only arbiter_mode==1 (teleop demonstration) ticks — what the
    training export keeps. Default replays every recorded tick."""

    hand_source: str = "cmd"
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


def _print_report(timeline: ReplayTimeline) -> None:
    modes = ", ".join(
        f"{ARBITER_NAMES.get(m, m)}({m})={n}" for m, n in sorted(timeline.arbiter_modes.items())
    )
    print(f"  ticks           {len(timeline)} ({timeline.duration_s:.2f}s at 50 Hz)")
    print(f"  recorded        {timeline.recorded_ticks}")
    print(f"  gap-filled      {len(timeline) - timeline.recorded_ticks}")
    print(f"  arbiter modes   {modes}")
    print(f"  hand targets    {timeline.hands_from}")
    if timeline.hand_ticks_before_first:
        print(
            f"    note: {timeline.hand_ticks_before_first} tick(s) precede the first hand "
            f"command row — clamped to it"
        )
    bound = float(np.abs(timeline.tokens).max())
    print(f"  |token| max     {bound:.4f}")

    if timeline.gaps:
        print(f"  gaps            {len(timeline.gaps)}")
        for gap in sorted(timeline.gaps, key=lambda g: g.ticks, reverse=True)[:10]:
            note = f" -> COMPRESSED to {gap.filled_ticks}" if gap.compressed else " -> blended"
            print(f"    after seq {gap.after_seq}: {gap.ticks} ticks ({gap.duration_s:.2f}s){note}")
        if len(timeline.gaps) > 10:
            print(f"    ... and {len(timeline.gaps) - 10} more")
    else:
        print("  gaps            none (contiguous 50 Hz recording)")


def build_stream(
    timeline: ReplayTimeline, config: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bracket the replay with the standing lead-in/out and the two blends."""
    rate = 1e9 / CONTROL_DT_NS
    return bracket_timeline(
        timeline,
        DEFAULT_INITIAL_MOTION_TOKEN,
        OPEN_HAND_Q,
        lead_in_ticks=round(config.lead_in_s * rate),
        lead_out_ticks=round(config.lead_out_s * rate),
        blend_ticks=round(config.blend_s * rate),
    )


def publish(
    sink, tokens: np.ndarray, left: np.ndarray, right: np.ndarray
) -> None:
    """Publish the stream on an absolute 50 Hz schedule.

    Absolute deadlines (not sleep(period - elapsed)) so a slow tick does not
    push the whole trajectory late: the replay's timing IS the recorded
    motion's timing. Late ticks are counted and reported — a tick beyond
    gearsonic's 500 ms staleness threshold would end the VLA session.
    """
    period = CONTROL_DT_NS / 1e9
    total = len(tokens)
    late = 0
    worst_late = 0.0
    start = time.monotonic()

    for i in range(total):
        deadline = start + (i + 1) * period
        sink.send_latent_action(
            motion_token=tokens[i],
            frame_index=i,
            left_hand_joints=left[i],
            right_hand_joints=right[i],
        )
        if i % 250 == 0:
            print(f"  tick {i}/{total}  ({i * period:.1f}s / {total * period:.1f}s)")

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        else:
            late += 1
            worst_late = max(worst_late, -remaining)

    elapsed = time.monotonic() - start
    print(f"Published {total} ticks in {elapsed:.2f}s (expected {total * period:.2f}s)")
    if late:
        print(f"WARNING: {late} tick(s) missed their deadline, worst overrun {worst_late * 1e3:.1f}ms")


def main(config: Config) -> None:
    session = Path(config.session)
    print(f"Session: {session}")
    timeline = load_session(
        session,
        arbiter_modes=(ARBITER_TELEOP,) if config.teleop_only else None,
        max_hold_ticks=config.max_gap_ticks,
        hand_source=config.hand_source,
    )
    _print_report(timeline)

    compressed = [g for g in timeline.gaps if g.compressed]
    if compressed and not config.force:
        worst = max(compressed, key=lambda g: g.ticks)
        print(
            f"\nRefusing to publish: {len(compressed)} gap(s) exceed --max-gap-ticks "
            f"({config.max_gap_ticks}), worst {worst.ticks} ticks ({worst.duration_s:.2f}s) "
            f"after seq {worst.after_seq}.\n"
            f"A compressed gap ramps a real pose change over "
            f"{config.max_gap_ticks * CONTROL_DT_NS / 1e9:.2f}s, which may be too fast for the "
            f"robot. Either raise --max-gap-ticks so the transition is spread over the whole "
            f"gap instead, restrict the replay (--teleop-only), or pass --force if the "
            f"faster ramp is acceptable."
        )
        sys.exit(1)

    tokens, left, right = build_stream(timeline, config)
    print(
        f"\nPublish plan: {len(tokens)} ticks ({len(tokens) * CONTROL_DT_NS / 1e9:.2f}s) "
        f"= {config.lead_in_s:.1f}s standing + {config.blend_s:.1f}s blend + "
        f"{timeline.duration_s:.2f}s replay + {config.blend_s:.1f}s blend + "
        f"{config.lead_out_s:.1f}s standing"
    )

    if config.dry_run:
        print("Dry run — nothing published.")
        return

    from kist_vla.io.dds import DdsActionSink

    sink = DdsActionSink(domain_id=config.domain)
    print("Waiting 1s for DDS discovery...")
    time.sleep(1.0)
    print("THE ROBOT MOVES NOW — VR e-stop is A+B+X+Y held 1s.")

    try:
        publish(sink, tokens, left, right)
    except KeyboardInterrupt:
        print(
            "\nStopped by user mid-stream — gearsonic sees the stream go stale after 500ms "
            "and runs its LOST recovery (blend to standing, planner reseed, back to origin)."
        )
    finally:
        sink.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
