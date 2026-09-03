"""The VLA runner: cameras + robot state -> GR00T inference -> 50 Hz stream.

The end-to-end assembly of the stage pipeline — everything else in vla/ is
a stage, this is the wiring:

    io          ColorSubscriber per checkpoint view + UnitreeStateReader,
                all on ONE DomainParticipant
    observation ObservationBuilder — fresh snapshots or nothing
    policy      SonicPolicy — Observation -> 40-step ActionChunk (~135 ms)
    chunking    ChunkCursor — the seam between the two clocks
    publisher   LatentActionStreamer — 50 Hz onto rt/kist/latent_action

Two loops: the streamer's 50 Hz Tx thread (its own), and THIS main thread
running inference back-to-back — every prediction is pushed with the ticks
it spent in flight already skipped, so the cursor splices it in at "now".
No inference throttle: a faster GPU simply replaces chunks more often, and
the horizon (0.78 s) rides out the slow moments. Ctrl+C stops the stream;
gearsonic's LOST recovery brings the robot to the safe standing pose.

Which cameras exist is not configured here: the checkpoint's modality
config names its views, and `cameras` only maps view names to
ext-sensor-io camera names — a 1-view checkpoint runs one subscriber, the
3-view one runs three.
"""

import time
from dataclasses import dataclass, field

import tyro

from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config

from .chunking import ChunkCursor
from .io.realsense import ColorSubscriber, color_topic_for
from .io.unitree import UnitreeStateReader
from .observation import ObservationBuilder
from .policy import SonicPolicy
from .publisher import LatentActionStreamer
from .publisher.latent_action_streamer import CONTROL_DT_NS

# The training prompt — a single-task checkpoint conditions on this exact
# string (UNITREE_G1_SONIC_3VIEWS.md §5); a different checkpoint needs its
# own training prompt passed via --prompt.
DEFAULT_PROMPT = (
    "Open the right door of the refrigerator. Hook the yellow tip attached "
    "to your right hand under the door handle and pull."
)


@dataclass
class Config:
    checkpoint: str = "/workspace/checkpoints/checkpoint-4500"
    """Finetuned checkpoint directory (mounted by docker/run.sh)."""

    prompt: str = DEFAULT_PROMPT
    """Task instruction — MUST match the checkpoint's training string."""

    embodiment_tag: str = "unitree_g1_sonic_3views"
    """Which of the checkpoint's embodiments to run."""

    device: str = "cuda:0"
    """CUDA device for the model."""

    cameras: dict[str, str] = field(
        default_factory=lambda: {
            "ego_view": "head",
            "left_wrist": "left_wrist",
            "right_wrist": "right_wrist",
        }
    )
    """Checkpoint view name -> ext-sensor-io camera name. Only the views the
    checkpoint's modality config asks for are subscribed."""

    config: str = "config/config.yaml"
    """Network settings (dds: domain_id, cyclonedds_xml) — gearsonic-style.
    A missing file at this default path falls back to built-in defaults."""

    domain: int | None = None
    """DDS domain id override (must match gearsonic). Default: the config
    file's dds.domain_id."""

    report_s: float = 2.0
    """Seconds between status lines (inference latency, cursor health)."""

    probe: str | None = None
    """Progress-probe .pt path (e.g. /data/vla/progress_probe/probe_succ3v.pt).
    Only valid with the checkpoint it was fitted on — the probe prints its
    extractor on load. None disables the probe entirely."""

    probe_dir: str = "shared/probe"
    """Where the per-rollout progress JSONL goes (host-mounted via shared/)."""


def main(config: Config) -> None:
    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    # Model first: 25+ s of loading during which DDS would just sit idle.
    print(f"Loading checkpoint {config.checkpoint} ({config.embodiment_tag})...")
    policy = SonicPolicy(
        config.checkpoint, device=config.device, embodiment_tag=config.embodiment_tag
    )
    views = policy.video_views
    missing = [v for v in views if v not in config.cameras]
    if missing:
        raise SystemExit(
            f"checkpoint wants views {views} but --cameras has no mapping for "
            f"{missing} (ext-sensor-io camera name needed)"
        )
    print(f"Checkpoint views: {views}")

    # Progress probe (optional): hooks the DiT latent inside predict() —
    # the score is a free byproduct of each inference, logged per rollout.
    probe = None
    progress_log = None
    if config.probe is not None:
        from .progress_probe import ProgressLog, ProgressProbe

        probe = ProgressProbe(config.probe)
        probe.check_prompt(config.prompt)
        probe.attach(policy.torch_model)
        progress_log = ProgressLog(config.probe_dir)

    # One participant for every Rx source (ChannelFactory convention); the
    # streamer's writer owns its Tx side separately.
    from cyclonedds.domain import DomainParticipant

    participant = DomainParticipant(domain)
    cameras = {}
    for view in views:
        camera = ColorSubscriber(view, topic=color_topic_for(config.cameras[view]))
        camera.start(participant=participant)
        cameras[view] = camera
    state_reader = UnitreeStateReader()
    state_reader.start(participant=participant)
    builder = ObservationBuilder(cameras, state_reader)

    cursor = ChunkCursor()
    streamer = LatentActionStreamer()
    tick_s = CONTROL_DT_NS / 1e9

    try:
        # Wait for the sensors, then warm up the model off-stream: the first
        # prediction pays CUDA init (~1 s) — better spent before gearsonic
        # can see us than as a stale first chunk.
        print("Waiting for fresh sensor streams...")
        observation = None
        while observation is None:
            observation = builder.build(config.prompt)
            if observation is None:
                time.sleep(0.05)
        print("Warmup inference...")
        policy.predict(observation)

        streamer.start(cursor, domain_id=domain)
        print(f"Streaming — prompt: {config.prompt!r}")

        last_report = time.monotonic()
        latency_sum = 0.0
        predictions = 0
        while True:
            t0 = time.monotonic()
            observation = builder.build(config.prompt)
            if observation is None:
                # The builder already reported which stream is stale; the
                # cursor plays out its horizon, then hands over to recovery.
                time.sleep(tick_s)
                continue

            chunk = policy.predict(observation)
            elapsed = time.monotonic() - t0
            cursor.push(chunk, skip_ticks=round(elapsed / tick_s))

            progress = None
            if probe is not None:
                # The hook fired inside predict(); this is that
                # observation's score. One dot product — negligible.
                progress = probe.read()
                progress_log.append(progress, elapsed * 1e3)

            latency_sum += elapsed
            predictions += 1
            if t0 - last_report >= config.report_s:
                stats = cursor.stats()
                progress_part = (
                    "" if probe is None else f"progress {progress:.2f} | "
                )
                print(
                    f"inference {latency_sum / predictions * 1e3:.0f}ms avg "
                    f"({predictions / (t0 - last_report):.1f}/s) | "
                    f"{progress_part}"
                    f"published {streamer.published} | held {stats['held']} "
                    f"starved {stats['starved']} stale {stats['stale_pushes']} "
                    f"late {streamer.late}",
                    flush=True,
                )
                last_report = t0
                latency_sum = 0.0
                predictions = 0
    except KeyboardInterrupt:
        print("\nStopping (gearsonic recovers to safe standing on stream loss)")
    finally:
        streamer.stop()
        for camera in cameras.values():
            camera.stop()
        state_reader.stop()
        if probe is not None:
            probe.detach()
            progress_log.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
