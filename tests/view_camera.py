#!/usr/bin/env python3
"""Manually view a kist-ext-sensor-io camera stream (not a pytest test).

Subscribes with the real ColorSubscriber (DDS + H.264 decode thread) and
shows what arrives. Two output modes:

- default: writes the newest frame to shared/camera_preview/<view>.jpg
  twice a second — made for the docker container (no display needed): the
  shared/ mount lets you open the file on the host, and any image viewer
  that auto-reloads acts as a ~2 fps monitor.
- --show: a live cv2.imshow window (needs a GUI opencv build and a
  DISPLAY; inside docker add `-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`).

Usage (ext-sensor-io must be publishing):
    python tests/view_camera.py                          # ego_view, default topic
    python tests/view_camera.py --view left_wrist --name left_wrist
    python tests/view_camera.py --show

Prints frame shape, sensor stamp, receive rate, and the latest() age once
a second — the same numbers the observation assembly will key off.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import tyro

from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config
from vla.io.realsense import ColorSubscriber, DEFAULT_COLOR_TOPIC, color_topic_for


@dataclass
class Config:
    view: str = "ego_view"
    """Observation key this stream maps to (names the output file too)."""

    name: str | None = None
    """ext-sensor-io camera name — sets the topic to
    rt/kist/camera/<name>/color/h264. Default: the unnamed default topic."""

    topic: str | None = None
    """Explicit topic override (wins over --name)."""

    config: str = "config/config.yaml"
    """Network settings (dds.domain_id + the CycloneDDS transport XML)."""

    domain: int | None = None
    """DDS domain id override. Default: the config file's dds.domain_id."""

    save_dir: str = "shared/camera_preview"
    """Where the newest frame is written as <view>.jpg (default mode)."""

    show: bool = False
    """Live cv2.imshow window instead of JPEG snapshots."""


def main(config: Config) -> None:
    topic = config.topic or (
        color_topic_for(config.name) if config.name else DEFAULT_COLOR_TOPIC
    )
    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    subscriber = ColorSubscriber(config.view, topic=topic)
    subscriber.start(domain_id=domain)

    save_path = Path(config.save_dir) / f"{config.view}.jpg"
    if not config.show:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing the newest frame to {save_path} (open it on the host)")

    last_stamp = None
    frames = 0
    last_received = last_lost = last_resyncs = 0
    last_report = time.monotonic()
    last_save = 0.0
    try:
        while True:
            frame, age = subscriber.latest()
            now = time.monotonic()

            if frame is not None and frame.stamp_ns != last_stamp:
                last_stamp = frame.stamp_ns
                frames += 1
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                if config.show:
                    try:
                        cv2.imshow(config.view, bgr)
                    except cv2.error:
                        raise SystemExit(
                            "--show needs a GUI opencv build and a DISPLAY; this "
                            "environment has opencv-python-headless (the docker "
                            "image does). Use the default JPEG mode instead."
                        ) from None
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif now - last_save >= 0.5:
                    cv2.imwrite(str(save_path), bgr)
                    last_save = now

            if now - last_report >= 1.0:
                if frame is None:
                    print(
                        f"waiting for frames on {topic} ... (is ext-sensor-io "
                        f"publishing there? a named camera needs --name <camera>)"
                    )
                else:
                    print(
                        f"{config.view}: {frame.rgb.shape[1]}x{frame.rgb.shape[0]}  "
                        f"{frames} new frames/s  age {age * 1000:.0f}ms  "
                        f"recv {subscriber.received - last_received}/s  "
                        f"lost {subscriber.lost - last_lost}  "
                        f"resyncs {subscriber.resyncs - last_resyncs}  "
                        f"stamp {frame.stamp_ns}"
                    )
                frames = 0
                last_received = subscriber.received
                last_lost = subscriber.lost
                last_resyncs = subscriber.resyncs
                last_report = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.stop()
        if config.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main(tyro.cli(Config))
