"""RealSense color streams over DDS — one subscriber (and decode thread) per view.

kist-ext-sensor-io owns the cameras and publishes H.264 Annex-B NAL units
as `kist_msgs::CompressedColorFrame` on `rt/kist/camera[/<name>]/color/h264`.

    camera_subscriber.py  CameraSubscriber — owns the DDS reader and a
                          decode thread (H.264 delta frames must be decoded
                          in arrival order, however often the consumer
                          looks); keeps the newest decoded frame
    frame.py              Frame — the product: one decoded RGB image

Consumer contract: `latest() -> (Frame | None, age_s)` — a snapshot with
its age, so staleness stays the consumer's decision (per view).
"""

from .camera_subscriber import DEFAULT_COLOR_TOPIC, CameraSubscriber, color_topic_for
from .frame import Frame

__all__ = ["CameraSubscriber", "DEFAULT_COLOR_TOPIC", "Frame", "color_topic_for"]
