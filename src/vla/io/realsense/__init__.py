"""RealSense color streams over DDS — one subscriber (and decode thread) per view.

kist-ext-sensor-io owns the cameras and publishes H.264 Annex-B NAL units
as `kist_msgs::CompressedColorFrame` on `rt/kist/camera[/<name>]/color/h264`.

    color_subscriber.py  ColorSubscriber — owns the DDS reader and a
                          decode thread (H.264 delta frames must be decoded
                          in arrival order, however often the consumer
                          looks); keeps the newest decoded frame
    color_frame.py              ColorFrame — the product: one decoded RGB image

Consumer contract: `latest() -> (ColorFrame | None, age_s)` — a snapshot with
its age, so staleness stays the consumer's decision (per view).
"""

from .color_subscriber import DEFAULT_COLOR_TOPIC, ColorSubscriber, color_topic_for
from .color_frame import ColorFrame

__all__ = ["ColorSubscriber", "DEFAULT_COLOR_TOPIC", "ColorFrame", "color_topic_for"]
