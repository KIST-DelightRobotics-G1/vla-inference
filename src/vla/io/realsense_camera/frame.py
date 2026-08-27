"""Frame — one decoded camera image, the camera subscriber's product."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Frame:
    """The newest decoded image of one camera view.

    `stamp_ns` is the sensor-side capture stamp (ext-sensor-io's clock);
    freshness on the consumer side comes from the subscriber's `latest()`
    age, not from this stamp.
    """

    rgb: np.ndarray  # (H, W, 3) uint8, RGB
    stamp_ns: int
