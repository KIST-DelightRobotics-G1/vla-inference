"""Unitree robot-state streams — poll-based, no thread needed.

The same topics the gearsonic C++ side reads: `rt/lowstate` (29 body motors
+ the pelvis IMU quaternion — NOT the torso IMU on rt/secondary_imu) and
`rt/dex3/{left,right}/state` (7 hand motors each, on their own clocks).

    robot_state_subscriber.py  RobotStateSubscriber — three KeepLast(1)
                               readers polled on the caller's thread (the
                               transport already keeps the newest sample);
                               non-finite samples dropped at intake
    robot_state.py             RobotState — the product: one combined
                               whole-robot snapshot

Consumer contract: `latest() -> (RobotState | None, age_s)` — same shape
as the camera subscriber; threading is each source's internal detail.
"""

from .robot_state import RobotState
from .robot_state_subscriber import RobotStateSubscriber

__all__ = ["RobotState", "RobotStateSubscriber"]
