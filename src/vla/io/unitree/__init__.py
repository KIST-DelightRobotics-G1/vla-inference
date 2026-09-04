"""Unitree robot-state streams — the python twin of gearsonic's unitree/.

Same topics the gearsonic C++ side reads, same converted structs, same
names (`UnitreeStateReader`, `UnitreeState`, `IMU`):

    unitree_state_reader.py  UnitreeStateReader — rt/lowstate +
                             rt/dex3/{left,right}/state; an Rx thread
                             stamps arrival times so ages are trustworthy
    unitree_state.py         UnitreeState + IMU — one converted LowState
                             sample (29 motors q/dq/tau, PELVIS imu, tick,
                             mode_machine)
    hand_state.py            HandState — one converted Dex3 sample (our
                             addition: gearsonic only writes hand commands,
                             the VLA observation needs the measured hands)

Consumer contract: `latest_state()` / `latest_left_hand()` /
`latest_right_hand()` each return `(snapshot | None, age_s)` — per-stream
ages, so freshness is judged per stream (the hands run on their own
clocks).
"""

from .hand_state import HandState
from .unitree_state import IMU, UnitreeState
from .unitree_state_reader import UnitreeStateReader

__all__ = ["HandState", "IMU", "UnitreeState", "UnitreeStateReader"]
