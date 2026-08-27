"""The input sources: everything the observation is assembled from.

Every source exposes the same consumer contract — `latest()` returns
`(snapshot | None, age_seconds)`, a latest-value read that never blocks —
while threading stays each source's internal detail (a camera needs a
decode thread because H.264 delta frames must be consumed continuously;
robot state just polls the transport's KeepLast(1) cache):

    realsense/  ext-sensor-io H.264 color streams, one
                       ColorSubscriber per view -> ColorFrame
    unitree/           rt/lowstate + rt/dex3/{left,right}/state,
                       RobotStateSubscriber -> RobotState
"""
