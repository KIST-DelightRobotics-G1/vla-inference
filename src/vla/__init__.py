"""GR00T N1.7 VLA inference — rebuilt as a stage pipeline (in progress).

Being rebuilt from scratch on the replay package's principles (stage
folders are the data flow, one dataclass contract between stages, only
inference-reachable code) instead of wrapping the whole Isaac-GR00T repo.
The previous runtime was removed 2026-08-27 — recover it from git history
(src/vla_old/) if a reference is needed.

    io/           the input sources: cameras (ext-sensor-io H.264 over
                  DDS, N views) and robot state (unitree rt/lowstate +
                  rt/dex3/*), each exposing latest*() -> (snapshot, age)
    observation/  fresh io snapshots -> Observation (the policy's input);
                  per-stream staleness is judged here
    policy/       Observation -> action chunk; carries the vendored GR00T
                  N1.7 inference core (extracted from Isaac-GR00T@5ac4e6b)
"""
