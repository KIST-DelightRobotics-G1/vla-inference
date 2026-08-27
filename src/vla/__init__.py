"""GR00T N1.7 VLA inference — rebuilt as a stage pipeline (in progress).

Being rebuilt from scratch on the replay package's principles (stage
folders are the data flow, one dataclass contract between stages, only
inference-reachable code) instead of wrapping the whole Isaac-GR00T repo.
The previous runtime lives in `src/vla_old/` until this replaces it.

    io/    the input sources: cameras (ext-sensor-io H.264 over DDS,
           N views), robot state (unitree rt/lowstate + rt/dex3/*), and
           the operator/prompt channel
"""
