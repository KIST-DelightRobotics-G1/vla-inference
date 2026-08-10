"""KIST GR00T N1.7 VLA inference service.

Runs an Isaac-GR00T N1.7 policy (UNITREE_G1_SONIC embodiment) and streams
64-dim SONIC motion tokens + hand joints to the kist-gearsonic-inference
C++ whole-body controller over ZMQ (latent protocol v4).
"""

__version__ = "0.1.0"
