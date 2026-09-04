"""The policy stage: Observation -> ActionChunk, via the vendored GR00T core.

    sonic_policy.py  SonicPolicy — loads the finetuned checkpoint once,
                     `predict(Observation) -> ActionChunk` per tick
    action_chunk.py  ActionChunk — the contract with the chunking stage:
                     40 future steps of motion_token + hand joint targets
    gr00t_format.py  the pure-numpy Observation <-> gr00t dict reshaping
                     (host-testable; normalization lives in the processor)
    gr00t/           the GR00T N1.7 inference core, extracted from
                     Isaac-GR00T at the pinned commit 5ac4e6b (16 files,
                     training pipeline left behind) with the
                     unitree_g1_sonic_3views embodiment registered — see
                     gr00t/__init__.py for the deviations from upstream

SonicPolicy (and the gr00t subpackage) imports torch/transformers —
inference-container only; it is exported lazily so ActionChunk and the
format helpers stay importable on the host.
"""

from .action_chunk import ActionChunk

__all__ = ["ActionChunk", "SonicPolicy"]


def __getattr__(name):
    if name == "SonicPolicy":
        from .sonic_policy import SonicPolicy

        return SonicPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
