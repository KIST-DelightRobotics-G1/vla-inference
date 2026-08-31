"""SonicPolicy — Observation -> ActionChunk, via the vendored GR00T core.

The stage's logic: load the finetuned checkpoint once (the vendored
Gr00tPolicy brings up the Cosmos-Reason2-2B backbone, the flow-matching
action head and the processor with the checkpoint's own statistics), then
`predict()` per inference tick. One prediction = one backbone forward + four
DiT Euler steps, in bfloat16 on the GPU.

This module imports torch through the vendored core — inference-container
only. The format conversion it delegates to (gr00t_format) stays
host-testable.
"""

from vla.observation import Observation

from .action_chunk import ActionChunk
from .gr00t_format import to_action_chunk, to_gr00t_observation
from .gr00t.policy import Gr00tPolicy

# The checkpoint this deployment targets: the 3-camera SONIC finetune
# (UNITREE_G1_SONIC_3VIEWS.md). Identical state/action spaces to
# unitree_g1_sonic — the tags share projector slot 11.
EMBODIMENT_TAG = "unitree_g1_sonic_3views"


class SonicPolicy:
    """Wrap one loaded checkpoint; turn Observations into ActionChunks.

    Args:
        checkpoint_path: the finetuned checkpoint directory (config.json,
            processor_config.json, statistics.json, safetensors).
        device: CUDA device for the model (bfloat16).
        embodiment_tag: which of the checkpoint's embodiments to run.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        device: str = "cuda:0",
        embodiment_tag: str = EMBODIMENT_TAG,
    ):
        self._policy = Gr00tPolicy(embodiment_tag, checkpoint_path, device=device)
        self._language_key = self._policy.language_key

    def predict(self, observation: Observation) -> ActionChunk:
        """One inference: fresh Observation -> decoded 40-step ActionChunk."""
        gr00t_observation = to_gr00t_observation(observation, self._language_key)
        action, _ = self._policy.get_action(gr00t_observation)
        return to_action_chunk(action)
