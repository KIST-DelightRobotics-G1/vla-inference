"""ActionStream — the finished publish plan, iterable as LatentActionSteps."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..constants import CONTROL_DT_NS, HAND_DIM, TOKEN_DIM
from ..latent_action_step import LatentActionStep


@dataclass(frozen=True)
class ActionStream:
    """A finished publish plan: T ticks, dense arrays, iterable as steps.

    Stays array-of-ticks internally (the blend/bracket math wants dense
    arrays); indexing or iterating views it one `LatentActionStep` at a time,
    which is the shape the publisher and the wire want.
    """

    tokens: np.ndarray  # (T, 64) float32
    left_hand: np.ndarray  # (T, 7) float32
    right_hand: np.ndarray  # (T, 7) float32

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if self.tokens.shape != (n, TOKEN_DIM):
            raise ValueError(f"tokens must be (T, {TOKEN_DIM}), got {self.tokens.shape}")
        for name, arr in (("left_hand", self.left_hand), ("right_hand", self.right_hand)):
            if arr.shape != (n, HAND_DIM):
                raise ValueError(f"{name} must be ({n}, {HAND_DIM}), got {arr.shape}")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def duration_s(self) -> float:
        return len(self) * CONTROL_DT_NS / 1e9

    def __getitem__(self, i: int) -> LatentActionStep:
        return LatentActionStep(
            frame_index=i if i >= 0 else len(self) + i,
            token_state=self.tokens[i],
            left_hand_joints=self.left_hand[i],
            right_hand_joints=self.right_hand[i],
        )

    def __iter__(self) -> Iterator[LatentActionStep]:
        for i in range(len(self)):
            yield self[i]
