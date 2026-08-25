"""Transport-agnostic interfaces between the runner and the outside world.

The runner core only sees these Protocols; the concrete implementations live
in ``common/cyclonedds/`` (camera from kist-ext-sensor-io H.264, robot state
directly from unitree DDS topics, latent actions via the
``kist_msgs::LatentActionStep`` IDL type), and tests inject fakes.

Structural typing (``typing.Protocol``) — implementations don't inherit,
they just match the shape.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

# Operator message convention: a prompt change arrives as one string message
# with this prefix ("prompt:<text>"); anything else is a single keystroke.
PROMPT_PREFIX = "prompt:"


@runtime_checkable
class CameraSource(Protocol):
    """Latest-frame camera source."""

    def read(self) -> dict[str, Any] | None:
        """Return ``{"timestamps": {...}, "images": {name: RGB uint8}}`` or None."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class StateSource(Protocol):
    """Latest-value robot state source."""

    def get_msg(self, clear: bool = True) -> dict[str, Any] | None:
        """Return a state dict (body_q, left/right_hand_q, base_quat) or None."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class OperatorSource(Protocol):
    """Operator command source (keystrokes / prompt changes)."""

    def read_msg(self) -> str | None: ...

    def close(self) -> None: ...


@runtime_checkable
class ActionWriter(Protocol):
    """Outbound channel toward the whole-body controller."""

    def send_latent_action(
        self,
        motion_token: np.ndarray,
        frame_index: int,
        left_hand_joints: np.ndarray,
        right_hand_joints: np.ndarray,
    ) -> None: ...

    def send_command(self, start: bool, planner: bool = False) -> None:
        """Control-loop lifecycle command (start/stop, planner/pose mode)."""
        ...

    def close(self) -> None: ...
