"""AlignedJoints — the joint stream resampled 1:1 onto the token rows."""

from dataclasses import dataclass

from ..io.joints import Joints


@dataclass
class AlignedJoints(Joints):
    """A `Joints` resampled onto the token rows' clock, one row per tick.

    Same fields and behavior as any `Joints`; the type exists so the encoder
    can require that the join already happened — the way holding an
    `AlignedTokens` proves the hand join did.
    """
