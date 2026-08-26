"""EncodedTokens — what the encoder produces: AlignedTokens whose values it made."""

from dataclasses import dataclass

from ..aligner import AlignedTokens


@dataclass
class EncodedTokens(AlignedTokens):
    """An `AlignedTokens` whose `values` came from the SONIC encoder, not the disk.

    Same shape and downstream behavior as any `AlignedTokens` (build_timeline
    is provenance-blind); the type exists so the origin is visible — holding
    an `EncodedTokens` proves the values were re-encoded from joints, the way
    holding an `ActionStream` proves the gate was passed. Grid, seq, modes,
    and hand rows are carried over from the aligned recording unchanged.
    """
