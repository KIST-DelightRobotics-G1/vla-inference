"""The encoding stage: AlignedJoints -> EncodedTokens through the SONIC encoder.

    encoder.py          obs_dict assembly (token_encoder.cpp port) + the
                        ONNX encoder; encode_tokens_from_joints() is the
                        stage's entry point
    encoded_tokens.py   EncodedTokens — what the encoder produces: an
                        AlignedTokens whose values it made (provenance as
                        a type)

Pure computation, no file I/O and no time handling: the align stage supplies
`AlignedTokens` (grid, hands) and `AlignedJoints` (the material, 1:1 with
the ticks); this stage returns an `EncodedTokens` that the builder consumes
like any other AlignedTokens. Re-encoding is how a recording survives a
decoder-checkpoint change — swap `models/model_encoder.onnx` together with
gearsonic's decoder.
"""

from .encoded_tokens import EncodedTokens
from .encoder import encode_tokens_from_joints, load_onnx_encoder

__all__ = ["EncodedTokens", "encode_tokens_from_joints", "load_onnx_encoder"]
