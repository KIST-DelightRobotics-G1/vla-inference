"""Time alignment: joins the recording's side streams onto the token clock.

One rule for every join — the newest source row at or before each token
tick's `recv_ns` (backward merge_asof, `align_by_recv_ns`); `align_tokens`
applies it to the hand streams, `align_joints` to the joint stream.
"""

import numpy as np

from ..constants import HAND_DIM
from ..io.joints import Joints
from ..io.tokens import Tokens
from .aligned_joints import AlignedJoints
from .aligned_tokens import AlignedTokens


def align_by_recv_ns(
    target_recv_ns: np.ndarray, src_recv_ns: np.ndarray, src_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Newest source row at or before each target time (backward merge_asof).

    Returns `(values, before_first)` where `values[i]` is the source row
    aligned to `target_recv_ns[i]`, and `before_first[i]` marks targets that
    precede the first source row — those clamp to the first row (the only
    honest choice without extrapolating) and the caller may warn about them.
    """
    src_recv_ns = np.asarray(src_recv_ns)
    if len(src_recv_ns) == 0:
        raise ValueError("no source rows to align against")

    idx = np.searchsorted(src_recv_ns, np.asarray(target_recv_ns), side="right") - 1
    before_first = idx < 0
    return src_values[np.clip(idx, 0, len(src_recv_ns) - 1)], before_first


def align_tokens(tokens: Tokens) -> AlignedTokens:
    """Join the raw hand streams onto the token ticks.

    Each tick gets the newest hand command at or before its `recv_ns`; a
    missing or empty hand stream degrades to the open-hand pose (zeros, Dex
    motor order). Ticks that precede the first hand row clamp to it and are
    counted in `hand_ticks_before_first`.
    """
    before_first = 0
    any_hands = False
    hands: list[np.ndarray] = []
    for source in (tokens.left_hand, tokens.right_hand):
        if source is None or len(source[0]) == 0:
            hands.append(np.zeros((len(tokens), HAND_DIM), dtype=np.float32))
            continue
        any_hands = True
        src_recv, src_q = source
        q, before = align_by_recv_ns(tokens.recv_ns, src_recv, src_q)
        before_first = max(before_first, int(before.sum()))
        hands.append(q.astype(np.float32))

    return AlignedTokens(
        recv_ns=tokens.recv_ns,
        stamp_ns=tokens.stamp_ns,
        seq=tokens.seq,
        arbiter_mode=tokens.arbiter_mode,
        encoder_mode=tokens.encoder_mode,
        values=tokens.values,
        left_hand=hands[0],
        right_hand=hands[1],
        hands_from=tokens.hands_from if any_hands else "none",
        hand_ticks_before_first=before_first,
    )


def align_joints(tokens: Tokens, joints: Joints) -> AlignedJoints:
    """Resample the joint stream onto the token ticks.

    Each tick gets the newest joint sample at or before its `recv_ns` —
    identity when the streams share a clock, as in parquet episodes; a real
    resample for CSV lowstate, which runs faster than the 50 Hz token stream.
    """
    q, _ = align_by_recv_ns(tokens.recv_ns, joints.recv_ns, joints.q)
    base_quat, _ = align_by_recv_ns(tokens.recv_ns, joints.recv_ns, joints.base_quat)
    dq = None
    if joints.dq is not None:
        dq, _ = align_by_recv_ns(tokens.recv_ns, joints.recv_ns, joints.dq)

    return AlignedJoints(recv_ns=tokens.recv_ns, q=q, base_quat=base_quat, dq=dq)
