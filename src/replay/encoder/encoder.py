"""Joints -> SONIC tokens: the offline encoding transform (g1 mode 0).

Pure computation — no file I/O, no time handling. The align stage supplies
`AlignedTokens` and `AlignedJoints` on one shared clock; this module turns
the joints into replacement token values with the GEAR-SONIC encoder ONNX.
Unlike the recorded tokens
(latents of the SONIC checkpoint that ran at collection time), these come
from whatever encoder you pass — re-encoding is how a recording survives a
decoder-checkpoint change (swap the encoder together with gearsonic's
decoder).

The observation assembly is a line-for-line port of gearsonic's
``src/control/token_encoder.cpp`` ``fill_obs()`` (g1 branch) and
``fill_anchor_orientation()``; the joint permutation is
``include/common/joint_order.hpp``. Do not edit the tables/offsets without
re-checking those files.

One deliberate simplification: gearsonic's heading alignment
(``apply_delta_heading``) maps the *planner's* reference frame onto the
*robot's* initial heading. Here the reference motion IS the recording, so
both initial quaternions are the same sample and the alignment is exactly
identity — the anchor reduces to ``conj(q_t) * q_tf`` (rotation from the
current base to the future frame's base).

Requires ``onnxruntime`` (the ``[encode]`` extra); imported lazily.
"""

from pathlib import Path

import numpy as np

from ..constants import CONTROL_DT_NS, TOKEN_DIM

# ── encoder input layout (token_encoder.cpp offsets, g1 mode) ────────────────
ENCODER_INPUT_DIM = 1762
OFF_MODE = 0        # obs[0] = 0.0 -> g1 mode (teleop slots stay zero)
OFF_MOTION_Q = 4    # 10 frames x 29 joints, IsaacLab order
OFF_MOTION_DQ = 294
OFF_ANCHOR_ORI = 601  # 10 frames x 6 (first two rotation-matrix columns)
NUM_FRAMES = 10     # *_10frame_step5
FRAME_STEP = 5
NUM_BODY_JOINTS = 29

# MuJoCo/Unitree-order joint i lives at IsaacLab index MUJOCO_TO_ISAACLAB[i]
# (gearsonic include/common/joint_order.hpp `isaaclab_to_mujoco` — that name
# reads from the IsaacLab side; from our MuJoCo-ordered recordings this is
# the scatter table).
MUJOCO_TO_ISAACLAB = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)

def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz, batched on the leading axes."""
    w1, x1, y1, z1 = (a[..., i] for i in range(4))
    w2, x2, y2, z2 = (b[..., i] for i in range(4))
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_to_first_two_columns(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 6): [m00, m01, m10, m11, m20, m21].

    The anchor-orientation encoding of fill_anchor_orientation(): the first
    two columns of the rotation matrix, flattened row-wise.
    """
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
            2 * (x * y + w * z),     1 - 2 * (x * x + z * z),
            2 * (x * z - w * y),     2 * (y * z + w * x),
        ],
        axis=-1,
    )


def assemble_encoder_obs(
    q_mujoco: np.ndarray,
    base_quat: np.ndarray,
    *,
    dq_mujoco: np.ndarray | None = None,
    dt: float = CONTROL_DT_NS / 1e9,
) -> np.ndarray:
    """Recorded joints + base quats -> encoder obs_dict rows (T, 1762).

    Args:
        q_mujoco: (T, 29) joint positions, MuJoCo/Unitree order (rad).
        base_quat: (T, 4) pelvis quaternion, wxyz.
        dq_mujoco: (T, 29) measured joint velocities (lowstate m*_dq) — when
            None (parquet episodes carry no velocities) the central finite
            difference of the positions is used instead.
        dt: recording tick period (50 Hz).

    Per tick t the 10 "future" frames are t, t+5, ..., t+45, clamped to the
    last tick (hold final pose) — exactly `target_frame()` with playing=true.
    """
    q_mujoco = np.asarray(q_mujoco, dtype=np.float64)
    base_quat = np.asarray(base_quat, dtype=np.float64)
    T = len(q_mujoco)
    assert q_mujoco.shape == (T, NUM_BODY_JOINTS) and base_quat.shape == (T, 4)

    # MuJoCo -> IsaacLab order (scatter); velocities measured or derived.
    q_isaac = np.empty_like(q_mujoco)
    q_isaac[:, MUJOCO_TO_ISAACLAB] = q_mujoco
    if dq_mujoco is not None:
        dq_isaac = np.empty_like(q_isaac)
        dq_isaac[:, MUJOCO_TO_ISAACLAB] = np.asarray(dq_mujoco, dtype=np.float64)
    else:
        dq_isaac = np.gradient(q_isaac, dt, axis=0)

    # Future-frame index grid: (T, NUM_FRAMES), clamped.
    t_idx = np.arange(T)[:, None] + np.arange(NUM_FRAMES)[None, :] * FRAME_STEP
    t_idx = np.minimum(t_idx, T - 1)

    obs = np.zeros((T, ENCODER_INPUT_DIM), dtype=np.float32)
    obs[:, OFF_MODE] = 0.0  # g1 mode (explicit no-op; other modes' slots stay 0)
    obs[:, OFF_MOTION_Q:OFF_MOTION_Q + NUM_FRAMES * NUM_BODY_JOINTS] = (
        q_isaac[t_idx].reshape(T, -1)
    )
    obs[:, OFF_MOTION_DQ:OFF_MOTION_DQ + NUM_FRAMES * NUM_BODY_JOINTS] = (
        dq_isaac[t_idx].reshape(T, -1)
    )

    # Anchor orientation: rotation from the current base to each future
    # frame's base (heading alignment = identity here, see module docstring).
    base_to_ref = _quat_mul(_quat_conjugate(base_quat)[:, None, :], base_quat[t_idx])
    obs[:, OFF_ANCHOR_ORI:OFF_ANCHOR_ORI + NUM_FRAMES * 6] = (
        _quat_to_first_two_columns(base_to_ref).reshape(T, -1).astype(np.float32)
    )
    return obs


def encode_joints(
    encoder,
    q_mujoco: np.ndarray,
    base_quat: np.ndarray,
    *,
    dq_mujoco: np.ndarray | None = None,
) -> np.ndarray:
    """Joints -> (T, 64) tokens: obs assembly + the encoder in one step."""
    if isinstance(encoder, (str, Path)):
        encoder = load_onnx_encoder(encoder)
    obs = assemble_encoder_obs(q_mujoco, base_quat, dq_mujoco=dq_mujoco)
    return np.asarray(encoder(obs), dtype=np.float32).reshape(-1, TOKEN_DIM)


def load_onnx_encoder(onnx_path: str | Path):
    """`model_encoder.onnx` -> a callable (N, 1762) -> (N, 64)."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "re-encoding joints requires onnxruntime — "
            "install the [encode] extra: uv pip install -e '.[encode]'"
        ) from e

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (inp,) = session.get_inputs()

    def encode(obs: np.ndarray) -> np.ndarray:
        out = [
            session.run(None, {inp.name: row[None, :].astype(np.float32)})[0][0]
            for row in obs
        ]
        return np.asarray(out, dtype=np.float32).reshape(len(obs), TOKEN_DIM)

    return encode


def encode_tokens_from_joints(tokens, joints, encoder) -> "EncodedTokens":
    """The encoding stage: a new `EncodedTokens` whose values are re-encoded
    from `joints` (g1 mode) — checkpoint portability.

    Both inputs are the align stage's products, already on the same clock:
    `tokens` (`AlignedTokens`) contributes the grid, seq, modes, and hand
    rows unchanged; `joints` (`AlignedJoints`, 1:1 with the ticks) is the
    material the new values are encoded from. Pure encoding — no time
    handling here.
    """
    from .encoded_tokens import EncodedTokens

    if len(joints) != len(tokens):
        raise ValueError(
            f"joints are not aligned to these tokens: {len(joints)} joint rows "
            f"vs {len(tokens)} ticks — run replay.aligner.align_joints first"
        )

    return EncodedTokens(
        recv_ns=tokens.recv_ns,
        stamp_ns=tokens.stamp_ns,
        seq=tokens.seq,
        arbiter_mode=tokens.arbiter_mode,
        encoder_mode=tokens.encoder_mode,
        values=encode_joints(encoder, joints.q, joints.base_quat, dq_mujoco=joints.dq),
        left_hand=tokens.left_hand,
        right_hand=tokens.right_hand,
        hands_from=tokens.hands_from,
        hand_ticks_before_first=tokens.hand_ticks_before_first,
    )
