"""Third reader: recorded joints -> SONIC tokens via the offline encoder.

Re-encodes a LeRobot episode's whole-body joints into motion tokens with the
GEAR-SONIC encoder (g1 mode 0), producing the same `(MotionTokenRows, hands)`
the other readers do — so gaps, gate, bracket, and publisher are all shared,
and gearsonic needs no change. Unlike the recorded tokens (which are latents
of the SONIC checkpoint that ran at collection time), these tokens come from
whatever encoder ONNX you pass — re-encoding is how a session survives a
decoder-checkpoint change.

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

Requires ``onnxruntime`` (the ``[encode]`` extra); imported lazily. The
encoder model is ``model_encoder.onnx`` from the public GEAR-SONIC HF repo —
it must be the encoder PAIRED with the decoder gearsonic runs.
"""

from pathlib import Path

import numpy as np

from ..constants import CONTROL_DT_NS, TOKEN_DIM
from .motion_token_rows import MotionTokenRows
from .parquet_io import read_episode_parquet

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

# The 29 body joints inside the recording's 43-dim state/wbc vectors
# (modality order: legs 0:12, waist 12:15, left_arm 15:22, left_hand 22:29,
# right_arm 29:36, right_hand 36:43) — hands excluded, MuJoCo order kept.
BODY_IN_STATE43 = np.array(list(range(0, 22)) + list(range(29, 36)), dtype=np.int64)


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
    q_mujoco: np.ndarray, base_quat: np.ndarray, *, dt: float = CONTROL_DT_NS / 1e9
) -> np.ndarray:
    """Recorded joints + base quats -> encoder obs_dict rows (T, 1762).

    Args:
        q_mujoco: (T, 29) joint positions, MuJoCo/Unitree order (rad).
        base_quat: (T, 4) pelvis quaternion, wxyz.
        dt: recording tick period (50 Hz).

    Per tick t the 10 "future" frames are t, t+5, ..., t+45, clamped to the
    last tick (hold final pose) — exactly `target_frame()` with playing=true.
    Velocities are the central finite difference of the positions.
    """
    q_mujoco = np.asarray(q_mujoco, dtype=np.float64)
    base_quat = np.asarray(base_quat, dtype=np.float64)
    T = len(q_mujoco)
    assert q_mujoco.shape == (T, NUM_BODY_JOINTS) and base_quat.shape == (T, 4)

    # MuJoCo -> IsaacLab order (scatter), then finite-difference velocities.
    q_isaac = np.empty_like(q_mujoco)
    q_isaac[:, MUJOCO_TO_ISAACLAB] = q_mujoco
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


def encode_episode_joints(
    path: str | Path,
    encoder,
    *,
    joint_source: str = "state",
) -> tuple[MotionTokenRows, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Re-encode one episode's joints into tokens; same shapes as the readers.

    Args:
        path: one `episode_XXXXXX.parquet`.
        encoder: callable (N, 1762) -> (N, 64) (see `load_onnx_encoder`), or
            an ONNX path handed to it.
        joint_source: "state" (measured — the motion that actually happened,
            default) or "wbc" (the commanded targets).

    Returns `(rows, hands)` exactly like `read_episode_parquet`, with
    `rows.tokens` replaced by the re-encoded ones.
    """
    if isinstance(encoder, (str, Path)):
        encoder = load_onnx_encoder(encoder)

    # Timestamps / seq / hands come from the normal parquet reader; only the
    # tokens are replaced.
    rows, hands = read_episode_parquet(path)

    column = {"state": "observation.state", "wbc": "action.wbc"}.get(joint_source)
    if column is None:
        raise ValueError(f"joint_source must be 'state' or 'wbc', got {joint_source!r}")

    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=[column, "observation.root_orientation"])
    state43 = np.asarray(table[column].to_pylist(), dtype=np.float64)
    base_quat = np.asarray(
        table["observation.root_orientation"].to_pylist(), dtype=np.float64
    )

    q_mujoco = state43[:, BODY_IN_STATE43]
    obs = assemble_encoder_obs(q_mujoco, base_quat)
    tokens = np.asarray(encoder(obs), dtype=np.float32).reshape(-1, TOKEN_DIM)
    if len(tokens) != len(rows):
        raise ValueError(f"encoder returned {len(tokens)} tokens for {len(rows)} ticks")

    rows.tokens = tokens
    return rows, hands
