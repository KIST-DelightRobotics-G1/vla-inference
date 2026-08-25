"""Loading a recording into a replay timeline.

Two sources produce the same `ReplayTimeline`: a raw collector session
directory (`load_session`, CSVs) and a LeRobot training-export episode
(`load_episode`, parquet).
"""

from pathlib import Path

import numpy as np

from .io.csv_io import read_hand_csv, read_motion_token_csv
from .io.parquet_io import read_episode_parquet, resolve_episode_path
from .timeline import ReplayTimeline, build_timeline


def load_session(
    session_dir: str | Path,
    *,
    arbiter_modes: tuple[int, ...] | None = None,
    max_hold_ticks: int = 25,
    hand_source: str = "cmd",
) -> ReplayTimeline:
    """Build a replay timeline from a collector session directory.

    Args:
        session_dir: a `sessions/<YYYYMMDD_HHMMSS>/` directory.
        arbiter_modes: see `build_timeline`.
        max_hold_ticks: see `build_timeline`.
        hand_source: `"cmd"` reads `hand_cmd_{side}.csv` (the commanded
            targets — the action twin of the token, and the same quantity
            `LatentActionStep` carries), `"state"` reads `hand_{side}.csv`
            (measured), `"none"` publishes the open-hand pose. A missing or
            empty file degrades to the open-hand pose rather than failing:
            the hands are independent of the whole-body token.

    Raises:
        FileNotFoundError: no `motion_token.csv` in the directory.
        ValueError: the token stream is unusable (see `build_timeline`).
    """
    session_dir = Path(session_dir)
    token_csv = session_dir / "motion_token.csv"
    if not token_csv.exists():
        raise FileNotFoundError(
            f"{token_csv} not found — the session was recorded without the "
            f"motion_token stream (config.yaml `motion_token.enabled`), so it "
            f"carries no latent action to replay"
        )

    rows = read_motion_token_csv(token_csv)

    hands: dict[str, tuple[np.ndarray, np.ndarray] | None] = {"left": None, "right": None}
    if hand_source in ("cmd", "state"):
        stem = "hand_cmd" if hand_source == "cmd" else "hand"
        for side in ("left", "right"):
            path = session_dir / f"{stem}_{side}.csv"
            if path.exists():
                recv, q = read_hand_csv(path)
                if len(recv):
                    hands[side] = (recv, q)
    elif hand_source != "none":
        raise ValueError(f"hand_source must be 'cmd', 'state' or 'none', got {hand_source!r}")

    return build_timeline(
        rows,
        left_hand=hands["left"],
        right_hand=hands["right"],
        arbiter_modes=arbiter_modes,
        max_hold_ticks=max_hold_ticks,
        hands_from=hand_source,
    )


def load_episode(
    path: str | Path,
    *,
    episode_index: int | None = None,
    max_hold_ticks: int = 25,
) -> ReplayTimeline:
    """Build a replay timeline from a LeRobot training-export episode.

    Args:
        path: either one `episode_XXXXXX.parquet` file, or a dataset root
            (the directory holding `meta/info.json`) — the latter requires
            `episode_index` and resolves the file via the dataset's own
            `data_path` template.
        episode_index: which episode, when `path` is a dataset root.
        max_hold_ticks: see `build_timeline`.

    Hand targets always come from the episode's `teleop.*_hand_joints`
    columns — the export carries no measured-hand alternative, so there is
    no `hand_source` choice here.

    Raises:
        ImportError: pyarrow is not installed (the `[parquet]` extra).
        FileNotFoundError / ValueError: see `resolve_episode_path` and
            `read_episode_parquet`.
    """
    path = Path(path)
    if path.is_dir():
        if episode_index is None:
            raise ValueError(
                f"{path} is a dataset directory — pass episode_index to pick an episode"
            )
        path = resolve_episode_path(path, episode_index)

    rows, hands = read_episode_parquet(path)
    return build_timeline(
        rows,
        left_hand=hands["left"],
        right_hand=hands["right"],
        max_hold_ticks=max_hold_ticks,
        hands_from="cmd",
    )


def load_reencoded_episode(
    path: str | Path,
    encoder,
    *,
    episode_index: int | None = None,
    joint_source: str = "state",
    max_hold_ticks: int = 25,
) -> ReplayTimeline:
    """Like `load_episode`, but the tokens are RE-ENCODED from the episode's
    recorded joints (g1 mode) instead of taken from `action.motion_token`.

    Use when the gearsonic-side SONIC decoder checkpoint differs from the
    one that ran at collection time — the recorded latents don't transfer,
    but the joints do, through the new checkpoint's paired encoder.

    Args:
        encoder: `model_encoder.onnx` path, or a callable
            (N, 1762) -> (N, 64) (see `replay.io.joint_encoder`).
        joint_source: "state" (measured, default) or "wbc" (commanded).
        Other args: see `load_episode`.
    """
    from .io.joint_encoder import encode_episode_joints

    path = Path(path)
    if path.is_dir():
        if episode_index is None:
            raise ValueError(
                f"{path} is a dataset directory — pass episode_index to pick an episode"
            )
        path = resolve_episode_path(path, episode_index)

    rows, hands = encode_episode_joints(path, encoder, joint_source=joint_source)
    return build_timeline(
        rows,
        left_hand=hands["left"],
        right_hand=hands["right"],
        max_hold_ticks=max_hold_ticks,
        hands_from="cmd",
    )
