"""Parquet episode replay: LeRobot export -> the same Timeline as CSVs.

The fixtures write real LeRobot-format episode parquets (list-typed columns,
`timestamp` as a length-1 list, `meta/info.json` with the `data_path`
template), so a schema change on the kist-vision-training export side shows
up here. Skipped without pyarrow (the [parquet] extra).
"""

import json

import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402

from replay import TOKEN_DIM  # noqa: E402
from replay.aligner import align_tokens  # noqa: E402
from replay.io.parquet_io import read_tokens, resolve_episode_path  # noqa: E402
from replay.builder import build_timeline  # noqa: E402


def parquet_timeline(path, *, episode_index=None, max_hold_ticks=25):
    """Path resolution -> reader -> align -> build_timeline, as cli.main wires it."""
    from pathlib import Path
    if Path(path).is_dir():
        if episode_index is None:
            raise ValueError(f"{path} is a dataset directory — pass episode_index")
        path = resolve_episode_path(path, episode_index)
    return build_timeline(align_tokens(read_tokens(path)), max_hold_ticks=max_hold_ticks)


FPS = 50


def write_episode_parquet(path, *, ticks, skip=()):
    """Write an episode whose token values encode the tick index.

    Token channel 0 is the tick index; hand joint 0 is index+0.5 (left) and
    index+0.25 (right), so alignment mistakes show up as value mismatches.
    `skip` drops those tick indices, emulating frames the export removed.
    """
    kept = [i for i in range(ticks) if i not in skip]
    tokens = [[float(i)] + [0.0] * (TOKEN_DIM - 1) for i in kept]
    left = [[i + 0.5] + [0.0] * 6 for i in kept]
    right = [[i + 0.25] + [0.0] * 6 for i in kept]
    table = pa.table(
        {
            "action.motion_token": pa.array(tokens, type=pa.list_(pa.float64())),
            "teleop.left_hand_joints": pa.array(left, type=pa.list_(pa.float32())),
            "teleop.right_hand_joints": pa.array(right, type=pa.list_(pa.float32())),
            "timestamp": pa.array([[i / FPS] for i in kept], type=pa.list_(pa.float32())),
            "frame_index": pa.array([[i] for i in kept], type=pa.list_(pa.int64())),
        }
    )
    pq.write_table(table, path)
    return path


def write_dataset(root, *, episodes):
    """A minimal LeRobot dataset root: meta/info.json + chunked parquets."""
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "total_episodes": len(episodes),
        "chunks_size": 1000,
        "fps": FPS,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    for idx, ticks in enumerate(episodes):
        write_episode_parquet(
            root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet", ticks=ticks
        )
    return root


def test_read_episode_matches_csv_shapes(tmp_path):
    path = write_episode_parquet(tmp_path / "e.parquet", ticks=10)
    tokens = read_tokens(path)

    assert len(tokens) == 10
    assert tokens.values.shape == (10, TOKEN_DIM)
    assert tokens.values.dtype == np.float32
    np.testing.assert_array_equal(tokens.values[:, 0], np.arange(10, dtype=np.float32))
    np.testing.assert_array_equal(tokens.seq, np.arange(10))
    # The export's 20 ms grid survives the float32-seconds round trip to
    # within nanoseconds — far below the half-tick the grid rounding absorbs.
    assert np.abs(np.diff(tokens.stamp_ns) - 20_000_000).max() < 1_000
    # Hands share the token clock, so downstream alignment is the identity.
    for hand, offset in ((tokens.left_hand, 0.5), (tokens.right_hand, 0.25)):
        recv, q = hand
        assert recv is tokens.recv_ns
        assert q.shape == (10, 7)
        np.testing.assert_allclose(q[:, 0], np.arange(10) + offset)


def test_load_episode_contiguous(tmp_path):
    path = write_episode_parquet(tmp_path / "e.parquet", ticks=20)
    aligned = align_tokens(read_tokens(path))
    timeline = build_timeline(aligned)

    assert len(timeline) == 20
    assert not timeline.gaps
    assert timeline.recorded_ticks == 20
    assert aligned.hands_from == "cmd"
    np.testing.assert_array_equal(timeline.tokens[:, 0], np.arange(20, dtype=np.float32))
    np.testing.assert_allclose(timeline.left_hand[:, 0], np.arange(20) + 0.5)


def test_load_episode_gap_is_blended(tmp_path):
    # Ticks 4..6 dropped by the export -> one 3-tick gap on the grid.
    path = write_episode_parquet(tmp_path / "e.parquet", ticks=10, skip=(4, 5, 6))
    timeline = parquet_timeline(path)

    assert len(timeline) == 10  # gap filled back to the full grid
    assert len(timeline.gaps) == 1
    gap = timeline.gaps[0]
    assert (gap.after_seq, gap.ticks, gap.compressed) == (3, 3, False)
    # The fill ramps from token 3 toward token 7, end-inclusive (alpha=(i+1)/3).
    np.testing.assert_allclose(
        timeline.tokens[4:7, 0], 3.0 + 4.0 * np.array([1, 2, 3]) / 3.0, rtol=1e-6
    )
    np.testing.assert_array_equal(timeline.synthetic[4:7], True)
    np.testing.assert_array_equal(timeline.synthetic[:4], False)


def test_load_episode_from_dataset_root(tmp_path):
    root = write_dataset(tmp_path / "ds", episodes=[5, 8, 12])

    timeline = parquet_timeline(root, episode_index=1)
    assert len(timeline) == 8

    with pytest.raises(ValueError, match="episode_index"):
        parquet_timeline(root)  # dataset dir without an index
    with pytest.raises(ValueError, match="out of range"):
        resolve_episode_path(root, 3)
    with pytest.raises(FileNotFoundError, match="dataset root"):
        resolve_episode_path(tmp_path, 0)  # no meta/info.json here


def test_missing_column_is_descriptive(tmp_path):
    table = pa.table({
        "timestamp": pa.array([[0.0]], type=pa.list_(pa.float32())),
        "frame_index": pa.array([[0]], type=pa.list_(pa.int64())),
    })
    pq.write_table(table, tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="action.motion_token"):
        read_tokens(tmp_path / "bad.parquet")
