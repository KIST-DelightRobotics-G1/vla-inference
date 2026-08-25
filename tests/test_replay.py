"""Session-replay tests: CSV parsing, 50 Hz resampling, gaps, hand alignment.

The fixtures write real collector-format CSVs (the header shapes are pinned
against kist-data-collector's motion_token_rows.hpp / dex3_cmd_rows.hpp), so
a schema change on the collector side shows up here.
"""

import numpy as np
import pytest

from replay import (
    ARBITER_TELEOP,
    ARBITER_VLA,
    CONTROL_DT_NS,
    align_by_recv_ns,
    blend,
    bracket_timeline,
    build_timeline,
    load_session,
    read_hand_csv,
    read_motion_token_csv,
)

T0 = 1_700_000_000_000_000_000  # arbitrary epoch-ns base


def write_motion_token_csv(path, *, ticks, start_seq=1, arbiter_mode=ARBITER_TELEOP,
                           skip=(), stamp_jitter_ns=0):
    """Write a motion_token.csv whose token values encode the tick index.

    `skip` drops ticks (by offset) without renumbering the rest — exactly how
    a real session records a non-CONTROL period: seq and stamp both jump.
    """
    header = "recv_ns,stamp_ns,seq,arbiter_mode,encoder_mode"
    header += "".join(f",t{i:02d}" for i in range(64))
    lines = [header]
    rng = np.random.default_rng(0)
    for i in range(ticks):
        if i in skip:
            continue
        stamp = T0 + i * CONTROL_DT_NS
        if stamp_jitter_ns:
            stamp += int(rng.integers(-stamp_jitter_ns, stamp_jitter_ns + 1))
        token = ",".join(f"{i + j * 0.001:.7g}" for j in range(64))
        # recv_ns trails stamp_ns by a plausible transport delay
        lines.append(f"{stamp + 200_000},{stamp},{start_seq + i},{arbiter_mode},1,{token}")
    path.write_text("\n".join(lines) + "\n")


def write_hand_cmd_csv(path, *, rows, base_ns=T0, period_ns=CONTROL_DT_NS, value=1.0):
    header = "recv_ns" + "".join(
        f",f{i}_mode,f{i}_q,f{i}_dq,f{i}_tau,f{i}_kp,f{i}_kd" for i in range(7)
    )
    lines = [header]
    for i in range(rows):
        motors = "".join(f",1,{value * (i + 1) + j:.7g},0,0,60,1" for j in range(7))
        lines.append(f"{base_ns + i * period_ns}{motors}")
    path.write_text("\n".join(lines) + "\n")


# ── CSV parsing ──────────────────────────────────────────────────────────────


def test_read_motion_token_csv(tmp_path):
    csv_path = tmp_path / "motion_token.csv"
    write_motion_token_csv(csv_path, ticks=5)
    rows = read_motion_token_csv(csv_path)

    assert len(rows) == 5
    assert rows.tokens.shape == (5, 64)
    assert rows.tokens.dtype == np.float32
    # token[i][j] == i + 0.001*j by construction
    assert rows.tokens[3][0] == pytest.approx(3.0)
    assert rows.tokens[3][10] == pytest.approx(3.01)
    assert rows.seq.tolist() == [1, 2, 3, 4, 5]
    assert (rows.arbiter_mode == ARBITER_TELEOP).all()
    assert (rows.recv_ns > rows.stamp_ns).all()


def test_read_hand_csv_picks_the_q_columns(tmp_path):
    csv_path = tmp_path / "hand_cmd_left.csv"
    write_hand_cmd_csv(csv_path, rows=3)
    recv, q = read_hand_csv(csv_path)

    assert q.shape == (3, 7)
    # row i, motor j -> value*(i+1)+j, so the mode/dq/kp columns were skipped
    assert q[0].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    assert recv[1] - recv[0] == CONTROL_DT_NS


def test_missing_column_is_a_clear_error(tmp_path):
    csv_path = tmp_path / "motion_token.csv"
    csv_path.write_text("recv_ns,stamp_ns,seq\n1,2,3\n")
    with pytest.raises(ValueError, match="missing expected column"):
        read_motion_token_csv(csv_path)


def test_header_only_csv_yields_no_rows(tmp_path):
    csv_path = tmp_path / "motion_token.csv"
    write_motion_token_csv(csv_path, ticks=0)
    assert len(read_motion_token_csv(csv_path)) == 0


# ── alignment / blending ─────────────────────────────────────────────────────


def test_align_by_recv_ns_takes_the_newest_at_or_before():
    src_recv = np.array([100, 200, 300])
    src_vals = np.array([[1.0], [2.0], [3.0]])

    vals, before = align_by_recv_ns(np.array([50, 100, 250, 300, 999]), src_recv, src_vals)
    assert vals.ravel().tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
    # only the t=50 target precedes the first source row
    assert before.tolist() == [True, False, False, False, False]


def test_blend_excludes_start_and_lands_on_end():
    out = blend(np.zeros(3), np.ones(3), 4)
    assert out.shape == (4, 3)
    assert out[0].tolist() == pytest.approx([0.25] * 3)
    assert out[-1].tolist() == pytest.approx([1.0] * 3)
    assert blend(np.zeros(3), np.ones(3), 0).shape == (0, 3)


# ── timeline ─────────────────────────────────────────────────────────────────


def test_contiguous_session_is_one_tick_per_row(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=50)
    timeline = load_session(tmp_path, hand_source="none")

    assert len(timeline) == 50
    assert timeline.recorded_ticks == 50
    assert timeline.gaps == []
    assert not timeline.synthetic.any()
    assert timeline.duration_s == pytest.approx(1.0)
    assert timeline.tokens[7][0] == pytest.approx(7.0)
    # no hand source -> open hand
    assert timeline.left_hand.shape == (50, 7)
    assert not timeline.left_hand.any()


def test_gap_is_blended_across_and_reported(tmp_path):
    # ticks 10..13 missing -> a 4-tick hole between seq 10 and seq 15
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=20, skip=(10, 11, 12, 13))
    timeline = load_session(tmp_path, hand_source="none")

    assert len(timeline) == 20  # the hole is filled, so the grid is intact
    assert timeline.recorded_ticks == 16
    assert len(timeline.gaps) == 1
    gap = timeline.gaps[0]
    assert (gap.after_seq, gap.ticks, gap.filled_ticks) == (10, 4, 4)
    assert not gap.compressed
    assert timeline.synthetic[10:14].all()
    assert not timeline.synthetic[[9, 14]].any()
    # the fill ramps monotonically from the pre-gap token (9) to the post-gap one (14)
    fill = timeline.tokens[10:14, 0]
    assert np.all(np.diff(fill) > 0)
    assert fill[0] > 9.0 and fill[-1] == pytest.approx(14.0)


def test_long_gap_is_compressed_and_flagged(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=200, skip=tuple(range(10, 110)))
    timeline = load_session(tmp_path, hand_source="none", max_hold_ticks=25)

    gap = timeline.gaps[0]
    assert gap.ticks == 100
    assert gap.filled_ticks == 25
    assert gap.compressed
    assert gap.duration_s == pytest.approx(2.0)
    # 100 recorded + 25 filled, not 200
    assert len(timeline) == 125
    assert timeline.worst_gap is gap


def test_stamp_jitter_still_lands_one_row_per_tick(tmp_path):
    # +-5ms jitter on a 20ms grid: rounding must not create or drop ticks
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=100, stamp_jitter_ns=5_000_000)
    timeline = load_session(tmp_path, hand_source="none")

    assert len(timeline) == 100
    assert timeline.gaps == []


def test_teleop_only_filter(tmp_path):
    csv_path = tmp_path / "motion_token.csv"
    write_motion_token_csv(csv_path, ticks=10, arbiter_mode=ARBITER_TELEOP)
    rows = read_motion_token_csv(csv_path)
    rows.arbiter_mode[5:] = ARBITER_VLA

    kept = build_timeline(rows, arbiter_modes=(ARBITER_TELEOP,))
    assert kept.recorded_ticks == 5
    assert kept.arbiter_modes == {ARBITER_TELEOP: 5}

    both = build_timeline(rows)
    assert both.arbiter_modes == {ARBITER_TELEOP: 5, ARBITER_VLA: 5}


def test_filter_that_keeps_nothing_names_the_modes_present(tmp_path):
    csv_path = tmp_path / "motion_token.csv"
    write_motion_token_csv(csv_path, ticks=4, arbiter_mode=ARBITER_VLA)
    rows = read_motion_token_csv(csv_path)
    with pytest.raises(ValueError, match=r"modes \[2\].*vla"):
        build_timeline(rows, arbiter_modes=(ARBITER_TELEOP,))


def test_hand_commands_align_by_recv_ns(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=20)
    # hand commands at half the token rate, starting on the same clock
    write_hand_cmd_csv(tmp_path / "hand_cmd_left.csv", rows=10, period_ns=2 * CONTROL_DT_NS)
    write_hand_cmd_csv(tmp_path / "hand_cmd_right.csv", rows=10, period_ns=2 * CONTROL_DT_NS,
                       value=-1.0)
    timeline = load_session(tmp_path, hand_source="cmd")

    assert timeline.hands_from == "cmd"
    assert timeline.hand_ticks_before_first == 0
    # token ticks 0,1 both fall on hand row 0 (value 1.0); ticks 2,3 on row 1 (2.0)
    assert timeline.left_hand[0][0] == pytest.approx(1.0)
    assert timeline.left_hand[1][0] == pytest.approx(1.0)
    assert timeline.left_hand[2][0] == pytest.approx(2.0)
    assert timeline.right_hand[2][0] == pytest.approx(-2.0)


def test_hands_starting_late_are_clamped_and_counted(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=20)
    # first hand command arrives 5 ticks into the session
    write_hand_cmd_csv(tmp_path / "hand_cmd_left.csv", rows=10, base_ns=T0 + 5 * CONTROL_DT_NS)
    timeline = load_session(tmp_path, hand_source="cmd")

    assert timeline.hand_ticks_before_first == 5
    assert timeline.left_hand[0][0] == pytest.approx(1.0)  # clamped to the first row


def test_missing_hand_csv_degrades_to_open_hands(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=10)
    timeline = load_session(tmp_path, hand_source="cmd")

    assert not timeline.left_hand.any()
    assert not timeline.right_hand.any()


def test_session_without_motion_token_csv_says_why(tmp_path):
    with pytest.raises(FileNotFoundError, match="motion_token.enabled"):
        load_session(tmp_path)


def test_empty_token_stream_is_rejected(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=0)
    with pytest.raises(ValueError, match="no rows"):
        load_session(tmp_path)


def test_bad_hand_source_is_rejected(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=5)
    with pytest.raises(ValueError, match="hand_source"):
        load_session(tmp_path, hand_source="measured")


# ── bracketing ───────────────────────────────────────────────────────────────


def test_bracket_wraps_the_timeline_in_standing_lead_and_blends(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=10)
    timeline = load_session(tmp_path, hand_source="none")

    standing = np.full(64, 5.0, dtype=np.float32)
    open_hand = np.zeros(7, dtype=np.float32)
    stream = bracket_timeline(
        timeline, standing, open_hand, lead_in_ticks=3, lead_out_ticks=2, blend_ticks=4
    )

    assert len(stream) == 3 + 4 + 10 + 4 + 2
    assert stream.left_hand.shape == (len(stream), 7)
    assert stream.right_hand.shape == (len(stream), 7)
    # lead-in / lead-out hold the standing token exactly
    assert np.array_equal(stream.tokens[:3], np.tile(standing, (3, 1)))
    assert np.array_equal(stream.tokens[-2:], np.tile(standing, (2, 1)))
    # per-tick struct view: frame_index counts the published ticks, and the
    # step fields are views of the same arrays the wire will carry
    steps = list(stream)
    assert [s.frame_index for s in steps[:3]] == [0, 1, 2]
    assert np.array_equal(steps[0].token_state, standing)
    assert np.array_equal(steps[-1].left_hand_joints, open_hand)
    assert stream[-1].frame_index == len(stream) - 1
    # blends land on the timeline's first / last tick
    assert stream.tokens[3 + 4 - 1][0] == pytest.approx(timeline.tokens[0][0])
    assert stream.tokens[3 + 4][0] == pytest.approx(timeline.tokens[0][0])
    assert stream.tokens[-3][0] == pytest.approx(standing[0])
    # the replay body is passed through untouched
    assert np.array_equal(stream.tokens[7:17], timeline.tokens)


def test_bracket_with_zero_lead_is_just_the_timeline(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=6)
    timeline = load_session(tmp_path, hand_source="none")
    stream = bracket_timeline(
        timeline,
        np.zeros(64, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        lead_in_ticks=0,
        lead_out_ticks=0,
        blend_ticks=0,
    )
    assert np.array_equal(stream.tokens, timeline.tokens)


def test_bracket_rejects_a_wrong_sized_standing_token(tmp_path):
    write_motion_token_csv(tmp_path / "motion_token.csv", ticks=4)
    timeline = load_session(tmp_path, hand_source="none")
    with pytest.raises(ValueError, match="standing_token must have 64"):
        bracket_timeline(
            timeline, np.zeros(32), np.zeros(7),
            lead_in_ticks=1, lead_out_ticks=1, blend_ticks=1,
        )
