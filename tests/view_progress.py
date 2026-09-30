#!/usr/bin/env python3
"""Live progress-probe viewer: the three camera views over a progress plot,
with s/f marks — a native pyqtgraph window (not a pytest test).

A second process next to run_vla.py --probe. It never touches the runner:
it tails the runner's progress JSONL (shared/probe/, host-mounted) and
subscribes to the same camera topics with its own ColorSubscribers.

    ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓   one black-framed strip,
    ┃ left_wrist  ┃  ego_view   ┃ right_wrist ┃   views joined edge to edge,
    ┗━━━━━━━━━━━━━┻━━━━━━━━━━━━━┻━━━━━━━━━━━━━┛   --fps (default 10)
    ┌─────────────────────────────────────────┐
    │ progress  thin grey: raw (7 Hz)         │   ▒ grey zone stuck..done
    │           bold blue: ProgressMonitor    │   | DONE (green) / STALLED (red)
    │           dashed:   runner's own progress (when logged)   ● s / ● f
    ├─────────────────────────────────────────┤
    │ raw 0.61  progress 0.64  slope ...  state RUNNING  ep#3  ...          │
    └─────────────────────────────────────────┘

The bold line is recomputed HERE from raw with the real ProgressMonitor and
the thresholds given on the command line — so the plot shows exactly what
the state machine would decide under those thresholds, and the same log can
be replayed under different ones. Episode boundaries (probe-only runs have
no subtask to reset the running max) come from the `r` key, or automatically
when the runner's own logged progress drops (a cortex subtask change).

Keys (window focused):
    s / f   mark success / fail at the latest sample (green / red dot)
    u       undo the last mark
    r       new episode: reset the running max, dashed line, episode += 1
    space   toggle auto-scroll (frozen: zoom/pan freely)
    q       quit

Marks go to <log stem>.marks.jsonl next to the log, one JSON row each
({"kind": "success"|"fail"|"reset", "episode", "t_key", "t_sample", "raw",
"progress"}); `u` drops the last row and rewrites the file.

Run (vla container; run.sh passes DISPLAY + the X socket):
    python tests/view_progress.py                       # newest log, live cameras
    python tests/view_progress.py --no-cameras
    python tests/view_progress.py --replay shared/probe/<run>.jsonl --speed 4
    python tests/view_progress.py --stuck-value-threshold 0.70   # contract case 2
"""

import glob
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
import tyro
from pyqtgraph.Qt import QtCore, QtWidgets

from vla.progress_probe import ProgressMonitor, ProgressState

TAIL_MS = 20            # JSONL poll period
NEWER_FILE_CHECK_S = 1.0
STRIP_SEP_PX = 4        # black seam between camera views in the strip
STRIP_FRAME_PX = 6      # black frame around the whole strip


def _fit_height(rgb: np.ndarray, height: int) -> np.ndarray:
    """Nearest-neighbour resize to `height` keeping aspect (views of unequal size)."""
    h, w = rgb.shape[:2]
    new_w = max(1, round(w * height / h))
    ys = (np.arange(height) * h // height)
    xs = (np.arange(new_w) * w // new_w)
    return rgb[ys][:, xs]


@dataclass
class Config:
    probe_dir: str = "shared/probe"
    """Where run_vla.py --probe writes its JSONL (the newest one is followed)."""

    log: str | None = None
    """Follow this JSONL instead of the newest in --probe-dir."""

    replay: str | None = None
    """Replay a finished JSONL at its recorded pace (no cameras)."""

    speed: float = 1.0
    """Replay speed multiplier."""

    cameras: dict[str, str] = field(
        default_factory=lambda: {
            "left_wrist": "left_wrist",
            "ego_view": "head",
            "right_wrist": "right_wrist",
        }
    )
    """View name -> ext-sensor-io camera name (same mapping as run_vla.py).
    Dict order = panel order, left to right: wrists flank the head view."""

    no_cameras: bool = False
    """Plot only — no DDS at all."""

    fps: float = 10.0
    """Camera strip refresh rate. 10 Hz keeps the strip's average display lag
    (half the period + decode) under ~100 ms; decode cost is unchanged by
    this (every H.264 frame is decoded anyway), only compose+paint scales."""

    window_s: float = 60.0
    """Visible time window while auto-scrolling."""

    config: str = "config/config.yaml"
    """Network settings (dds.domain_id + cyclonedds XML), as run_vla.py."""

    domain: int | None = None
    """DDS domain id override."""

    # ProgressMonitor thresholds — the design doc's four knobs.
    done_threshold: float = 0.70
    stuck_value_threshold: float = 0.55
    slope_stuck_threshold: float = 0.027
    slope_window_s: float = 5.0


# ── sources ──────────────────────────────────────────────────────────────────


def newest_log(directory: str) -> str | None:
    files = [f for f in glob.glob(os.path.join(directory, "*.jsonl")) if not f.endswith(".marks.jsonl")]
    return max(files, key=os.path.getmtime) if files else None


class LogTail:
    """Read a runner JSONL as it grows; switch to a newer file when one appears."""

    def __init__(self, path: str | None, directory: str, follow_newest: bool):
        self.directory = directory
        self.follow_newest = follow_newest
        self.path: str | None = None
        self._file = None
        self._buf = ""
        self._last_check = 0.0
        if path is not None:
            self._open(path)

    def _open(self, path: str) -> None:
        if self._file is not None:
            self._file.close()
        self.path = path
        self._file = open(path, "r")
        self._buf = ""
        print(f"[view_progress] following {path}")

    def poll(self) -> tuple[list[dict], bool]:
        """(new rows, switched) — switched=True means a new run's file was opened."""
        switched = False
        now = time.monotonic()
        if self.follow_newest and now - self._last_check >= NEWER_FILE_CHECK_S:
            self._last_check = now
            candidate = newest_log(self.directory)
            if candidate is not None and candidate != self.path:
                self._open(candidate)
                switched = True
        if self._file is None:
            return [], switched
        rows = []
        chunk = self._file.read()
        if chunk:
            self._buf += chunk
            *lines, self._buf = self._buf.split("\n")   # last piece may be partial
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn write; the next poll completes it
        return rows, switched


class Replay:
    """Release the rows of a finished JSONL at their recorded pace."""

    def __init__(self, path: str, speed: float):
        self.path = path
        with open(path) as f:
            self._rows = [json.loads(l) for l in f if l.strip()]
        self._i = 0
        self._speed = speed
        self._t0 = self._rows[0]["t"] if self._rows else 0.0
        self._start = time.monotonic()

    def poll(self) -> tuple[list[dict], bool]:
        elapsed = (time.monotonic() - self._start) * self._speed
        out = []
        while self._i < len(self._rows) and self._rows[self._i]["t"] - self._t0 <= elapsed:
            out.append(self._rows[self._i])
            self._i += 1
        return out, False

    @property
    def done(self) -> bool:
        return self._i >= len(self._rows)


class Marks:
    """<stem>.marks.jsonl — appended per mark, rewritten on undo."""

    def __init__(self, log_path: str):
        self.path = log_path[: -len(".jsonl")] + ".marks.jsonl"
        self.rows: list[dict] = []
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.rows = [json.loads(l) for l in f if l.strip()]

    def add(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def undo(self) -> dict | None:
        if not self.rows:
            return None
        last = self.rows.pop()
        with open(self.path, "w") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")
        return last


# ── the window ───────────────────────────────────────────────────────────────

GROUND = (22, 25, 31)   # window ground: dark grey, so the strip's black frame shows
GREY = (150, 150, 150)
BLUE = (60, 140, 255)
GREEN = (60, 200, 90)
RED = (240, 70, 70)


class ProgressViewer(QtWidgets.QWidget):
    def __init__(self, config: Config, source, cameras: dict):
        super().__init__()
        self.config = config
        self.source = source
        self.cameras = cameras          # view -> ColorSubscriber (may be empty)
        self.monitor = ProgressMonitor(
            done_threshold=config.done_threshold,
            stuck_value_threshold=config.stuck_value_threshold,
            slope_stuck_threshold=config.slope_stuck_threshold,
            window_s=config.slope_window_s,
        )
        self.marks: Marks | None = None
        self.replay = isinstance(source, Replay)
        self.follow = True
        self._reset_run()

        self.setWindowTitle("progress probe")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(GROUND)
        layout.addWidget(self.glw)

        # rows 0-1: the camera strip — the views side by side as ONE image in
        # one black-framed box (no gaps), view name + frame age overlaid.
        views = list(cameras) or []
        self.views = views
        self.cam_labels: dict[str, pg.TextItem] = {}
        self.strip: pg.ImageItem | None = None
        if views:
            vb = self.glw.addViewBox(row=0, col=0, colspan=len(views), lockAspect=True, enableMenu=False)
            vb.invertY(True)
            vb.setMouseEnabled(False, False)
            vb.setDefaultPadding(0.01)
            self.strip = pg.ImageItem(axisOrder="row-major")
            self.strip.setBorder(pg.mkPen((0, 0, 0), width=STRIP_FRAME_PX))
            vb.addItem(self.strip)
            for view in views:
                label = pg.TextItem(view, color=(235, 235, 235), fill=(0, 0, 0, 140), anchor=(0, 0))
                label.setZValue(10)
                vb.addItem(label)
                self.cam_labels[view] = label
            self._strip_vb = vb
        ncols = max(1, len(views))
        plot_row = 1 if views else 0

        # the plot (row 2 under the cameras, row 0 without them)
        self.plot = self.glw.addPlot(row=plot_row, col=0, colspan=ncols)
        self.plot.setLabel("left", "progress")
        self.plot.setLabel("bottom", "s since run start")
        self.plot.setYRange(-0.05, 1.05, padding=0)
        self.plot.setMouseEnabled(x=True, y=False)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.addLegend(offset=(10, 10))
        self.zone = pg.LinearRegionItem(
            values=(config.stuck_value_threshold, config.done_threshold),
            orientation="horizontal", movable=False, brush=(128, 128, 128, 45),
            pen=pg.mkPen((128, 128, 128, 120), style=QtCore.Qt.PenStyle.DashLine),
        )
        self.plot.addItem(self.zone)
        self.raw_curve = self.plot.plot(pen=pg.mkPen(GREY, width=1), name="raw")
        self.logged_curve = self.plot.plot(
            pen=pg.mkPen((255, 200, 60), width=1, style=QtCore.Qt.PenStyle.DashLine),
            name="runner progress",
        )
        self.prog_curve = self.plot.plot(pen=pg.mkPen(BLUE, width=3), name="progress (monitor)")
        self.scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w", width=1))
        self.plot.addItem(self.scatter)
        self._event_lines: list[pg.InfiniteLine] = []

        # status line under the plot
        self.status = self.glw.addLabel(
            "waiting for samples…", row=plot_row + 1, col=0, colspan=ncols, justify="left"
        )
        if views:
            self.glw.ci.layout.setRowStretchFactor(0, 3)
            self.glw.ci.layout.setRowStretchFactor(1, 4)

        self.tail_timer = QtCore.QTimer(self)
        self.tail_timer.timeout.connect(self.on_tail)
        self.tail_timer.start(TAIL_MS)
        if cameras:
            self.cam_timer = QtCore.QTimer(self)
            self.cam_timer.timeout.connect(self.on_cameras)
            self.cam_timer.start(int(1000 / config.fps))

    # ── run / episode state ───────────────────────────────────────────────

    def _reset_run(self) -> None:
        self.t0: float | None = None
        self.ts: list[float] = []
        self.raws: list[float] = []
        self.progs: list[float] = []
        self.logged_ts: list[float] = []
        self.logged: list[float] = []
        self._last_logged: float | None = None
        self._last_state = ProgressState.RUNNING
        self.episode = 1
        self.last_row: dict | None = None
        self.last_reading = None
        self.monitor.reset()
        self._pending_resets: list[float] = []   # replay: reset marks' t_sample
        self._mark_points: list[tuple[float, float, str]] = []

    def _open_marks(self, log_path: str) -> None:
        self.marks = Marks(log_path)
        self.setWindowTitle(os.path.basename(log_path)[: -len(".jsonl")])
        # Existing marks (viewer restart / replay): draw s/f, schedule resets.
        for row in self.marks.rows:
            if row["kind"] == "reset":
                self._pending_resets.append(row["t_sample"])
            elif "progress" in row and row.get("t_sample") is not None:
                self._mark_points.append((row["t_sample"], row["progress"], row["kind"]))
        self._pending_resets.sort()

    def _x(self, t: float) -> float:
        return t - (self.t0 if self.t0 is not None else t)

    def _add_line(self, t: float, color, dashed: bool = False) -> None:
        style = QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine
        line = pg.InfiniteLine(pos=self._x(t), angle=90, pen=pg.mkPen(color, width=1, style=style))
        self.plot.addItem(line)
        self._event_lines.append(line)

    def _clear_plot_items(self) -> None:
        for line in self._event_lines:
            self.plot.removeItem(line)
        self._event_lines.clear()
        self.scatter.clear()
        for c in (self.raw_curve, self.prog_curve, self.logged_curve):
            c.clear()

    # ── data ──────────────────────────────────────────────────────────────

    def on_tail(self) -> None:
        rows, switched = self.source.poll()
        if switched or (self.marks is None and self.source.path is not None):
            if switched:
                self._clear_plot_items()
                self._reset_run()
            self._open_marks(self.source.path)
        for row in rows:
            self._process(row)
        if rows:
            self._redraw()
        self._update_status()

    def _process(self, row: dict) -> None:
        t = row.get("t")
        raw = row.get("raw", row.get("progress"))
        if t is None or raw is None:
            return
        if self.t0 is None:
            self.t0 = t
        # Replay: a recorded reset mark lands here.
        while self._pending_resets and self._pending_resets[0] <= t:
            self._pending_resets.pop(0)
            self._new_episode(t, record=False)
        # Runner-side reset (cortex subtask change): its running max fell.
        logged = row["progress"] if "raw" in row else None
        if logged is not None:
            if self._last_logged is not None and logged < self._last_logged - 1e-6:
                self._new_episode(t, record=False)
            self._last_logged = logged
            self.logged_ts.append(t)
            self.logged.append(logged)

        reading = self.monitor.update(raw, t)
        self.ts.append(t)
        self.raws.append(raw)
        self.progs.append(reading.progress)
        if reading.state is not self._last_state:
            if reading.state is ProgressState.DONE:
                self._add_line(t, GREEN)
            elif reading.state is ProgressState.STALLED:
                self._add_line(t, RED)
            self._last_state = reading.state
        self.last_row = row
        self.last_reading = reading

    def _redraw(self) -> None:
        if not self.ts:
            return
        xs = np.asarray(self.ts) - self.t0
        self.raw_curve.setData(xs, np.asarray(self.raws))
        self.prog_curve.setData(xs, np.asarray(self.progs))
        if self.logged:
            self.logged_curve.setData(np.asarray(self.logged_ts) - self.t0, np.asarray(self.logged))
        if self._mark_points:
            self.scatter.setData(
                x=[self._x(t) for t, _, _ in self._mark_points],
                y=[p for _, p, _ in self._mark_points],
                brush=[pg.mkBrush(GREEN if k == "success" else RED) for _, _, k in self._mark_points],
                symbol=["t1" if k == "success" else "t" for _, _, k in self._mark_points],
            )
        if self.follow:
            x_last = xs[-1]
            self.plot.setXRange(max(0.0, x_last - self.config.window_s), x_last + 1.0, padding=0)

    def _update_status(self) -> None:
        if self.last_reading is None:
            return
        r = self.last_reading
        if self.replay:
            age = f"replay {self._x(self.last_row['t']):.1f} s"
        else:
            age = f"sample age {(time.time() - self.last_row['t']) * 1e3:.0f} ms"
        slope = "warmup" if r.slope_per_s is None else f"{r.slope_per_s:+.3f}/s"
        flags = " ".join(n for n, v in (("stalled", r.stalled), ("plateaued", r.plateaued)) if v)
        marks = len([m for m in (self.marks.rows if self.marks else []) if m["kind"] != "reset"])
        follow = "" if self.follow else "   <b>[PAUSED]</b>"
        self.status.setText(
            f"raw <b>{r.raw:.3f}</b>   progress <b>{r.progress:.3f}</b>   slope {slope}   "
            f"state <b>{r.state.value}</b> {flags}   ep#<b>{self.episode}</b>   marks {marks}   "
            f"{age}{follow}   <span style='color:#888'>keys: s f u r space q</span>"
        )

    def on_cameras(self) -> None:
        """Compose the latest frames into one strip: [view0 | sep | view1 | ...]."""
        tiles, texts = [], []
        height = None
        for view, sub in self.cameras.items():
            frame, age = sub.latest()
            if frame is None:
                tiles.append(None)
                texts.append(f"{view} — no frames")
                continue
            rgb = frame.rgb
            height = rgb.shape[0] if height is None else min(height, rgb.shape[0])
            tiles.append(rgb)
            texts.append(f"{view}  {age * 1e3:.0f} ms")
        if height is None:
            return  # nothing decoded yet
        width0 = next(t.shape[1] for t in tiles if t is not None)
        parts, x_starts, x = [], [], 0
        for i, tile in enumerate(tiles):
            if tile is None:
                tile = np.zeros((height, width0, 3), np.uint8)
            elif tile.shape[0] != height:
                tile = _fit_height(tile, height)
            if i > 0:
                parts.append(np.zeros((height, STRIP_SEP_PX, 3), np.uint8))
                x += STRIP_SEP_PX
            x_starts.append(x)
            parts.append(tile)
            x += tile.shape[1]
        strip = np.concatenate(parts, axis=1)
        self.strip.setImage(strip, autoLevels=False, levels=(0, 255))
        for view, text, x0 in zip(self.views, texts, x_starts):
            label = self.cam_labels[view]
            label.setText(text)
            label.setPos(x0 + 6, 4)

    # ── keys ──────────────────────────────────────────────────────────────

    def keyPressEvent(self, event) -> None:
        key = event.key()
        K = QtCore.Qt.Key
        if key == K.Key_S:
            self._mark("success")
        elif key == K.Key_F:
            self._mark("fail")
        elif key == K.Key_U:
            self._undo()
        elif key == K.Key_R:
            if self.last_row is not None:
                self._new_episode(self.last_row["t"], record=True)
                self._redraw()
        elif key == K.Key_Space:
            self.follow = not self.follow
            self._redraw()
            self._update_status()
        elif key == K.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def _mark(self, kind: str) -> None:
        if self.marks is None or self.last_reading is None:
            return
        r = self.last_reading
        self.marks.add({
            "kind": kind, "episode": self.episode,
            "t_key": round(time.time(), 3), "t_sample": self.last_row["t"],
            "raw": round(r.raw, 4), "progress": round(r.progress, 4),
        })
        self._mark_points.append((self.last_row["t"], r.progress, kind))
        print(f"[mark] {kind} ep#{self.episode} progress={r.progress:.3f} raw={r.raw:.3f}")
        self._redraw()
        self._update_status()

    def _undo(self) -> None:
        if self.marks is None:
            return
        last = self.marks.undo()
        if last is None:
            return
        if last["kind"] == "reset":
            # The episode line stays on screen (the monitor already reset);
            # only the record and the counter are taken back.
            self.episode = max(1, self.episode - 1)
        elif self._mark_points:
            self._mark_points.pop()
        print(f"[mark] undo {last['kind']}")
        self._redraw()
        self._update_status()

    def _new_episode(self, t: float, *, record: bool) -> None:
        self.monitor.reset()
        self._last_state = ProgressState.RUNNING
        self.episode += 1
        self._add_line(t, (200, 200, 200), dashed=True)
        if record and self.marks is not None:
            self.marks.add({
                "kind": "reset", "episode": self.episode,
                "t_key": round(time.time(), 3), "t_sample": t,
            })
        print(f"[episode] #{self.episode}")

    def closeEvent(self, event) -> None:
        for sub in self.cameras.values():
            sub.stop()
        super().closeEvent(event)


# ── main ─────────────────────────────────────────────────────────────────────


def start_cameras(config: Config) -> dict:
    """view -> started ColorSubscriber; empty on any DDS problem (plot still runs)."""
    if config.no_cameras or config.replay is not None:
        return {}
    try:
        from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config
        from cyclonedds.domain import DomainParticipant
        from vla.io.realsense import ColorSubscriber, color_topic_for
    except ImportError as e:
        print(f"[view_progress] cameras off ({e})")
        return {}
    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id
    participant = DomainParticipant(domain)
    subs = {}
    for view, name in config.cameras.items():
        sub = ColorSubscriber(view, topic=color_topic_for(name))
        sub.start(participant=participant)
        subs[view] = sub
    return subs


def main(config: Config) -> None:
    if config.replay is not None:
        source = Replay(config.replay, config.speed)
    else:
        path = config.log or newest_log(config.probe_dir)
        if path is None:
            print(f"[view_progress] no JSONL in {config.probe_dir} yet — waiting for the runner")
        source = LogTail(path, config.probe_dir, follow_newest=config.log is None)

    pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
    app = pg.mkQApp("progress probe")
    cameras = start_cameras(config)
    viewer = ProgressViewer(config, source, cameras)
    viewer.resize(1280, 860)
    viewer.show()
    pg.exec()


if __name__ == "__main__":
    main(tyro.cli(Config))
