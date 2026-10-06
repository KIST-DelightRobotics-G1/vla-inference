#!/usr/bin/env python3
"""Manual check: type probe scores, watch the subtask machine transition.

A stand-in for ProgressProbe — the runner's `raw = probe.read()` becomes a
number you type. Everything downstream is the real thing, wired exactly as
run_vla.py --probe --cortex does it:

    raw -> ProgressMonitor.update(raw, now) -> Reading
        -> SubtaskMachine.on_progress(state, progress, now) -> Effect
    monitor.reset() whenever machine.subtask_id() changes

Two clocks:

  offline (default)   No DDS, no numpy beyond what vla.cortex pulls in.
                      Virtual clock: each reading advances --dt seconds and
                      the bridge's 10 Hz tick is simulated in between, so a
                      5 s plateau is ten lines — or one `0.62 x10`.

  --domain N          Real CortexBridge on DDS: its 10 Hz thread ticks the
                      machine and publishes SubtaskState, so a cortex mock
                      on the same domain sees the DONE/FAILED go by. The
                      clock is real: `0.62 x50` streams 50 readings at 10 Hz
                      (= a 5 s plateau). Start subtasks with the `cmd` line
                      below or with scripts/send_subtask.py --domain N.

REPL:

    0.42            one raw reading
    0.62 x20        20 readings (offline: dt apart; DDS: at 10 Hz)
    hold 6          repeat the last raw for 6 s (the plateau shortcut)
    cmd Open it.    start a subtask (action "open", fresh index) —
                    ignored while one is RUNNING: `cancel` first
    cancel          cancel the running subtask
    dt 0.5          offline step size
    state           print the current SubtaskState fields
    q               quit

Try (offline):

    cmd Open the fridge.          -> RUNNING
    0.1 x4   0.3 x4   0.5 x4      -> RUNNING, slope healthy
    0.62 x12                      -> grey-zone plateau: stays RUNNING (no verdict)
    cmd Grab the cup.  0.4 x28    -> low & flat past 10 s, held 3 s -> FAILED x3 -> IDLE
    cmd Close it.      0.2  0.8   -> >= 0.75: DONE at once, slope irrelevant

Thresholds are CLI flags so the contract table in the design doc (3.1.1)
can be tried live: --stuck-value-threshold 0.70 turns a grey-zone plateau
into STALLED -> FAILED instead of RUNNING.
"""

import shlex
import time
from dataclasses import dataclass

import tyro

from common.cyclonedds.cortex_msgs import SubtaskStatus
from vla.chunking import ChunkCursor
from vla.cortex import Effect, SubtaskMachine
from vla.progress_probe import ProgressMonitor

TICK_S = 0.1  # the bridge's STATE_PERIOD_S


@dataclass
class Config:
    domain: int | None = None
    """DDS domain for a real CortexBridge; None = offline with a virtual clock.
    Keep OFF the robot bus (0) for bench runs."""

    dt: float = 0.5
    """Offline: seconds of virtual time per reading (the inference period)."""

    step_timeout_s: float = 0.0
    """SubtaskMachine step timeout (0 = off)."""

    done_threshold: float = 0.75
    stuck_value_threshold: float = 0.55
    slope_stuck_threshold: float = 0.027
    window_s: float = 5.0
    stall_min_elapsed_s: float = 10.0
    stall_hold_s: float = 3.0


class Driver:
    def __init__(self, config: Config):
        self.config = config
        self.cursor = ChunkCursor()
        self.machine = SubtaskMachine(step_timeout_s=config.step_timeout_s)
        self.monitor = ProgressMonitor(
            done_threshold=config.done_threshold,
            stuck_value_threshold=config.stuck_value_threshold,
            slope_stuck_threshold=config.slope_stuck_threshold,
            window_s=config.window_s,
            stall_min_elapsed_s=config.stall_min_elapsed_s,
            stall_hold_s=config.stall_hold_s,
        )
        self.online = config.domain is not None
        self.bridge = None
        self.vnow = 0.0
        self.last_raw: float | None = None
        self.last_subtask_id: tuple[str, int] = ("", 0)
        self.last_status = self.machine.state_fields().status
        self.cmd_index = 0

    # ── clock ─────────────────────────────────────────────────────────────

    def now(self) -> float:
        return time.monotonic() if self.online else self.vnow

    def advance(self) -> None:
        """One inference period passes: real sleep online, virtual + simulated
        bridge ticks offline (the ticks drive the DONE/FAILED x3 -> IDLE)."""
        if self.online:
            time.sleep(TICK_S)
            return
        ticks = max(1, round(self.config.dt / TICK_S))
        for _ in range(ticks):
            self.vnow += TICK_S
            self.apply(self.machine.tick(self.vnow))
            self.report_transition("tick")

    # ── the runner's probe block, verbatim in shape ───────────────────────

    def reading(self, raw: float) -> None:
        current = self.machine.subtask_id()
        if current == ("", 0):
            # The runner runs no inference without a RUNNING subtask
            # (bridge.instruction() is None) — so no probe reading either.
            print(f"    ({self.machine.state_fields().status.name}: no RUNNING subtask, "
                  "runner would not infer — reading skipped)")
            self.advance()
            return
        if current != self.last_subtask_id:
            self.monitor.reset()
            self.last_subtask_id = current
        now = self.now()
        r = self.monitor.update(raw, now)
        effect = self.machine.on_progress(r.state, r.progress, now)
        self.apply(effect)
        self.last_raw = raw

        slope = "  warmup " if r.slope_per_s is None else f"{r.slope_per_s:+.3f}/s"
        flags = " ".join(
            n for n, v in (("stalled", r.stalled), ("plateaued", r.plateaued), ("done", r.done)) if v
        )
        f = self.machine.state_fields()
        print(
            f"t={now:7.1f}  raw={raw:.2f}  prog={r.progress:.2f}  slope={slope}  "
            f"verdict={r.state.value:<7}  {flags:<18} | {f.status.name}"
            + (f"  effect={effect.value}" if effect is not Effect.NONE else "")
        )
        self.report_transition("probe")
        self.advance()

    def apply(self, effect: Effect) -> None:
        # What the bridge does with the effect — the runner should too.
        if effect is Effect.FREEZE:
            self.cursor.freeze()

    def report_transition(self, cause: str) -> None:
        f = self.machine.state_fields()
        if f.status is not self.last_status:
            print(
                f"    ==> {self.last_status.name} -> {f.status.name}"
                f"  detail={f.detail!r}  progress={f.progress:.2f}  ({cause})"
            )
            self.last_status = f.status

    # ── commands ──────────────────────────────────────────────────────────

    def cmd(self, instruction: str) -> None:
        plan_id = f"p-fake-{self.cmd_index:04d}"
        effect = self.machine.on_cmd(
            plan_id=plan_id, index=self.cmd_index, action="open",
            instruction=instruction, cancel=False, now=self.now(),
        )
        self.cmd_index += 1
        self.apply(effect)
        print(f"cmd {plan_id}  instruction={instruction!r}")
        self.report_transition("cmd")

    def cancel(self) -> None:
        plan_id, index = self.machine.subtask_id()
        if not plan_id:
            print("nothing RUNNING to cancel")
            return
        effect = self.machine.on_cmd(
            plan_id=plan_id, index=index, action="", instruction="",
            cancel=True, now=self.now(),
        )
        self.apply(effect)
        self.report_transition("cancel")

    def state(self) -> None:
        f = self.machine.state_fields()
        print(
            f"status={f.status.name} plan_id={f.plan_id!r} index={f.index} "
            f"progress={f.progress:.2f} detail={f.detail!r} "
            f"cursor_frozen={self.cursor.stats().get('frozen')}"
        )

    def hold(self, seconds: float) -> None:
        if self.last_raw is None:
            print("no reading to hold yet")
            return
        period = TICK_S if self.online else self.config.dt
        for _ in range(max(1, round(seconds / period))):
            self.reading(self.last_raw)

    # ── loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        if self.online:
            from vla.cortex import CortexBridge

            self.bridge = CortexBridge(self.machine, self.cursor)
            self.bridge.start(domain_id=self.config.domain)
            print(f"bridge up on domain {self.config.domain} — real clock, "
                  f"{self.config.window_s:.0f}s window is {self.config.window_s:.0f} real seconds")
        else:
            print(f"offline — virtual clock, dt={self.config.dt}s per reading, "
                  f"{self.config.window_s:.0f}s window = "
                  f"{round(self.config.window_s / self.config.dt)} readings")
        print(f"thresholds: done>={self.config.done_threshold} "
              f"stuck_value={self.config.stuck_value_threshold} "
              f"slope<{self.config.slope_stuck_threshold}/s window={self.config.window_s}s")
        self.state()
        try:
            while True:
                try:
                    line = input("probe> ").strip()
                except EOFError:
                    break
                if not line:
                    continue
                if not self.dispatch(line):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if self.bridge is not None:
                self.bridge.stop()
            self.state()

    def dispatch(self, line: str) -> bool:
        parts = shlex.split(line)
        head = parts[0].lower()
        if head in ("q", "quit", "exit"):
            return False
        if head == "cmd":
            self.cmd(" ".join(parts[1:]) or "Open the fridge door with the right hand.")
        elif head == "cancel":
            self.cancel()
        elif head == "state":
            self.state()
        elif head == "dt" and len(parts) == 2:
            self.config.dt = float(parts[1])
            print(f"dt={self.config.dt}s")
        elif head == "hold" and len(parts) == 2:
            self.hold(float(parts[1]))
        else:
            try:
                raw = float(parts[0])
            except ValueError:
                print("?  a number, `0.6 x20`, hold N, cmd <text>, cancel, dt N, state, q")
                return True
            repeat = 1
            if len(parts) == 2 and parts[1].lower().startswith("x"):
                repeat = int(parts[1][1:])
            for _ in range(repeat):
                self.reading(raw)
        return True


if __name__ == "__main__":
    Driver(tyro.cli(Config)).run()
