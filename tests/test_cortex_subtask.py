"""Cortex subtask protocol: machine transitions, cursor freeze, bridge cycle.

The machine is pure logic (clock injected), the cursor freeze is asserted
tick by tick, and the bridge runs one cycle against injected fake DDS
endpoints — no network anywhere.
"""

import numpy as np

from common.cyclonedds.cortex_msgs import SubtaskStatus
from vla.chunking import ChunkCursor
from vla.cortex import CortexBridge, Effect, SubtaskMachine
from vla.policy import ActionChunk


def running_machine(**kwargs) -> SubtaskMachine:
    m = SubtaskMachine(**kwargs)
    effect = m.on_cmd(
        plan_id="p-1-0001",
        index=0,
        action="open",
        instruction="Open the fridge door with the right hand.",
        cancel=False,
        now=100.0,
    )
    assert effect is Effect.NONE
    return m


def indexed_chunk() -> ActionChunk:
    tokens = np.zeros((40, 64), dtype=np.float32)
    tokens[:, 0] = np.arange(40)
    return ActionChunk(
        motion_token=tokens,
        left_hand_joints=np.zeros((40, 7), dtype=np.float32),
        right_hand_joints=np.zeros((40, 7), dtype=np.float32),
    )


# ── machine transitions ───────────────────────────────────────────────────────


def test_idle_reports_empty_identity():
    f = SubtaskMachine().state_fields()
    assert (f.plan_id, f.index, f.action) == ("", 0, "")
    assert f.status is SubtaskStatus.IDLE
    assert f.detail == ""


def test_valid_cmd_runs_and_exposes_instruction():
    m = running_machine()
    assert m.instruction() == "Open the fridge door with the right hand."
    f = m.state_fields()
    assert f.status is SubtaskStatus.RUNNING
    assert (f.plan_id, f.index, f.action) == ("p-1-0001", 0, "open")


def test_matching_cancel_freezes_to_idle_cancelled():
    m = running_machine()
    effect = m.on_cmd(
        plan_id="p-1-0001", index=0, action="", instruction="", cancel=True, now=101.0
    )
    assert effect is Effect.FREEZE
    assert m.instruction() is None
    f = m.state_fields()
    assert f.status is SubtaskStatus.IDLE
    assert f.detail == "cancelled"
    assert f.plan_id == ""  # IDLE identity is empty per the ICD


def test_stale_cancel_ignored():
    m = running_machine()
    effect = m.on_cmd(
        plan_id="p-1-0001", index=7, action="", instruction="", cancel=True, now=101.0
    )
    assert effect is Effect.NONE
    assert m.state_fields().status is SubtaskStatus.RUNNING


def test_preempt_replaces_instruction_without_effect():
    m = running_machine()
    effect = m.on_cmd(
        plan_id="p-1-0001",
        index=1,
        action="pick",
        instruction="Pick up the cucumber.",
        cancel=False,
        now=102.0,
    )
    assert effect is Effect.NONE
    assert m.instruction() == "Pick up the cucumber."
    assert m.state_fields().index == 1


def test_move_to_fails_unsupported():
    m = SubtaskMachine()
    effect = m.on_cmd(
        plan_id="p-1-0002", index=0, action="move_to", instruction="", cancel=False, now=1.0
    )
    assert effect is Effect.NONE  # nothing was running, nothing to freeze
    f = m.state_fields()
    assert f.status is SubtaskStatus.FAILED
    assert f.detail.startswith("unsupported:")
    assert f.plan_id == "p-1-0002"  # FAILED names the rejected cmd


def test_unsupported_preempting_running_task_freezes():
    m = running_machine()
    effect = m.on_cmd(
        plan_id="p-1-0001", index=1, action="open", instruction="  ", cancel=False, now=101.0
    )
    assert effect is Effect.FREEZE
    assert m.state_fields().status is SubtaskStatus.FAILED
    assert m.instruction() is None


def test_failed_recovers_on_next_valid_cmd():
    m = SubtaskMachine()
    m.on_cmd(plan_id="p", index=0, action="move_to", instruction="", cancel=False, now=1.0)
    m.on_cmd(plan_id="p", index=1, action="open", instruction="Open.", cancel=False, now=2.0)
    assert m.state_fields().status is SubtaskStatus.RUNNING


def test_timeout_fails_with_appendix_detail():
    m = running_machine(step_timeout_s=30.0)
    assert m.tick(now=129.9) is Effect.NONE
    assert m.tick(now=130.0) is Effect.FREEZE
    f = m.state_fields()
    assert f.status is SubtaskStatus.FAILED
    assert f.detail == "timeout 30s"
    assert m.tick(now=131.0) is Effect.NONE  # fires once


def test_timeout_disabled_by_default():
    m = running_machine()
    assert m.tick(now=1e9) is Effect.NONE
    assert m.state_fields().status is SubtaskStatus.RUNNING


# ── cursor freeze ─────────────────────────────────────────────────────────────


def test_freeze_pins_last_emitted_step():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk())
    for _ in range(5):
        cursor.step()
    cursor.freeze()
    for _ in range(100):  # far past hold_ticks: never goes silent
        step = cursor.step()
        assert step is not None
        assert step.motion_token[0] == 4  # the last emitted step (index 4)
    assert cursor.stats()["frozen"] == 100


def test_freeze_before_any_step_pins_step_zero():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk())
    cursor.freeze()
    assert cursor.step().motion_token[0] == 0


def test_freeze_without_chunk_stays_silent():
    cursor = ChunkCursor()
    cursor.freeze()
    assert cursor.step() is None


def test_push_unpins_freeze():
    cursor = ChunkCursor()
    cursor.push(indexed_chunk())
    cursor.step()
    cursor.freeze()
    cursor.push(indexed_chunk(), skip_ticks=3)
    assert cursor.step().motion_token[0] == 3  # normal splice resumes


# ── bridge (fake endpoints) ───────────────────────────────────────────────────


class FakeReader:
    def __init__(self):
        self.pending = []

    def take(self):
        out, self.pending = self.pending, []
        return out


class FakeWriter:
    def __init__(self):
        self.written = []

    def write(self, sample):
        self.written.append(sample)


def test_bridge_cycle_applies_cmd_and_publishes_state():
    SubtaskCmd, _ = __import__(
        "common.cyclonedds.cortex_msgs", fromlist=["get_cortex_types"]
    ).get_cortex_types()
    machine = SubtaskMachine()
    cursor = ChunkCursor()
    cursor.push(indexed_chunk())
    cursor.step()

    bridge = CortexBridge(machine, cursor)
    reader, writer = FakeReader(), FakeWriter()
    bridge._reader, bridge._writer = reader, writer
    _, bridge._state_type = __import__(
        "common.cyclonedds.cortex_msgs", fromlist=["get_cortex_types"]
    ).get_cortex_types()

    reader.pending.append(
        SubtaskCmd(
            plan_id="p-9-0000", index=0, action="open",
            args=["fridge_door"], instruction="Open.", cancel=False,
        )
    )
    bridge._cycle()
    assert bridge.instruction() == "Open."
    assert writer.written[-1].status == int(SubtaskStatus.RUNNING)
    assert writer.written[-1].plan_id == "p-9-0000"

    reader.pending.append(
        SubtaskCmd(
            plan_id="p-9-0000", index=0, action="", args=[], instruction="", cancel=True
        )
    )
    bridge._cycle()
    assert bridge.instruction() is None
    assert writer.written[-1].status == int(SubtaskStatus.IDLE)
    assert writer.written[-1].detail == "cancelled"
    # the freeze landed: the cursor repeats the last emitted step
    assert cursor.step().motion_token[0] == 0
    assert cursor.stats()["frozen"] == 1
