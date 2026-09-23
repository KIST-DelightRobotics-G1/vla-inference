#!/usr/bin/env python3
"""Manual check: the cortex bridge alone on real DDS, no model, no robot.

The protocol-level half of validating --cortex (the tests/view_*.py family):
a CortexBridge with a real CycloneDDS reader/writer and a ChunkCursor primed
with one synthetic chunk, plus a stand-in for the inference loop that steps
the cursor and prints every instruction change. Drive it from another shell
with scripts/send_subtask.py on the same domain and watch:

    start   -> "instruction -> 'Open ...'"  + state RUNNING on the wire
    cancel  -> "instruction -> None"        + state IDLE detail='cancelled'
              + "cursor frozen: repeating step N" (the posture hold)
    move_to -> state FAILED detail='unsupported: ...'

Run (vla container, NO robot needed — pick a domain off the robot bus):
    python tests/run_cortex_bridge_standalone.py --domain 42
    python scripts/send_subtask.py --domain 42 --action open --instruction "Open."
"""

import time
from dataclasses import dataclass

import numpy as np
import tyro

from vla.chunking import ChunkCursor
from vla.cortex import CortexBridge, SubtaskMachine
from vla.policy import ActionChunk


@dataclass
class Config:
    domain: int = 42
    """DDS domain — keep OFF the robot bus (0) for bench runs."""

    step_timeout_s: float = 0.0
    """Subtask timeout to exercise the FAILED path (0 = off)."""

    run_s: float = 60.0
    """How long to keep the bridge up."""


def main(config: Config) -> None:
    cursor = ChunkCursor()
    machine = SubtaskMachine(step_timeout_s=config.step_timeout_s)
    bridge = CortexBridge(machine, cursor)
    bridge.start(domain_id=config.domain)

    tokens = np.zeros((40, 64), dtype=np.float32)
    tokens[:, 0] = np.arange(40)
    chunk = ActionChunk(
        motion_token=tokens,
        left_hand_joints=np.zeros((40, 7), dtype=np.float32),
        right_hand_joints=np.zeros((40, 7), dtype=np.float32),
    )

    print(f"bridge up on domain {config.domain} for {config.run_s:.0f}s — "
          "drive it with scripts/send_subtask.py")
    last_instruction = None
    deadline = time.monotonic() + config.run_s
    try:
        while time.monotonic() < deadline:
            instruction = bridge.instruction()
            if instruction != last_instruction:
                print(f"instruction -> {instruction!r}")
                last_instruction = instruction
                if instruction is not None:
                    # the inference loop stand-in: a RUNNING task gets a chunk
                    cursor.push(chunk)
            step = cursor.step()
            if step is not None and cursor.stats()["frozen"] == 1:
                print(f"cursor frozen: repeating step {step.motion_token[0]:.0f}")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        print(f"cursor stats: {cursor.stats()}")


if __name__ == "__main__":
    main(tyro.cli(Config))
