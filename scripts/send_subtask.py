#!/usr/bin/env python3
"""Orchestrator stand-in: publish one SubtaskCmd and watch SubtaskState.

Drives a --cortex run_vla.py by hand until the real orchestrator exists —
same topics, same types (idl/cortex_subtask.idl).

Usage (inside the vla container, run_vla.py --cortex running):
    # start a subtask
    python scripts/send_subtask.py --action open \\
        --instruction "Open the fridge door with the right hand."

    # cancel it (same identity)
    python scripts/send_subtask.py --plan-id p-1758351012-0000 --index 0 --cancel

Each start invents a fresh plan_id unless --plan-id pins one; the cmd is
published once, then SubtaskState is echoed for --watch-s seconds so the
IDLE/RUNNING/FAILED transition is visible.
"""

import time
from dataclasses import dataclass, field

import tyro

from common.cyclonedds.config import apply_cyclonedds_xml, load_dds_config
from common.cyclonedds.cortex_msgs import (
    CORTEX_VLA_CMD_TOPIC,
    CORTEX_VLA_STATE_TOPIC,
    SubtaskStatus,
    get_cortex_types,
    subtask_qos,
)


@dataclass
class Config:
    instruction: str = ""
    """VLA english instruction (required unless --cancel)."""

    action: str = "open"
    """Semantic action name."""

    args: list[str] = field(default_factory=list)
    """Signature-order args (informational for the VLA module)."""

    plan_id: str = ""
    """Plan identity; empty invents p-<epoch>-0000."""

    index: int = 0
    """Position within the plan."""

    cancel: bool = False
    """Cancel (plan_id, index) instead of starting it."""

    watch_s: float = 5.0
    """Echo SubtaskState for this long after publishing (0 = fire and forget)."""

    config: str = "config/config.yaml"
    """Network settings — same file run_vla.py reads."""

    domain: int | None = None
    """DDS domain id override."""


def main(config: Config) -> None:
    if not config.cancel and not config.instruction.strip():
        raise SystemExit("--instruction is required unless --cancel")

    dds_cfg = load_dds_config(config.config)
    apply_cyclonedds_xml(dds_cfg.cyclonedds_xml)
    domain = config.domain if config.domain is not None else dds_cfg.domain_id

    from cyclonedds.domain import DomainParticipant
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic

    SubtaskCmd, SubtaskState = get_cortex_types()
    participant = DomainParticipant(domain)
    qos = subtask_qos()
    writer = DataWriter(
        participant, Topic(participant, CORTEX_VLA_CMD_TOPIC, SubtaskCmd), qos=qos
    )
    reader = DataReader(
        participant, Topic(participant, CORTEX_VLA_STATE_TOPIC, SubtaskState), qos=qos
    )

    time.sleep(1.0)  # discovery settle, same as the streamer

    plan_id = config.plan_id or f"p-{int(time.time())}-0000"
    writer.write(
        SubtaskCmd(
            plan_id=plan_id,
            index=config.index,
            action=config.action,
            args=config.args,
            instruction="" if config.cancel else config.instruction,
            cancel=config.cancel,
        )
    )
    kind = "cancel" if config.cancel else f"{config.action} {config.instruction!r}"
    print(f"sent {kind} ({plan_id}#{config.index}) on {CORTEX_VLA_CMD_TOPIC}")

    deadline = time.monotonic() + config.watch_s
    last = None
    while time.monotonic() < deadline:
        for s in reader.take():
            line = (
                f"{SubtaskStatus(s.status).name} {s.plan_id}#{s.index}"
                f" {s.action} progress={s.progress:.2f} detail={s.detail!r}"
            )
            if line != last:
                print(f"  state: {line}")
                last = line
        time.sleep(0.05)


if __name__ == "__main__":
    main(tyro.cli(Config))
