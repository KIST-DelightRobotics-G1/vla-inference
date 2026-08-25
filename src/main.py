"""Unified entry point (planned): pick the token source with an argument.

    python src/run.py vla    [options...]   -> vla.runner    (today: scripts/run_vla.py)
    python src/run.py replay [options...]   -> replay.cli    (today: scripts/replay_session.py)

TODO: wire the argparse dispatch between the two publishers. Until then the
scripts/ entry points remain the way to run each mode.
"""

import sys


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(
        "not wired yet — use scripts/run_vla.py (policy) or "
        "scripts/replay_session.py (recorded session) for now"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
