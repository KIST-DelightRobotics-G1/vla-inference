"""`python -m replay` — same entry point as scripts/replay_session.py."""

import tyro

from .cli import Config, main

if __name__ == "__main__":
    main(tyro.cli(Config))
