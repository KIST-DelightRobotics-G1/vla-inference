#!/usr/bin/env python3
"""Executable launcher — thin shell over `main.main()` (see src/main.py)."""

import sys

from main import main

if __name__ == "__main__":
    main(sys.argv[1:])
