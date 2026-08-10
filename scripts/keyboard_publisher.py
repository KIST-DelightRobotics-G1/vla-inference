#!/usr/bin/env python3
"""Operator console: publishes keystrokes to the runner over ZMQ.

Single keys are sent as-is (p, k, i, [, ] — see runner.py for meanings).
Press 't' to type a new language prompt (sent as ``prompt:<text>``).
Press 'q' or Ctrl-C to quit.
"""

import sys
import termios
import tty

import tyro

from kist_vla.io.keyboard import DEFAULT_KEYBOARD_PORT, KeyboardPublisher


def read_single_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main(port: int = DEFAULT_KEYBOARD_PORT):
    publisher = KeyboardPublisher(port=port)
    print("Keys: p=pause/resume  k=start/stop C++ loop  i=initial pose")
    print("      [=left hand  ]=right hand  t=change prompt  q=quit")
    try:
        while True:
            key = read_single_key()
            if key in ("q", "\x03"):  # q or Ctrl-C
                break
            if key == "t":
                prompt = input("\nNew prompt: ").strip()
                if prompt:
                    publisher.send_prompt(prompt)
                    print(f'Sent prompt: "{prompt}"')
                continue
            publisher.send_key(key)
            print(f"Sent key: {key!r}")
    finally:
        publisher.close()
        print("Keyboard publisher closed.")


if __name__ == "__main__":
    tyro.cli(main)
