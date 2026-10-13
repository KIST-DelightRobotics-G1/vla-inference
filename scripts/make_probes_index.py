#!/usr/bin/env python3
"""Write probes.json for a directory of probe .pt files, from their metadata.

Each probe .pt records the prompt it was fitted on (meta["prompt"]); the
runner's --probe index maps that prompt to the file. Typing the prompt by
hand invites a one-character mismatch, so generate the index instead:

    python scripts/make_probes_index.py shared/probe/checkpoint-10000
    → shared/probe/checkpoint-10000/probes.json

A .pt without a recorded prompt is listed but skipped (add it by hand). Two
.pt fitted on the same prompt is an error — keep one, or pick the other at
run time with send_subtask.py --args probe=<name>.

Hand-written parts of an existing probes.json survive regeneration:
timeout-only entries ("<prompt>": <seconds> — a sentence with no head that
the runner ends as DONE after that long) and the per-sentence thresholds
of object entries ({"pt": ..., "done_threshold": 0.8}) — those are re-attached
to the .pt found for that prompt, or dropped with a note if none is.

Runs inside the vla container (needs torch).
"""

import glob
import json
import os
import sys

import torch


def main(directory: str) -> int:
    files = sorted(glob.glob(os.path.join(directory, "*.pt")))
    if not files:
        print(f"no .pt in {directory}", file=sys.stderr)
        return 1
    out = os.path.join(directory, "probes.json")
    index: dict = {}
    thresholds: dict[str, dict] = {}
    if os.path.exists(out):
        with open(out) as f:
            for prompt, value in json.load(f).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    index[prompt] = value
                    print(f"  keep  DONE at {value:g}s (no head)  <-  {prompt!r}")
                elif isinstance(value, dict):
                    thresholds[prompt] = {k: v for k, v in value.items() if k != "pt"}
    for path in files:
        meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
        prompt = meta.get("prompt")
        name = os.path.basename(path)
        if prompt is None:
            print(f"  skip  {name}: no prompt in meta (add to probes.json by hand)")
            continue
        if isinstance(index.get(prompt), str):
            print(f"error: {name} and {index[prompt]} are both fitted on {prompt!r}", file=sys.stderr)
            return 1
        if prompt in index:
            print(f"  note  {name} replaces the {index[prompt]:g}s timeout entry for {prompt!r}")
        over = thresholds.pop(prompt, None)
        index[prompt] = {"pt": name, **over} if over else name
        extra = "" if not over else "  keep " + " ".join(f"{k}={v}" for k, v in over.items())
        print(f"  {name}  <-  {prompt!r}  (extractor {meta.get('extractor_checkpoint', '?')}){extra}")
    for prompt, over in thresholds.items():
        print(f"  drop  thresholds {over} for {prompt!r}: no .pt fitted on it any more")
    with open(out, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {out}: {len(index)} prompts")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
