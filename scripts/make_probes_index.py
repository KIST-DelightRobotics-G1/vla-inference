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
    index: dict[str, str] = {}
    for path in files:
        meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
        prompt = meta.get("prompt")
        name = os.path.basename(path)
        if prompt is None:
            print(f"  skip  {name}: no prompt in meta (add to probes.json by hand)")
            continue
        if prompt in index:
            print(f"error: {name} and {index[prompt]} are both fitted on {prompt!r}", file=sys.stderr)
            return 1
        index[prompt] = name
        print(f"  {name}  <-  {prompt!r}  (extractor {meta.get('extractor_checkpoint', '?')})")
    out = os.path.join(directory, "probes.json")
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
