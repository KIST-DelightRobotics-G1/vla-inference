# Vendored: Isaac-GR00T N1.7 inference path

- **Upstream**: https://github.com/NVIDIA/Isaac-GR00T
- **Commit**: `9c7e746b2cd37a810070a98ef41d290a07e806c2` (2026-07-08)
- **License**: Apache-2.0 — see `LICENSE` here; upstream headers are kept on every file
- **Re-vendor**: `bash scripts/vendor_gr00t.sh [<commit>]`

## Why this commit

The `rab-v2b-20260806` checkpoints (`checkpoint-18000` and friends) were
finetuned with the KIST fork `foodbanana/Isaac-GR00T @ 5ac4e6b`. That fork
branched from NVIDIA at `9c7e746`, and across the files vendored here the
two differ **only** in `policy/server_client.py` (the socket fix carried as
`patches/0001`). Model, processor and config code are byte-identical, so
this is the NVIDIA commit that matches the checkpoint.

A GR00T checkpoint does not record the code it was built with, and its
`config.json` is a delta filled in from defaults in
`configs/model/gr00t_n1d7.py`. Newer upstream code can therefore load the
checkpoint and silently emit wrong actions (renamed modules are randomly
initialised behind a warning; preprocessing changes lower quality with no
error at all). **Change the commit only together with a checkpoint that was
finetuned against it**, then review `git diff thirdparty/gr00t` and run
`scripts/smoke_test_policy.py`. NVIDIA `main` is already ahead in
`processing_gr00t_n1d7.py` (`238ef45`, inference image path) — do not take it.

## What is vendored (20 files, overwritten by the script)

```
policy/gr00t_policy.py                 Gr00tPolicy: from_pretrained -> processor -> model.get_action -> decode_action
policy/policy.py                       BasePolicy / PolicyWrapper
policy/server_client.py                ZMQ PolicyServer / PolicyClient (remote mode)  [+ patches/0001]
model/gr00t_n1d7/gr00t_n1d7.py         Gr00tN1d7 (HF PreTrainedModel) + action head; AutoModel.register
model/gr00t_n1d7/processing_gr00t_n1d7.py  Gr00tN1d7Processor: preprocessing, (un)normalisation; AutoProcessor.register
model/gr00t_n1d7/image_augmentations.py
model/modules/dit.py                   flow-matching DiT
model/modules/embodiment_conditioned_mlp.py
model/modules/qwen3_backbone.py        Cosmos-Reason2-2B VLM backbone (gated HF repo)
configs/model/gr00t_n1d7.py            Gr00tN1d7Config — defaults for keys absent from config.json
configs/data/embodiment_configs.py
data/collator/collators.py
data/state_action/{state_action_processor,action_chunking,pose}.py
data/{interfaces,types,embodiment_tags,utils}.py
utils/initial_actions.py
```

Chosen by tracing imports from `gr00t.policy.gr00t_policy`; the training
plumbing that upstream's package `__init__` files drag in (`setup.py`,
`DatasetFactory`, `base_config`, `training_config`, lerobot loaders,
`replay_policy`, `run_gr00t_server`) is left out.

## What is ours (never touched by the script)

| Path | Purpose |
|---|---|
| every `__init__.py` | package markers; two of them are the **training cuts**: `model/__init__.py` registers the model/processor without importing `setup.py`; `configs/model/__init__.py` is a 3-line `register_model_config` stub instead of upstream's glob-import of the training configs |
| `patches/0001-policyclient-close-abandoned-socket.patch` | `PolicyClient._init_socket` closes the abandoned socket with `linger=0`, otherwise a dead PolicyServer makes the client unkillable by Ctrl+C |
| `VENDORED_FROM.md`, `LICENSE` | this file; upstream licence |

Upstream's `gr00t/__init__.py` (test-only `from_pretrained` monkeypatches)
is intentionally not vendored.

## Import rewrite

The script rewrites `gr00t.*` → `thirdparty.gr00t.*` in import statements
only. String literals such as `"gr00t.initial_actions"` (a file-format tag in
`utils/initial_actions.py`) are left alone on purpose.

## Editing rules

- `server_client.py`, `gr00t_policy.py` validation/batching, logging, the
  image input path: fine to change. Add the change as a new
  `patches/NNNN-*.patch` so re-vendoring keeps it:
  `git diff --relative=thirdparty/gr00t thirdparty/gr00t/<file> > thirdparty/gr00t/patches/NNNN-<name>.patch`
- `gr00t_n1d7.py`, `dit.py`, `qwen3_backbone.py`, `embodiment_conditioned_mlp.py`,
  `configs/model/gr00t_n1d7.py`, and the normalisation logic in
  `processing_gr00t_n1d7.py`: **do not change independently of training.**
  They must stay byte-identical to the code the checkpoint was finetuned with.
