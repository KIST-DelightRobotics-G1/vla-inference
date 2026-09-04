"""Vendored GR00T N1.7 inference core.

Extracted from NVIDIA Isaac-GR00T at commit 5ac4e6b (the pinned fork the
checkpoint was trained against): the import closure of `Gr00tPolicy`, with
the training pipeline (dataset factory, sharded loaders, dist utils,
experiment/eval/deployment) left behind. The directory layout mirrors
upstream so files still diff against it.

Deviations from upstream, each marked with a comment at the site:

- import prefix `gr00t.` -> `vla.policy.gr00t.` (mechanical rewrite);
- `policy/gr00t_policy.py` registers Gr00tN1d7 by importing the two modules
  that own the AutoModel/AutoProcessor registrations directly, instead of
  `import gr00t.model` (whose __init__ pulls the training pipeline);
- the `unitree_g1_sonic_3views` embodiment is registered (embodiment_tags,
  embodiment_configs, projector-index group 11) per the checkpoint handover
  doc UNITREE_G1_SONIC_3VIEWS.md — upstream 5ac4e6b predates it. At load
  time the checkpoint's own processor_config.json / embodiment_id.json carry
  the modality config and projector index; the code-side registration keeps
  the tables consistent and makes the EmbodimentTag resolvable;
- `_patch_mistral_regex_offline()` (processing_gr00t_n1d7.py) stops
  transformers' tokenizer load from phoning home, so the baked-backbone
  image runs fully offline (HF_HUB_OFFLINE=1);
- dead-at-inference code was removed, verified against the deterministic
  regression baseline (tests/smoke_policy.py --baseline): the training
  forward/loss path, the Beta time sampler, requires_grad plumbing,
  training image augmentations and the torchvision pipeline, mask-based
  domain randomization, training-side action normalization,
  mean/std + sin/cos state encodings, RELATIVE<->ABSOLUTE conversion (and
  its pose/action-chunking machinery + scipy), save_pretrained paths,
  the sim-eval wrapper and its in-model collation (which also saved one
  Qwen3VLProcessor load at model construction), dm-tree, ShardedDataset.
  Every trimmed-but-config-reachable branch fails loud with a pointer back
  to upstream. The RTC (real-time chunking) inpainting path in
  `get_action_with_features` is deliberately KEPT — it is an inference
  capability the chunking stage may use for smooth chunk transitions.

This package imports torch/transformers/diffusers/albumentations — it is
only importable inside the inference container, never on the replay path.
"""
