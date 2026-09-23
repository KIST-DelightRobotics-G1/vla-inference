"""ProgressProbe: read task progress [0, 1] out of the VLA's own DiT latent.

A linear probe (ridge regression, nn.Linear(2048, 1) + feature statistics)
fitted on the 2048-dim mean-pooled output of `action_head.vl_self_attention`
— see vla_data/README.md §2. The heavy part is the VLA itself: a forward
hook copies the latent out of the `get_action` forward that runs anyway,
so `read()` after `predict()` costs one dot product on the GPU (fused
normalize + linear) plus one .item() sync.

The probe is NOT standalone: its weights live in the latent space of the
one checkpoint it was fitted against (`meta["extractor_checkpoint"]`, the
published groot_n17_0820succ_6k_step4500) under that checkpoint's training
prompt. A different extractor or prompt gives meaningless scores — attach()
and check_prompt() print what was recorded so the operator can verify the
pairing.

This module imports torch — inference-container only, like sonic_policy.
"""

import torch

FEATURE_DIM = 2048


class ProgressProbe:
    """Load one probe .pt; hook a Gr00tN1d7 model; score each prediction.

    Lifecycle:

        probe = ProgressProbe("/data/vla/progress_probe/probe_succ3v.pt")
        probe.check_prompt(config.prompt)      # warn on distribution mismatch
        probe.attach(policy.torch_model)       # forward hook, model untouched
        ...
        chunk = policy.predict(observation)    # hook fires inside this call
        progress = probe.read()                # the score for THAT observation
    """

    def __init__(self, probe_path: str):
        payload = torch.load(probe_path, map_location="cpu", weights_only=False)
        linear = torch.nn.Linear(FEATURE_DIM, 1)
        linear.load_state_dict(payload["w"])
        linear.eval()
        w = linear.weight.detach().squeeze(0).float()   # (2048,)
        b = float(linear.bias.detach().item())
        mu = payload["mu"].float()                       # (2048,)
        sd = payload["sd"].float()                       # (2048,)
        # Fuse (x - mu)/sd then Linear into ONE dot product:
        #   y = w·((x-mu)/sd) + b = (w/sd)·x + (b - (w/sd)·mu)
        # Saves 2048 subs + 2048 divs per read().
        self._w_eff = (w / sd).contiguous()              # (2048,)
        self._b_eff = b - float((self._w_eff * mu).sum().item())
        self.meta: dict = payload.get("meta", {})
        self._feature: torch.Tensor | None = None
        self._handle = None
        extractor = self.meta.get("extractor_checkpoint", "<unrecorded>")
        print(f"[ProgressProbe] {probe_path}")
        print(f"[ProgressProbe] fitted on extractor: {extractor}")
        print("[ProgressProbe] the running checkpoint must be that same model")

    def check_prompt(self, prompt: str) -> None:
        """Warn when the runner's prompt differs from the probe's fit prompt."""
        fit_prompt = self.meta.get("prompt")
        if fit_prompt is not None and fit_prompt != prompt:
            print(
                "[ProgressProbe] WARNING: prompt differs from the probe's fit "
                f"prompt — scores may not be meaningful.\n"
                f"  fit:     {fit_prompt!r}\n"
                f"  running: {prompt!r}"
            )

    def attach(self, model: torch.nn.Module) -> None:
        """Hook `model.action_head.vl_self_attention` (fires once per predict)."""
        module = model.action_head.vl_self_attention

        def grab(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # (B, seq, 2048) -> (2048,): stay on GPU (no .cpu() mid-forward).
            # read() forces the sync at the end via .item() — this preserves
            # CUDA overlap during predict().
            feature = tensor.detach().float().mean(dim=1).squeeze(0)
            # Lazy-move fused weights to the feature's device on first fire.
            if self._w_eff.device != feature.device:
                self._w_eff = self._w_eff.to(feature.device)
            self._feature = feature

        self._handle = module.register_forward_hook(grab)

    def read(self) -> float | None:
        """Progress of the latest prediction, or None before the first one."""
        if self._feature is None:
            return None
        with torch.no_grad():
            # One dot product + bias. .item() drains the sync predict() left.
            return float((self._w_eff @ self._feature).item()) + self._b_eff

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
