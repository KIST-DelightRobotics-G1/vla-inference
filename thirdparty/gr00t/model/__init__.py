"""Register the GR00T N1.7 model and processor with transformers' Auto* classes.

``Gr00tPolicy.__init__`` does ``import thirdparty.gr00t.model`` for exactly this
side effect: the checkpoint's ``config.json`` says ``model_type: "Gr00tN1d7"``
and ``processor_config.json`` says ``processor_class: "Gr00tN1d7Processor"``,
and the two modules below call ``AutoConfig/AutoModel/AutoProcessor.register``
at import time so ``from_pretrained`` can resolve those names.

LOCAL CUT (vs upstream ``gr00t/model/__init__.py``): upstream imports
``.gr00t_n1d7.setup`` (``Gr00tN1d7Pipeline``) and ``.registry``, which pull in
``DatasetFactory``, ``base_config``, ``training_config`` and the lerobot
dataset loaders — the finetuning stack, plus pandas/tqdm/pyyaml. Inference
only needs the registrations, so those files are not vendored.
"""

from . import gr00t_n1d7  # noqa: F401  (Gr00tN1d7 + Gr00tN1d7Processor registration)
