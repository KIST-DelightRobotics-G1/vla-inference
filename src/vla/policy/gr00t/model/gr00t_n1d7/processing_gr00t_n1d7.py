# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict
import warnings

import albumentations as A
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

from vla.policy.gr00t.configs.data.embodiment_configs import ModalityConfig
from vla.policy.gr00t.data.embodiment_tags import EmbodimentTag
from vla.policy.gr00t.data.interfaces import BaseProcessor
from vla.policy.gr00t.data.state_action.state_action_processor import StateActionProcessor
from vla.policy.gr00t.data.utils import parse_modality_configs

from .image_augmentations import apply_images, build_eval_image_transform


try:
    from transformers import Qwen3VLProcessor
except ImportError:
    Qwen3VLProcessor = None

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)

### Projector-index assignments, declared as ``{projector_index: {tags}}``.
#
# This grouped form is the source of truth: a tag only shares a projector with
# another tag if it is deliberately placed inside that index's set, so an
# accidental collision can't slip in unnoticed. Multiple tags share an index
# only when they describe the *same physical embodiment*. To add a brand-new
# embodiment, give it an unused index; to add a data-source/subtask variant of
# an existing one, add its tag to that group.
#
# ``EMBODIMENT_TAG_TO_PROJECTOR_INDEX`` below is derived from this and is the
# public, tag-keyed lookup used everywhere else.
_PROJECTOR_INDEX_GROUPS: dict[int, set[str]] = {
    0: {"simpler_env_google"},
    1: {"simpler_env_widowx"},
    2: {"libero_sim"},
    # Finetune placeholder projector; sim-eval robocasa tags piggyback on
    # `new_embodiment`.
    10: {"new_embodiment", "robocasa_panda_omron", "robocasa_gr1_tabletop"},
    # Same G1+SONIC embodiment with one vs three cameras — state and action
    # spaces are identical, so the two tags deliberately share the projector.
    11: {"unitree_g1_sonic", "unitree_g1_sonic_3views"},
    24: {"oxe_droid_relative_eef_relative_joint"},
    # Same G1 embodiment either side of the pretrain/posttrain boundary
    # (`real_g1_*` is pretrain, `unitree_g1_full_body_*` is posttrain).
    25: {
        "real_g1_relative_eef_relative_joints",
        "unitree_g1_full_body_with_waist_height_nav_cmd",
    },
    # One R1 Pro Sharpa robot, four data-source variants.
    26: {
        "real_r1_pro_sharpa_relative_eef",
        "real_r1_pro_sharpa_relative_eef_human",
        "real_r1_pro_sharpa_relative_eef_maxinsights",
        "real_r1_pro_sharpa_relative_eef_mecka",
    },
    # xdof base + subtask refinement.
    27: {
        "xdof_relative_eef_relative_joint",
        "xdof_relative_eef_relative_joint_subtask",
    },
}


def _build_tag_to_projector_index(groups: dict[int, set[str]]) -> dict[str, int]:
    """Flatten ``{index: {tags}}`` into ``{tag: index}``.

    Guards against a tag accidentally appearing in two groups, which would
    otherwise be silently resolved by insertion order into a single mapping.
    """
    mapping: dict[str, int] = {}
    for index, tags in groups.items():
        for tag in tags:
            if tag in mapping:
                raise ValueError(
                    f"Embodiment tag {tag!r} is assigned to multiple projector "
                    f"indices ({mapping[tag]} and {index}) in "
                    "_PROJECTOR_INDEX_GROUPS; each tag must map to exactly one index."
                )
            mapping[tag] = index
    return mapping


EMBODIMENT_TAG_TO_PROJECTOR_INDEX: dict[str, int] = _build_tag_to_projector_index(
    _PROJECTOR_INDEX_GROUPS
)


def _patch_mistral_regex_offline() -> None:
    """Stop transformers' tokenizer load from phoning home for non-Mistral repos.

    transformers 4.57's ``PreTrainedTokenizerBase._patch_mistral_regex`` calls
    ``huggingface_hub.model_info()`` unconditionally for hub repo ids — even
    under ``HF_HUB_OFFLINE=1``, where the call raises ``OfflineModeIsEnabled``
    and kills the load. Qwen3VL/Cosmos is never Mistral, so short-circuit any
    id that doesn't mention "mistral" and swallow network failures otherwise.
    (Vendored from upstream gr00t/__init__.py's ``_patch_mistral``, which was
    test-gated there; the baked-backbone image runs fully offline, so it is
    applied unconditionally here.)
    """
    try:
        import transformers.tokenization_utils_base as _tub

        _cls = _tub.PreTrainedTokenizerBase
        _orig = _cls._patch_mistral_regex.__func__
        if getattr(_orig, "_kist_patched", False):
            return

        def _safe(cls, tokenizer, pretrained_model_name_or_path, **kwargs):
            name = str(pretrained_model_name_or_path)
            if os.path.isdir(name) or "mistral" not in name.lower():
                return tokenizer
            try:
                return _orig(cls, tokenizer, pretrained_model_name_or_path, **kwargs)
            except Exception:
                return tokenizer

        _safe._kist_patched = True  # type: ignore[attr-defined]
        _cls._patch_mistral_regex = classmethod(_safe)
    except Exception:
        pass


_patch_mistral_regex_offline()


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> Qwen3VLProcessor:
    if Qwen3VLProcessor is None:
        raise ImportError(
            "Qwen3VLProcessor is not available. "
            "Please upgrade transformers: pip install transformers>=4.52.0"
        )
    return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


def validate_action_horizons(modality_configs, max_action_horizon: int) -> None:
    """Fail at processor construction if any configured embodiment's action horizon
    (the number of action ``delta_indices``) exceeds ``max_action_horizon``.

    ``max_action_horizon`` is set from the model's ``action_horizon``; without this
    check a horizon/model mismatch (e.g. a 50-step embodiment on a 40-step model)
    surfaces only deep in the first forward, after the model and dataset are built.
    """
    offenders: dict[str, int] = {}
    for tag, config in modality_configs.items():
        action = config.get("action") if isinstance(config, dict) else None
        delta_indices = getattr(action, "delta_indices", None)
        if delta_indices is None:
            continue
        horizon = len(delta_indices)
        if horizon > max_action_horizon:
            offenders[tag] = horizon
    if offenders:
        required = max(offenders.values())
        details = ", ".join(f"{tag}={horizon}" for tag, horizon in sorted(offenders.items()))
        raise ValueError(
            f"Embodiment action horizon exceeds max_action_horizon ({max_action_horizon}): "
            f"{details}. Increase model config action_horizon to >= {required} (or reduce the "
            "embodiment action delta_indices)."
        )


class Gr00tN1d7DataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict = {},
    ):
        ### We need to use the same processor for padding input ids and concat
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # Handle vlm_content specially - extract text and images
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                vlm_inputs = self.processor(
                    text=text_list,
                    images=image_inputs,
                    return_tensors="pt",
                    padding=True,
                )
                for k, v in vlm_inputs.items():
                    batch[k] = v
            elif key in (
                "pixel_values",
                "image_grid_thw",
                "attention_mask",
                "input_ids",
            ):
                raise Exception("Not implemented")
            else:
                # state, state_mask, action and action_mask - stack to form batch dimension
                batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"Gr00tN1d7DataCollator(model_name={self.model_name}, model_type={self.model_type})"


class Gr00tN1d7Processor(BaseProcessor):
    data_collator_class = Gr00tN1d7DataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        random_rotation_angle: int | None = None,
        color_jitter_params: dict[str, float] | None = None,
        formalize_language: bool = True,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        model_type: str = "qwen",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        max_action_horizon: int = 50,
        apply_sincos_state_encoding: bool = False,
        use_albumentations: bool = False,
        extra_augmentation_config: dict | None = None,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
        # State augmentation
        exclude_state: bool = False,
        state_dropout_prob: float = 0.0,
        # Normalization
        use_mean_std: bool = False,
        letter_box_transform: bool = False,
    ):
        self.modality_configs = parse_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.use_mean_std = use_mean_std
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action
        self.extra_augmentation_config = extra_augmentation_config

        # State augmentation settings
        self.exclude_state = exclude_state
        self.state_dropout_prob = state_dropout_prob

        self.letter_box_transform = letter_box_transform

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        validate_action_horizons(self.modality_configs, self.max_action_horizon)

        # Save image processing settings
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        self.random_rotation_angle = random_rotation_angle
        self.color_jitter_params = color_jitter_params
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.embodiment_id_mapping = embodiment_id_mapping or EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # Merge any missing pre-trained embodiment tags into the custom mapping
        for k, v in EMBODIMENT_TAG_TO_PROJECTOR_INDEX.items():
            if k not in self.embodiment_id_mapping:
                self.embodiment_id_mapping[k] = v
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        # Only the albumentations eval pipeline survived the inference trim
        # (the SONIC checkpoints were trained with use_albumentations=True;
        # the torchvision variant and all training transforms live upstream).
        self.use_albumentations = use_albumentations
        if not use_albumentations:
            raise NotImplementedError(
                "The torchvision image pipeline was trimmed from this extraction "
                "(every SONIC checkpoint sets use_albumentations=True). Restore "
                "image_augmentations.py from Isaac-GR00T@5ac4e6b to run a "
                "checkpoint that needs it."
            )
        self.eval_image_transform = build_eval_image_transform(
            image_target_size,
            image_crop_size,
            shortest_image_edge,
            crop_fraction,
            letter_box_transform=self.letter_box_transform,
        )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.train()

    @property
    def collator(self):
        return self._collator

    def train(self):
        super().train()
        self.state_action_processor.train()

    def eval(self):
        super().eval()
        self.state_action_processor.eval()

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]

        # Create conversation with images and text
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(
        self,
        messages: list[dict[str, Any]],
    ):
        assert len(messages) == 1
        content = messages[0]["content"]
        embodiment_tag = content.embodiment
        action_data = content.actions
        state_data = content.states

        if self.training:
            raise NotImplementedError(
                "The training path (action normalization/padding, state dropout, "
                "train image transforms) was trimmed from this extraction — this "
                "processor only runs in eval mode. Restore from "
                "Isaac-GR00T@5ac4e6b to train."
            )
        assert not action_data, "Inference processes observations only; got actions"

        # Normalize the state (inference passes no actions)
        norm_state_dict, _ = self.state_action_processor.apply(
            state=state_data,
            action=action_data,
            embodiment_tag=embodiment_tag.value,
        )

        # Concatenate states
        state_keys = self.modality_configs[embodiment_tag.value]["state"].modality_keys
        exclude_state = self.exclude_state or getattr(
            self.modality_configs[embodiment_tag.value]["state"], "exclude_state", False
        )
        if exclude_state:
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(state_data[key])) for key in state_keys], dim=-1
            )
        else:
            normalized_states = torch.cat(
                [torch.from_numpy(norm_state_dict[key]) for key in state_keys], dim=-1
            )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    normalized_states.shape[0],
                    self.max_state_dim - normalized_states.shape[1],
                ),
            ],
            dim=-1,
        )

        # Crop and resize images.
        image_transform = self.eval_image_transform
        image_keys = self.modality_configs[embodiment_tag.value]["video"].modality_keys

        if self.formalize_language:
            language = content.text.lower()
            language = re.sub(r"[^\w\s]", "", language)
        else:
            language = content.text

        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            masks=content.masks,
            image_transform=image_transform,
            language=language,
        )

        transformed_inputs = {
            "state": normalized_states.to(torch.get_default_dtype()),
        }
        # Add VLM inputs
        transformed_inputs.update(vlm_inputs)
        transformed_inputs["embodiment_id"] = self.embodiment_id_mapping[embodiment_tag.value]
        return transformed_inputs

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform: A.Compose,
        language: str,
    ):
        # Mask-based transforms were training-only augmentation and were
        # trimmed with them; inference never carries masks.
        assert masks is None, "mask transforms were trimmed from this extraction"

        temporal_stacked_images = {}
        for view in image_keys:
            assert view in images, f"{view} not in {images}"
            transformed_images = apply_images(image_transform, images[view])
            temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
            .flatten(0, 1)
            .numpy()
        )  # (T*V, C, H, W), processor expects numpy array

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )
        hub_keys = (
            "_commit_hash",
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        )
        hub_kwargs = {key: kwargs.pop(key) for key in hub_keys if key in kwargs}
        use_auth_token = kwargs.pop("use_auth_token", None)
        if "token" not in hub_kwargs and use_auth_token is not None:
            hub_kwargs["token"] = use_auth_token
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)
        config_file = pretrained_model_name_or_path / "processor_config.json"
        statistics_file = pretrained_model_name_or_path / "statistics.json"
        embodiment_id_file = pretrained_model_name_or_path / "embodiment_id.json"
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if not is_local:
            config_file = Path(
                cached_file(pretrained_model_name_or_path, "processor_config.json", **hub_kwargs)
            )
            statistics_file = Path(
                cached_file(pretrained_model_name_or_path, "statistics.json", **hub_kwargs)
            )
            embodiment_id_file = Path(
                cached_file(pretrained_model_name_or_path, "embodiment_id.json", **hub_kwargs)
            )

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if embodiment_id_file.exists():
            with open(embodiment_id_file, "r") as f:
                embodiment_id_mapping = json.load(f)
        else:
            embodiment_id_mapping = None
        processor_kwargs = config["processor_kwargs"]
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

        # Backfill fields that older checkpoints may not have serialized.
        # Without these, __init__ defaults silently apply — correct today but
        # fragile if defaults ever change.
        processor_kwargs.setdefault("model_name", "nvidia/Cosmos-Reason2-2B")
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        # Directly override other processor kwargs
        if kwargs:
            # Override modality configs while keeping pretrained embodiment configs
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "random_rotation_angle",
                "color_jitter_params",
                "use_relative_action",
                "exclude_state",
                "state_dropout_prob",
                "use_mean_std",
                "model_name",
                "model_type",
                "max_action_horizon",
                "max_state_dim",
                "max_action_dim",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)


AutoProcessor.register("Gr00tN1d7", Gr00tN1d7Processor)
