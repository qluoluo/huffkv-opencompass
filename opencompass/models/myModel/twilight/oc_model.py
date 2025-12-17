import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from transformers import AutoModelForCausalLM

from opencompass.models.myModel.hf_strip_model import (
    HuggingFaceCausalLM_Strip as HuggingFaceCausalLM,
)
from opencompass.registry import MODELS

# Try the installed twilight package first, then fall back to the local repo.
try:
    from twilight.pyimpl import enable_sparse_attention, reset_sparse_config
except ImportError:
    twilight_repo = Path(__file__).resolve().parents[5] / "ffa-twi" / "Twilight"
    sys.path.append(str(twilight_repo))
    try:
        from twilight.pyimpl import enable_sparse_attention, reset_sparse_config
    except ImportError as exc:  # pragma: no cover - import-time guard
        raise ImportError(
            "Unable to import twilight.pyimpl. Please install Twilight or keep "
            "ffa-twi/Twilight alongside this repository."
        ) from exc


DEFAULT_SPARSE_CONFIG = {
    "compressor": {"type": "none"},
    "selector": {"type": "quest", "token_budget": 8192, "chunk_size": 16},
    "weight_estimator": {"type": "min_max_quant", "quant_bit": 4},
    "weight_pruner": {"type": "top_p", "threshold": 0.85},
    "skip_first_two_layers": True,
    "use_estimated_weights_in_attn": False,
}


@MODELS.register_module()
class LlamaForCausalLM_Twilight_OC(HuggingFaceCausalLM):
    """HuggingFace wrapper that enables Twilight sparse attention."""

    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        model_kwargs = kwargs.copy()

        # Optional Twilight overrides
        sparse_config_path = model_kwargs.pop("twilight_config_path", None)
        sparse_config_dict = model_kwargs.pop("twilight_sparse_config", None)
        token_budget = model_kwargs.pop("twilight_token_budget", None)
        chunk_size = model_kwargs.pop("twilight_chunk_size", None)
        top_p = model_kwargs.pop("twilight_top_p", None)
        quant_bit = model_kwargs.pop("twilight_quant_bit", None)
        selector_type = model_kwargs.pop("twilight_selector_type", None)
        weight_estimator = model_kwargs.pop("twilight_weight_estimator", None)
        weight_pruner = model_kwargs.pop("twilight_weight_pruner", None)
        skip_first_two_layers = model_kwargs.pop(
            "twilight_skip_first_two_layers", None
        )
        use_estimated_weights = model_kwargs.pop(
            "twilight_use_estimated_weights_in_attn", None
        )
        compressor_type = model_kwargs.pop("twilight_compressor_type", None)
        enable_budget_info = bool(
            model_kwargs.pop("twilight_enable_budget_info", False)
        )
        enable_score_info = bool(
            model_kwargs.pop("twilight_enable_score_info", False)
        )
        # Drop any other twilight_* hints to avoid breaking HF loading.
        for extra_key in list(model_kwargs.keys()):
            if extra_key.startswith("twilight_"):
                model_kwargs.pop(extra_key)

        sparse_config = self._build_sparse_config(
            sparse_config_path=sparse_config_path,
            sparse_config_dict=sparse_config_dict,
            token_budget=token_budget,
            chunk_size=chunk_size,
            top_p=top_p,
            quant_bit=quant_bit,
            selector_type=selector_type,
            weight_estimator=weight_estimator,
            weight_pruner=weight_pruner,
            skip_first_two_layers=skip_first_two_layers,
            use_estimated_weights=use_estimated_weights,
            compressor_type=compressor_type,
        )

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

        # Apply Twilight monkey patches after the model is fully constructed.
        reset_sparse_config()
        self.budget_info, self.score_info = enable_sparse_attention(
            self.model,
            sparse_config=sparse_config,
            enable_budget_info=enable_budget_info,
            enable_score_info=enable_score_info,
        )
        self.sparse_config = sparse_config

    @staticmethod
    def _build_sparse_config(
        sparse_config_path: Optional[str],
        sparse_config_dict: Optional[dict],
        token_budget: Optional[int],
        chunk_size: Optional[int],
        top_p: Optional[float],
        quant_bit: Optional[int],
        selector_type: Optional[str],
        weight_estimator: Optional[str],
        weight_pruner: Optional[str],
        skip_first_two_layers: Optional[bool],
        use_estimated_weights: Optional[bool],
        compressor_type: Optional[str],
    ) -> dict:
        """Merge Twilight config from defaults, file, and explicit overrides."""

        if sparse_config_dict is not None:
            sparse_config = deepcopy(sparse_config_dict)
        elif sparse_config_path is not None:
            with open(Path(sparse_config_path).expanduser(), "r") as f:
                sparse_config = json.load(f)
        else:
            sparse_config = deepcopy(DEFAULT_SPARSE_CONFIG)

        # Ensure required sections exist before overrides.
        sparse_config.setdefault("compressor", {})
        sparse_config.setdefault("selector", {})
        sparse_config.setdefault("weight_estimator", {})
        sparse_config.setdefault("weight_pruner", {})

        if compressor_type is not None:
            sparse_config["compressor"]["type"] = compressor_type
        if selector_type is not None:
            sparse_config["selector"]["type"] = selector_type
        if token_budget is not None:
            sparse_config["selector"]["token_budget"] = token_budget
        if chunk_size is not None:
            sparse_config["selector"]["chunk_size"] = chunk_size
        if weight_estimator is not None:
            sparse_config["weight_estimator"]["type"] = weight_estimator
        if quant_bit is not None:
            sparse_config["weight_estimator"]["quant_bit"] = quant_bit
        if weight_pruner is not None:
            sparse_config["weight_pruner"]["type"] = weight_pruner
        if top_p is not None:
            sparse_config["weight_pruner"]["threshold"] = top_p
        if skip_first_two_layers is not None:
            sparse_config["skip_first_two_layers"] = skip_first_two_layers
        if use_estimated_weights is not None:
            sparse_config["use_estimated_weights_in_attn"] = use_estimated_weights

        return sparse_config

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        # Twilight relies on per-request state; no extra handling required here.
        return super().generate(inputs, **kwargs)
