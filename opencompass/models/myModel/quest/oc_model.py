import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

import torch
from transformers import AutoModelForCausalLM

from opencompass.models.myModel.hf_strip_model import (
    HuggingFaceCausalLM_Strip as HuggingFaceCausalLM,
)
from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

DEFAULT_MODEL_KWARGS = dict(device_map="auto", trust_remote_code=True)
DEFAULT_TOKEN_BUDGET = 256
DEFAULT_CHUNK_SIZE = 16


def _str_to_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    """Map common dtype strings to torch dtypes."""
    if name is None:
        return None
    if not isinstance(name, str):
        return name

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _resolve_quest_repo(user_path: Optional[Union[str, Path]]) -> Path:
    """Find the Quest repo path from user input, env or default location."""
    candidates = []
    if user_path:
        candidates.append(Path(user_path).expanduser())

    env_repo = os.getenv("QUEST_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())

    parents = Path(__file__).resolve().parents
    if len(parents) >= 6:
        candidates.append(parents[5] / "ffa-quest" / "quest")

    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.exists():
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Cannot locate Quest repo. Tried: {searched}.")


def _add_repo_to_sys_path(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))


@MODELS.register_module()
class LlamaForCausalLM_Quest_OC(HuggingFaceCausalLM):
    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        model_kwargs = DEFAULT_MODEL_KWARGS.copy()
        model_kwargs.update(kwargs)

        quest_enabled = bool(model_kwargs.pop("quest", True))
        token_budget = model_kwargs.pop("token_budget", DEFAULT_TOKEN_BUDGET)
        chunk_size = model_kwargs.pop("chunk_size", DEFAULT_CHUNK_SIZE)
        quest_repo = model_kwargs.pop("quest_repo", None)
        device = model_kwargs.pop("device", None)

        dtype_hint = model_kwargs.pop("dtype", None)
        torch_dtype = model_kwargs.pop("torch_dtype", None)
        dtype = _str_to_dtype(dtype_hint if dtype_hint is not None else torch_dtype)
        if device is not None and str(device).startswith("cpu") and dtype not in (
            None,
            "auto",
            torch.float32,
        ):
            dtype = torch.float32
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        # Normalize device_map string values like "none" -> None
        if isinstance(model_kwargs.get("device_map"), str) and model_kwargs["device_map"].lower() == "none":
            model_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        if model_kwargs.get("device_map") is None and device is not None:
            self.model = self.model.to(device)

        self.model.eval()
        self.model.generation_config.do_sample = False

        if quest_enabled:
            quest_repo_path = _resolve_quest_repo(quest_repo)
            _add_repo_to_sys_path(quest_repo_path)

            from evaluation import quest_attention
            from evaluation.quest_attention import enable_quest_attention_eval

            if hasattr(self.model.config, "num_hidden_layers"):
                quest_attention.layer_id = self.model.config.num_hidden_layers

            quest_args = SimpleNamespace(token_budget=token_budget, chunk_size=chunk_size)
            enable_quest_attention_eval(self.model, quest_args)
        else:
            # Explicitly keep vanilla attention when quest flag is off.
            pass
