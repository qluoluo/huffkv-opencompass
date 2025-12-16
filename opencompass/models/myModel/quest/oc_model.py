import sys
from pathlib import Path
from typing import List, Optional

import torch

from opencompass.models.myModel.hf_strip_model import (
    HuggingFaceCausalLM_Strip as HuggingFaceCausalLM,
)
from opencompass.registry import MODELS

# 先尝试已安装的 quest 包，失败时回退到本仓库的 ffa-quest/quest 目录
try:
    from quest import LlamaForCausalLM
except ImportError:
    quest_repo = Path(__file__).resolve().parents[5] / "ffa-quest" / "quest"
    sys.path.append(str(quest_repo))
    from quest import LlamaForCausalLM


@MODELS.register_module()
class LlamaForCausalLM_Quest_OC(HuggingFaceCausalLM):
    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):
        model_kwargs = kwargs.copy()

        # Quest 额外参数（留默认即可）
        quest_page_size = model_kwargs.pop("quest_page_size", 16)
        quest_token_budget = model_kwargs.pop("quest_token_budget", 512)
        quest_max_seq_len = model_kwargs.pop("quest_max_seq_len", getattr(self, "max_seq_len", None))
        quest_device = model_kwargs.pop("quest_device", None)
        quest_dtype = model_kwargs.pop("quest_dtype", model_kwargs.get("torch_dtype", None))

        self.model = LlamaForCausalLM.from_pretrained(path, **model_kwargs)

        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

        # Quest 初始化（逻辑同 ffa-quest 示例）
        if quest_max_seq_len is None:
            quest_max_seq_len = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(quest_dtype, str) and hasattr(torch, quest_dtype):
            quest_dtype = getattr(torch, quest_dtype)
        if quest_dtype is None:
            quest_dtype = next(self.model.parameters()).dtype
        if quest_device is None:
            quest_device = next(self.model.parameters()).device
        else:
            quest_device = torch.device(quest_device)

        self.model.quest_init(
            page_size=quest_page_size,
            max_seq_len=quest_max_seq_len,
            token_budget=quest_token_budget,
            dtype=quest_dtype,
            device=quest_device,
        )

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        # 每次生成前后清理 Quest 状态，避免跨请求残留
        if hasattr(self.model, "quest_clear"):
            self.model.quest_clear()
        outputs = super().generate(inputs, **kwargs)
        if hasattr(self.model, "quest_clear"):
            self.model.quest_clear()
        return outputs
