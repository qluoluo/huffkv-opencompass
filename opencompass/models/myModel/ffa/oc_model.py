# from .cache_utils import *
# import opencompass.models.myModel.general_quant.cache_utils

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from opencompass.models.base import BaseModel
from opencompass.models.base_api import APITemplateParser
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

# from .huggingface import HuggingFaceCausalLM
# from opencompass.models.huggingface import HuggingFaceCausalLM
from opencompass.models.myModel.hf_strip_model import (
    HuggingFaceCausalLM_Strip as HuggingFaceCausalLM,
)
from .modeling_llama import LlamaForCausalLM
from .modeling_qwen3 import Qwen3ForCausalLM
from .quantized_cache import QuantizedCache
from transformers import AutoConfig


@MODELS.register_module()
class HF_ForCausalLM_FFA_OC(HuggingFaceCausalLM):
    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):

        model_kwargs = kwargs
        attn_defaults = dict(
            use_ffa_prefill=False,
            use_ffa_decode=False,
            delta=5.0,
            pattern_layers=None,
            k_bits=None,
            k_quant_dim=1,
            BS=None,
            SBS=None,
            return_skip_ratio=False,
            use_fp_k=False,
            ffa_decode_kernel="q2",
            use_fp8_residual=True,
            fp8_residual_dtype=None,
        )
        # 允许外部传入 use_ffa 作为简写，默认只开启 decode 路径（prefill 尚未实现）
        use_ffa_flag = model_kwargs.pop("use_ffa", None)

        # 使用字典推导式提取值并设置默认值
        config_attn_settings = {key: model_kwargs.pop(key, default) for key, default in attn_defaults.items()}
        if use_ffa_flag is not None:
            config_attn_settings["use_ffa_decode"] = use_ffa_flag
        # 清理掉为 None 的键，避免把不需要的参数传到内核
        config_attn_settings = {k: v for k, v in config_attn_settings.items() if v is not None}

        # 设置模型参数的数据类型
        # self._set_model_kwargs_torch_dtype(model_kwargs)

        # 从预训练路径加载配置
        trust_remote_code = model_kwargs.get("trust_remote_code", False)

        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        model_type = getattr(config, "model_type", None)
        if model_type not in ("llama", "qwen3"):
            raise ValueError(
                "LlamaForCausalLM_FFA_OC only supports llama/qwen3 models, "
                f"got model_type={model_type} for path={path}"
            )
        config.attn_settings = config_attn_settings
        self.config_attn_settings = config_attn_settings

        if model_type == "llama":
            self.model = LlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=path, config=config, **model_kwargs
            )
        else:
            self.model = Qwen3ForCausalLM.from_pretrained(path, config=config, **model_kwargs)

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        decode_kernel = self.config_attn_settings.get("ffa_decode_kernel", "q2")
        use_fp8_residual = self.config_attn_settings.get("use_fp8_residual", True)
        store_fp8_residual = (
            isinstance(decode_kernel, str)
            and decode_kernel.strip().lower() in ("q2fp8", "q2_fp8", "fp8")
            and use_fp8_residual
        )

        self.generation_kwargs['past_key_values'] = QuantizedCache(
            key_bits=self.config_attn_settings.get("k_bits", 2),
            key_quant_dim=self.config_attn_settings.get("k_quant_dim", 1),
            store_fp8_residual=store_fp8_residual,
            fp8_residual_dtype=self.config_attn_settings.get("fp8_residual_dtype"),
        )
        
        return super().generate(inputs, **kwargs)
