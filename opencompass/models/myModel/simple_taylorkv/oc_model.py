# from .cache_utils import *
import opencompass.models.myModel.general_quant.cache_utils

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
from transformers import LlamaConfig


@MODELS.register_module()
class LlamaForCausalLM_Simple_TaylorKV_OC(HuggingFaceCausalLM):
    def _load_model(
        self,
        path: str,
        kwargs: dict,
        peft_path: Optional[str] = None,
        peft_kwargs: dict = dict(),
    ):

        model_kwargs = kwargs
        config_kvcache_settings = {
            # "k_bits": None,
            # "v_bits": None,
            # "k_quant_dim": None,  # [bsz, num_heads, seq_len, head_dim] KIVI: k->head_dim*numheads v: seq_len
            # "v_quant_dim": None,
            "window_size": 128,
            "sparse_num": 128,
            "pool_kernel_size": -1,
            "debug": True,
            # "debug": False,
        }

        # 使用字典推导式提取值并设置默认值
        config_kvcache_settings = {
            key: model_kwargs.pop(key, default)
            for key, default in config_kvcache_settings.items()
        }

        # 设置模型参数的数据类型
        # self._set_model_kwargs_torch_dtype(model_kwargs)

        # 从预训练路径加载配置
        config = LlamaConfig.from_pretrained(path)

        config.kvcache_settings = config_kvcache_settings

        self.model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=path, config=config, **model_kwargs
        )

        # self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        if peft_path is not None:
            from peft import PeftModel

            peft_kwargs["is_trainable"] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False
