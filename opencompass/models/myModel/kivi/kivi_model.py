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
from opencompass.models.myModel.hf_strip_model import HuggingFaceCausalLM_Strip as HuggingFaceCausalLM
from .llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM

@MODELS.register_module()
class LlamaForCausalLM_KIVI_OC(HuggingFaceCausalLM):
    def _load_model(self,
                    path: str,
                    kwargs: dict,
                    peft_path: Optional[str] = None,
                    peft_kwargs: dict = dict()):
        # from transformers import AutoModelForCausalLM
        model_kwargs = kwargs
        k_bits, v_bits, group_size, residual_length = model_kwargs.pop("k_bits"), model_kwargs.pop("v_bits"), model_kwargs.pop("group_size"), model_kwargs.pop("residual_length")
        use_flash = model_kwargs.pop("use_flash", True)
        # rope_scaling = model_kwargs.pop('rope_scaling', None)

        # if model_kwargs.get("torch_dtype", "torch.bfloat16") == "torch.bfloat16":
        #     model_kwargs["torch_dtype"] = torch.bfloat16
        # elif model_kwargs.get("torch_dtype") == "torch.float16":
        #     model_kwargs["torch_dtype"] = torch.float16
        # else:
        #     raise ValueError("Unknown torch_dtype")

        # self._set_model_kwargs_torch_dtype(model_kwargs)

        config = LlamaConfig.from_pretrained(path)
        config.k_bits = k_bits # current support 2/4 bit for KV Cache
        config.v_bits = v_bits # current support 2/4 bit for KV Cache
        config.group_size = group_size
        config.residual_length = residual_length # the number of recent fp16 tokens
        config.use_flash = use_flash
        # config.rope_scaling = rope_scaling
        CACHE_DIR = './cache'

        print(f"{model_kwargs=}")

        self.model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=path,
            config=config,
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
            **model_kwargs
        )

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)
        
        self.model.eval()
        self.model.generation_config.do_sample = False