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
from opencompass.models.huggingface import HuggingFaceCausalLM

def extract_last_quoted_sentence(text):
    # 从右侧找到最后一个单引号的位置
    end_index = text.rfind("'")
    if end_index == -1:
        return None  # 没有单引号
    
    # 从最后一个单引号之前找到前一个单引号
    start_index = text.rfind("'", 0, end_index)
    if start_index == -1:
        return None  # 只有一个单引号
    
    # 提取两个单引号之间的内容
    return text[start_index+1:end_index]

def process_for_niah(text: str):
    text = text.strip()
    index = text.rfind("Please answer in the format")
    p = text[index:]
    p = extract_last_quoted_sentence(p)
    text = text[:index] + p.removesuffix("________.").strip()
    return text

@MODELS.register_module()
class HuggingFaceCausalLMForNIAH(HuggingFaceCausalLM):
    # def _load_model(self,
    #                 path: str,
    #                 model_kwargs: dict,
    #                 peft_path: Optional[str] = None):
        
    #     rope_scaling = model_kwargs.pop('rope_scaling', None)
    #     super._load_model(path, model_kwargs, peft_path)


    def generate(self,
                 inputs: List[str],
                 **kwargs) -> List[str]:
        inputs = [process_for_niah(x) for x in inputs]
        return super().generate(inputs, **kwargs)