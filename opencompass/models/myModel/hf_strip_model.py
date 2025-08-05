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
from opencompass.models.huggingface_above_v4_33 import HuggingFaceBaseModel as HuggingFaceCausalLM

@MODELS.register_module()
class HuggingFaceCausalLM_Strip(HuggingFaceCausalLM):
    # def _load_model(self,
    #                 path: str,
    #                 model_kwargs: dict,
    #                 peft_path: Optional[str] = None):
        
    #     rope_scaling = model_kwargs.pop('rope_scaling', None)
    #     super._load_model(path, model_kwargs, peft_path)


    def generate(self,
                 inputs: List[str],
                 **kwargs) -> List[str]:
        inputs = [x.strip() for x in inputs]
        return super().generate(inputs, **kwargs)