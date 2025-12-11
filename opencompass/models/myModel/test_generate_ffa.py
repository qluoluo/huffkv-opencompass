import random
import numpy as np
import os
import time
import torch

from transformers import AutoTokenizer, AutoConfig
from ffa.modeling_llama import LlamaForCausalLM
from ffa.quantized_cache import QuantizedCache

# 模型路径
model_path = '/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

# 加载配置
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)

config_attn_settings = {
    "use_ffa_decode": True,
    "delta": 5.0,
    "k_bits": 2,
    "k_quant_dim": 1,
}

config.attn_settings = config_attn_settings

# 加载模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16, 
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
    trust_remote_code=True,
    config=config,
)
model.eval()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 准备输入
text = "Please find the magic number in the below text: " + "sun is red, grass is green." * (6 * 1024) + "\n\nThe Magic Number is " 

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"{inputs.input_ids.shape=}")

with torch.no_grad():
    cache = QuantizedCache(
        key_bits=config.attn_settings.get("k_bits", 2),
        key_quant_dim=config.attn_settings.get("k_quant_dim", 1),
    )
    outputs = model.generate(**inputs, past_key_values=cache, max_new_tokens=30, do_sample=True)
    print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
