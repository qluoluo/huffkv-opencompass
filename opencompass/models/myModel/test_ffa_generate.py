import random
import numpy as np
import os
import time
import torch
from peft import PeftModel

from transformers import AutoTokenizer, AutoConfig
from ffa.modeling_llama import LlamaForCausalLM

# 模型路径
# model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

# 加载配置
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)

config_attn_settings = dict(
    use_ffa = True,
    BS = 256,
    SBS = 256,
    delta = 5.0,
)

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
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))