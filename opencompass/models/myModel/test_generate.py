import random
import numpy as np
import os
import time
import torch

from transformers import AutoTokenizer, AutoConfig
from transformers import LlamaForCausalLM

# 模型路径
model_path = '/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

# 加载配置
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)

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

prompt_path = os.path.join(os.path.dirname(__file__), "test_text.txt")
with open(prompt_path, "r", encoding="utf-8") as prompt_file:
    text = prompt_file.read()

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"{inputs.input_ids.shape=}")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    # outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
    
    print(tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
