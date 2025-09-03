import random
import numpy as np
import os
import torch
from peft import PeftModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from bucket_attn.modeling_llama import LlamaForCausalLM


# model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

config = AutoConfig.from_pretrained(model_path)

config_kvcache_settings = dict(
    use_bucket = False,
    accurate_kv_num = 128,
    accurate_bound = -3.0,
    lost_bound = -10.0,
    bucket_step = 0.1,
)

config.kvcache_settings = config_kvcache_settings

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    # torch_dtype=torch.float32,
    attn_implementation="flash_attention_2",  # 强制使用flash attention
    # device_map="auto",
    device_map="cuda:0",
    trust_remote_code=True,  # 避免安全检查干扰错误显示
    config=config,
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)


# 测试生成
# text = "User: Please write a story about a robot and a fish.\nAssistant:"
# text = "User: Please introduce yourself.\nAssistant:"
text = "hello " * (4 * 1024)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"{inputs.input_ids.shape=}")
outputs = model.generate(**inputs, max_new_tokens=100,
    repetition_penalty=1.5,          # 重复惩罚系数 >1.0 表示惩罚
    do_sample=True,                # 启用采样（用于更自然的结果）
    top_k=50,                        # Top-k 采样
    top_p=0.95,                      # Top-p (nucleus) 采样
    temperature=0.7,
)
print("### Generate Content:")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))