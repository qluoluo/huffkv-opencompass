import random
import numpy as np
import os, json
import torch
from peft import PeftModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huffkv.huffkv_modeling_llama import LlamaForCausalLM

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

config = AutoConfig.from_pretrained(model_path)

config_kvcache_settings = {
    "k_bits": 4,
    "v_bits": 4,
    "k_quant_dim": -2, # [bsz, num_heads, seq_len, head_dim] KIVI: k->head_dim*numheads v: seq_len
    "v_quant_dim": -2,
    "window_size": 128,
    "sparse_num": 128,
    "debug": True,
}
# from types import SimpleNamespace
# config_kvcache_settings = SimpleNamespace(**config_kvcache_settings)
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

# text = "hello " * (4 * 1024)


text_fp = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/script/Llama-3_2-3B-huffkv/eval-result/niah/20250801_152805/predictions/GQ-kc2-vc2/Length32000Depth42_origin_en_32k.json'
with open(text_fp) as f:
    text_dict = json.load(f)

for text_value in text_dict.values():
    text = text_value['origin_prompt']
    
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    print(f"{input_ids.shape=}")

    model.forward(input_ids)

    break