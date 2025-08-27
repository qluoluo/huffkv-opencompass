import random
import numpy as np
import os
import torch
from peft import PeftModel

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from taylor_kv.modeling_llama import LlamaForCausalLM

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

# model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'
model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

config = AutoConfig.from_pretrained(model_path)

config_kvcache_settings = {
    "window_size": 8,
    "sparse_num": 512,

    "use_remain": True,
    # "use_remain": False,
    "remain_cluster_k": 64,
    "remain_group_size": -1,
    "remain_order": 1,
    "remain_u_mode": "diag",
    "remain_save_full_prefill_cache": True,
    # "remain_save_full_prefill_cache": False,
    "remain_kmeans_args": {
        "iters": 50,
        "init_method": "k-means++",
        "random_state": 0,
    },
    
    # "debug": True,
    "debug": False,
}
config.kvcache_settings = config_kvcache_settings

model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
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

# model(1)

# 测试生成
# text = "User: Please write a story about a robot and a fish.\nAssistant:"
# text = "User: Please introduce yourself.\nAssistant:"
text = "hello " * (4 * 1024)
text_fp = os.path.join(os.path.dirname(__file__), "needle_bench_0.txt")
with open(text_fp, "r", encoding='utf-8') as f:
    text = f.read().strip().strip('\\n')

# import ipdb; ipdb.set_trace()

print(f"{text[-100:]=}")

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"{inputs.input_ids.shape=}")
outputs = model.generate(**inputs, max_new_tokens=100,
    # repetition_penalty=1.5,          # 重复惩罚系数 >1.0 表示惩罚
    do_sample=False,
    # top_k=50,                        # Top-k 采样
    # top_p=0.95,                      # Top-p (nucleus) 采样
    # temperature=0.7,
)
print("### Generate Content:")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:]))