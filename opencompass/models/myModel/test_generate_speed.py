import random
import numpy as np
import os
import torch
from peft import PeftModel

import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_memlab import LineProfiler

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

# os.environ['CUDA_VISIBLE_DEVICES'] = "7"

# model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/train/models/Qwen2.5-7B'
model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/train/script/25.7.15/qwen7b-ssa-r28-lcr0_98/saves/checkpoint-1200'

print(f"{os.path.basename(model_path)=}")


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    # torch_dtype=torch.float32,
    attn_implementation="flash_attention_2",  # 强制使用flash attention
    # device_map="auto",
    device_map="cuda:0",
    trust_remote_code=True,  # 避免安全检查干扰错误显示
    # config=config,
)
model.eval()

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

inputs = tokenizer(text, return_tensors="pt").to(model.device)
print(f"{inputs.input_ids.shape=}")


with torch.profiler.profile(
    record_shapes=True,  # 记录操作的输入形状
    profile_memory=True,  # 记录内存分配
    activities=[  # 指定分析 CPU 和 GPU（若可用）
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    with_stack=True,
) as prof:
    # outputs = model.generate(**inputs, max_new_tokens=100,
    #     repetition_penalty=1.5,          # 重复惩罚系数 >1.0 表示惩罚
    #     do_sample=True,                  # 启用采样（用于更自然的结果）
    #     top_k=50,                        # Top-k 采样
    #     top_p=0.95,                      # Top-p (nucleus) 采样
    #     temperature=0.7,
    # )
    with torch.no_grad():
        outputs = model(**inputs)

    del outputs

prof.export_chrome_trace("/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/train/simple/trace2.json")  