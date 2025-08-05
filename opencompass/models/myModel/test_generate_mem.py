import random
import numpy as np
import os
import torch
from peft import PeftModel

import torch
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
# from pytorch_memlab import LineProfiler

from transformers import AutoTokenizer, AutoConfig

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['TOKENIZERS_PARALLELISM'] = "false"
model_path = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)


from transformers import LlamaForCausalLM
# from huffkv.huffkv_modeling_llama import LlamaForCausalLM 

config.kvcache_settings = {
    "k_bits": 2,
    "v_bits": 2,
    "k_quant_dim": -1, # [bsz, num_heads, seq_len, head_dim] KIVI: k->head_dim*numheads v: seq_len
    "v_quant_dim": -1,
    "window_size": 128,
    "sparse_num": 128,
    "pool_kernel_size": -1,
}


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
model.eval()

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

num_repeats = 1

# model(1)

# 测试生成
# text = "User: Please write a story about a robot and a fish.\nAssistant:"
# text = "User: Please introduce yourself.\nAssistant:"
text = "hello " * (64 * 1024)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
device = inputs.input_ids.device
print(f"{inputs.input_ids.shape=}")

torch.cuda.memory._record_memory_history(max_entries=100000)

for i in range(num_repeats):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prefill_mem = torch.cuda.max_memory_allocated(device)
    print(f"num {i} Max mem = {prefill_mem / (1024 ** 3):.2f} GB")
    del outputs


# torch.cuda.memory._dump_snapshot("/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/train/simple/mem.pickle")
# torch.cuda.memory._record_memory_history(enabled=None)