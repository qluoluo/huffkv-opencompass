import random
import numpy as np
import os
import time
import torch
from peft import PeftModel

from transformers import AutoTokenizer, AutoConfig
from bucket_attn.modeling_llama import LlamaForCausalLM

# 模型路径
model_path = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/models/Llama-3_2-3B'

print(f"{os.path.basename(model_path)=}")

# 加载配置
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
)

config_attn_settings = dict(
    # use_bucket_attn = False,
    use_bucket_attn = True,
    accurate_kv_num = 128,
    accurate_bound = -3.0,
    lost_bound = -10.0,
    bucket_step = 0.1,
)

config.attn_settings = config_attn_settings

# 加载模型
model = LlamaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
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

# 预热运行（避免第一次运行的额外开销）
print("进行预热运行...")
with torch.no_grad():
    warmup_outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    del warmup_outputs
torch.cuda.synchronize()  # 确保所有CUDA操作完成

# 多次运行并计时
num_runs = 5
cuda_times = []

print(f"开始进行 {num_runs} 次运行并计时...")
for i in range(num_runs):
    # 创建CUDA事件
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 确保所有之前的CUDA操作已完成
    torch.cuda.synchronize()
    
    # 记录开始时间
    start_event.record()
    
    # 执行生成
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    
    # 记录结束时间
    end_event.record()
    
    # 等待所有CUDA操作完成
    torch.cuda.synchronize()
    
    # 计算CUDA时间（毫秒）
    cuda_time_ms = start_event.elapsed_time(end_event)
    cuda_times.append(cuda_time_ms)
    
    # 清理内存
    del outputs
    torch.cuda.empty_cache()
    
    print(f"运行 {i+1}/{num_runs}: {cuda_time_ms:.2f} 毫秒")

# 计算统计信息
avg_cuda_time = sum(cuda_times) / len(cuda_times)
min_cuda_time = min(cuda_times)
max_cuda_time = max(cuda_times)

print("\n===== 统计结果 =====")
print(f"运行次数: {num_runs}")
print(f"平均CUDA时间: {avg_cuda_time:.2f} 毫秒")
print(f"最短CUDA时间: {min_cuda_time:.2f} 毫秒")
print(f"最长CUDA时间: {max_cuda_time:.2f} 毫秒")
print(f"时间标准差: {np.std(cuda_times):.2f} 毫秒")

# 可选：同时测量一次CPU时间作为对比
print("\n测量一次CPU时间作为对比...")
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
end_time = time.time()
cpu_time = end_time - start_time
print(f"CPU时间: {cpu_time:.4f} 秒 ({cpu_time*1000:.2f} 毫秒)")

# 清理
del outputs
torch.cuda.empty_cache()