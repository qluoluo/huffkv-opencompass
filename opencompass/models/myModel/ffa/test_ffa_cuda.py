import torch
# from ffa_cuda.flash_attn_interface import flash_attn_func
from hopper.flash_attn_interface import flash_attn_func

# 确保使用支持printf的设备 (Compute Capability >= 7.0)
device = "cuda"
dtype = torch.float16

# 配置参数
batch_size = 1
seqlen_q = 1  # Flash Decoding 的关键特征: q的序列长度为1
seqlen_k = 1024 # KV cache 中有较长的序列
num_heads = 16
head_dim = 128

# 创建输入张量
q = torch.randn(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
# 假设这是已经存在的KV Cache
kv_cache = torch.randn(batch_size, seqlen_k, 2, num_heads, head_dim, device=device, dtype=dtype)
k_cache, v_cache = kv_cache.unbind(2)

print("Calling custom FFA kernel...")

# 使用kv_cache参数会触发split-k (Flash Decoding) 逻辑
# 注意，根据flash_attn版本，你可能需要传入kv_cache而不是k和v
# 我们这里直接传入 k, v
# flash_attn_func 内部会自动处理 cache
output = flash_attn_func(q, k_cache, v_cache, causal=True)

print("FFA kernel execution finished.")
print("Output shape:", output.shape)
