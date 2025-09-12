import torch
from flash_attn import flash_attn_func

# 检查是否可用 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置随机种子以便重现结果
torch.manual_seed(42)

# 创建随机测试数据（batch_size=2, seq_len=128, 12个注意力头，每个头维度64）
batch_size = 2
seq_len = 128
n_heads = 12
head_dim = 64

q = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
k = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=torch.float16)
v = torch.randn(batch_size, seq_len, n_heads, head_dim, device=device, dtype=torch.float16)

# 使用 Flash Attention 计算注意力
output = flash_attn_func(q, k, v, causal=True)

# 检查输出形状和数值
print(f"Output shape: {output.shape}")
print(f"Output contains NaN: {torch.isnan(output).any().item()}")
print(f"Output contains Inf: {torch.isinf(output).any().item()}")

# 简单验证输出值的合理性
print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
print(f"Output mean: {output.mean().item():.6f}")

print("Flash Attention 测试完成！")