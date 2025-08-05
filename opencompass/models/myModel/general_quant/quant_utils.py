import torch

def quantize_tensor(x, nbit, dim):
    # 沿着指定维度找到最大值和最小值
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    
    # 计算scale和zero_point
    scale = (x_max - x_min) / (2**nbit - 1)
    
    zero_point = x_min
    
    # 量化
    x_quantized = ((x - zero_point) / scale).round().clamp(0, 2**nbit - 1).to(dtype=torch.int64)
    
    return x_quantized, scale, zero_point

def dequantize_tensor(x_quantized, scale, zero_point):
    # 还原
    x_dequantized = x_quantized * scale + zero_point
    return x_dequantized

def quantize(x, nbit, dim, global_residual_length=0, local_residual_length=128, group_size=-1):
    seq_len = x.shape[-2]
    mid_len = seq_len - local_residual_length - global_residual_length
    
    # 如果分组，调整 local_residual_length 以处理无法整除的情况
    if group_size > 0:
        local_residual_length += (mid_len % group_size)
    
    # 提取全局残差和局部残差
    x_global = None
    x_local = None
    if global_residual_length > 0:
        x_global = x[..., :global_residual_length, :]
        x = x[..., global_residual_length:, :]
    if local_residual_length > 0:
        x_local = x[..., -local_residual_length:, :]
        x = x[..., :-local_residual_length, :]
    
    # 分组逻辑
    if group_size > 0:
        num_groups = x.shape[-2] // group_size
        assert num_groups * group_size == x.shape[-2], f"x.shape[-2] ({x.shape[-2]}) cannot be divided by group ({group_size})"
        x = x.view(*x.shape[:-2], num_groups, group_size, x.shape[-1])
    
    # 对每个组进行差分操作
    x_quantized, scale, zero_point = quantize_tensor(x, nbit=nbit, dim=dim)
    
    # 返回 global、local、每组第一个 feature 和量化后的差分
    return x_global, x_local, x_quantized, scale, zero_point

def dequantize(x_global, x_local, x_quantized, scale, zero_point, group_size=-1):

    x_restored = dequantize_tensor(x_quantized, scale, zero_point)

    if group_size > 0:
        x_restored = x_restored.view(*x_restored.shape[:-3], -1, x_restored.shape[-1])

    # 合并全局和局部残差
    if x_global is not None:
        x_restored = torch.cat((x_global, x_restored), dim=-2)
    if x_local is not None:
        x_restored = torch.cat((x_restored, x_local), dim=-2)
    
    return x_restored

# 示例使用
if __name__ == "__main__":
    # 创建一个随机张量
    # tensor = torch.randn((2, 3))
    tensor = torch.tensor([[1.0, 2.2, 3.0], [4.0, 5.6, 6.2]])
    nbit = 4
    dim = -1  # 假设我们沿着第二个维度进行量化

    # 量化
    quantized_tensor, scale, zero_point = quantize_tensor(tensor, nbit=nbit, dim=dim)

    print(f"{tensor.shape=}, {quantized_tensor.shape=}, {scale.shape=}, {zero_point.shape=}")

    print("Quantized Tensor: ", quantized_tensor)

    # 还原
    dequantized_tensor = dequantize_tensor(quantized_tensor, scale, zero_point)

    print("Original Tensor: ", tensor)
    print("Dequantized Tensor: ", dequantized_tensor)