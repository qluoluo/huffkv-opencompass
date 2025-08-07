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

    print(f"Quant func {nbit=} {dim=} \n{x_quantized.shape=}, {x_quantized.dtype=}, {x_quantized.flatten()[:10]=}\n \
        {scale.shape=}, {scale.dtype=}\n \
        {zero_point.shape=}, {zero_point.dtype=}")
    
    return x_quantized, (scale, zero_point)

def dequantize_tensor(x_quantized, metadata):
    # 还原
    scale, zero_point = metadata

    print(f"DeQuant func \n{x_quantized.shape=}, {x_quantized.dtype=}, {x_quantized.flatten()[:10]=}\n \
        {scale.shape=}, {scale.dtype=}\n \
        {zero_point.shape=}, {zero_point.dtype=}")

    x_dequantized = x_quantized * scale + zero_point
    return x_dequantized


# 示例使用
if __name__ == "__main__":
    # 创建一个随机张量
    # tensor = torch.randn((2, 3))
    tensor = torch.tensor([[1.0, 2.5, 3.0], [4.0, 5.6, 6.3]])
    nbit = 4
    dim = -1  # 假设我们沿着第二个维度进行量化

    # 量化
    quantized_tensor, (scale, zero_point) = quantize_tensor(tensor, nbit=nbit, dim=dim)

    print(f"{tensor.shape=}, {quantized_tensor.shape=}, {scale.shape=}, {zero_point.shape=}")

    print("Quantized Tensor: ", quantized_tensor)

    # 还原
    dequantized_tensor = dequantize_tensor(quantized_tensor, (scale, zero_point))

    print("Original Tensor: ", tensor)
    print("Dequantized Tensor: ", dequantized_tensor)