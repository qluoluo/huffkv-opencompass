import torch
import numpy as np
from typing import Tuple, Dict, Any

# 支持的量化位数
SUPPORTED_BITS = [1, 2, 4, 8]

def _pack_tensor(tensor: torch.Tensor, nbits: int) -> torch.Tensor:
    """
    将量化张量打包成uint8格式，沿最后维度打包
    """
    if nbits == 8:
        return tensor.to(torch.uint8)  # 确保是uint8类型
    
    # 获取原始形状
    original_shape = tensor.shape
    
    # 计算每个uint8可以存储多少个量化值
    values_per_byte = 8 // nbits
    
    # 计算最后一个维度需要填充的大小
    last_dim_size = original_shape[-1]
    pad_size = (values_per_byte - (last_dim_size % values_per_byte)) % values_per_byte
    
    # 如果需要填充，在最后一个维度上填充零
    if pad_size > 0:
        pad_shape = list(original_shape)
        pad_shape[-1] = pad_size
        padding = torch.zeros(pad_shape, dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=-1)
    
    # 将最后一个维度重塑为 (..., -1, values_per_byte)
    new_shape = list(tensor.shape[:-1]) + [-1, values_per_byte]
    tensor_reshaped = tensor.view(new_shape).to(torch.uint8)
    
    # 执行位打包
    packed = torch.zeros(tensor_reshaped.shape[:-1], dtype=torch.uint8, device=tensor.device)
    for i in range(values_per_byte):
        packed |= (tensor_reshaped[..., i] << (i * nbits))
    
    return packed

def _unpack_tensor(packed_tensor: torch.Tensor, nbits: int, original_shape: torch.Size) -> torch.Tensor:
    """
    将打包的uint8张量解包回原始量化值，沿最后维度解包
    """
    if nbits == 8:
        return packed_tensor  # 无需解包
    
    # 计算每个uint8存储的量化值数量
    values_per_byte = 8 // nbits
    mask = (2 ** nbits) - 1
    
    # 解包：从每个打包的字节中提取各个位段
    unpacked_list = []
    for i in range(values_per_byte):
        values = (packed_tensor >> (i * nbits)) & mask
        unpacked_list.append(values)
    
    # 将解包的值在最后一个维度上堆叠
    unpacked = torch.stack(unpacked_list, dim=-1)
    
    # 将最后两个维度合并：(..., packed_groups, values_per_byte) -> (..., packed_groups * values_per_byte)
    new_shape = list(unpacked.shape[:-2]) + [-1]
    unpacked = unpacked.view(new_shape)
    
    # 截断到原始大小（移除填充的零）
    original_last_dim = original_shape[-1]
    if unpacked.shape[-1] > original_last_dim:
        # 创建切片，只保留原始最后维度的大小
        slices = [slice(None)] * (len(original_shape) - 1) + [slice(0, original_last_dim)]
        unpacked = unpacked[slices]
    
    # 确保最终形状正确
    unpacked = unpacked.view(original_shape)
    
    return unpacked

def quantize(tensor: torch.Tensor, nbits: int, dim: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    对输入张量进行非对称量化
    
    Args:
        tensor: 输入张量
        nbits: 量化位数 (1, 2, 4, 8)
        dim: 量化维度
        
    Returns:
        quantized_packed: 打包后的uint8张量
        metadata: 包含反量化所需信息的字典（新增压缩率）
    """
    if nbits not in SUPPORTED_BITS:
        raise ValueError(f"不支持的量化位数: {nbits}, 支持的位数: {SUPPORTED_BITS}")
    
    if dim < 0:
        dim = tensor.ndim + dim
    if dim >= tensor.ndim or dim < 0:
        raise ValueError(f"维度 {dim} 超出张量维度范围 [0, {tensor.ndim-1}]")
    
    # 保存原始类型和设备
    orig_dtype = tensor.dtype
    orig_device = tensor.device
    
    # 转换为float32进行量化计算
    tensor = tensor.float()
    
    # 计算量化范围
    qmin = 0
    qmax = (2 ** nbits) - 1
    
    # 沿指定维度计算最小值和最大值
    min_vals = tensor.min(dim=dim, keepdim=True)[0]
    max_vals = tensor.max(dim=dim, keepdim=True)[0]
    
    # 处理相同最小值和最大值的情况
    eps = torch.finfo(tensor.dtype).eps
    max_vals = torch.maximum(max_vals, min_vals + eps)
    
    # 计算scale和zero_point
    scale = (max_vals - min_vals) / (qmax - qmin)
    zero_point = min_vals  # 浮点数最小值对应量化值0
    
    # 执行量化
    quantized = torch.round((tensor - zero_point) / scale).clamp(qmin, qmax)
    quantized = quantized.to(torch.uint8)
    
    # 打包量化结果
    quantized_packed = _pack_tensor(quantized, nbits)

    tensor = tensor.to(orig_device).to(orig_dtype)
    
    # 构建metadata
    metadata = {
        'nbits': nbits,
        'dim': dim,
        'scale': scale,
        'zero_point': zero_point,
        'original_shape': tensor.shape,
        'original_dtype': orig_dtype,  # 保存原始类型
        'original_device': orig_device,  # 保存原始设备
        'packed_shape': quantized_packed.shape,
    }

    metadata['compression_ratio'] = get_compression_ratio(tensor, quantized_packed)
    
    return quantized_packed, metadata

def dequantize(quantized_packed: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    """
    反量化函数
    
    Args:
        quantized_packed: 打包的量化张量
        metadata: 量化时生成的元数据
        
    Returns:
        dequantized: 反量化后的张量（恢复到原始类型和设备）
    """
    # 解包量化张量
    quantized = _unpack_tensor(
        quantized_packed, 
        metadata['nbits'], 
        metadata['original_shape']
    )
    
    # 执行反量化
    scale = metadata['scale']
    zero_point = metadata['zero_point']
    
    dequantized = quantized.float() * scale + zero_point
    
    # 恢复到原始类型和设备
    dequantized = dequantized.to(
        dtype=metadata['original_dtype'],
        device=metadata.get('original_device', 'cpu')
    )
    
    return dequantized

def get_compression_ratio(original_tensor: torch.Tensor, quantized_tensor: torch.Tensor) -> float:
    """
    计算压缩比（基于两个张量的存储大小）
    
    Args:
        original_tensor: 原始张量
        quantized_tensor: 量化后的张量（打包后的）
        
    Returns:
        压缩比 = 原始张量总字节数 / 量化张量总字节数
    """
    # 计算原始数据大小 (字节)
    original_size = original_tensor.element_size() * original_tensor.numel()
    
    # 计算量化后大小
    quantized_size = quantized_tensor.element_size() * quantized_tensor.numel()
    
    return original_size / quantized_size

# 使用示例和测试
def test_quantizer():
    """测试量化器的功能"""
    
    # 创建测试张量 (使用不同浮点类型)
    torch.manual_seed(42)
    test_tensor = torch.rand(40, 40, dtype=torch.bfloat16)
    
    print(f"原始张量: {test_tensor.shape=}, dtype={test_tensor.dtype}")
    print(f"原始值:\n{test_tensor}")

    # 测试4位量化
    quantized_packed, metadata = quantize(test_tensor, nbits=4, dim=-2)

    print(f"\n打包后: {quantized_packed.shape=}, dtype={quantized_packed.dtype}")
    print(f"元数据: nbits={metadata['nbits']}, dim={metadata['dim']}, compression_ratio={metadata['compression_ratio']:.2f}x")

    # 反量化
    dequant_tensor = dequantize(quantized_packed, metadata)

    print(f"\n反量化后: {dequant_tensor.shape=}, dtype={dequant_tensor.dtype}")
    print(f"反量化值:\n{dequant_tensor}")
    
    # 计算误差
    error = torch.abs(test_tensor - dequant_tensor)
    print(f"\n最大绝对误差: {error.max().item():.6f}")
    print(f"平均绝对误差: {error.mean().item():.6f}")

if __name__ == "__main__":
    test_quantizer()