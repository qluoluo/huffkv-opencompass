import torch
import heapq
from collections import Counter
from typing import Dict, Tuple


class HuffmanNode:
    def __init__(self, char: int, freq: int, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman_tree(freq_counter: Counter) -> HuffmanNode:
    """构建Huffman树"""
    if len(freq_counter) == 1:
        char = next(iter(freq_counter.keys()))
        return HuffmanNode(char, freq_counter[char])
    
    heap = []
    for char, freq in freq_counter.items():
        heapq.heappush(heap, HuffmanNode(char, freq))
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(-1, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    
    return heap[0]


def _generate_codes(root: HuffmanNode, code: str = "", codes: Dict[int, str] = None) -> Dict[int, str]:
    """生成Huffman编码表"""
    if codes is None:
        codes = {}
    
    if root.char != -1:  # 叶子节点
        codes[root.char] = code if code else "0"
    else:
        _generate_codes(root.left, code + "0", codes)
        _generate_codes(root.right, code + "1", codes)
    
    return codes


def _bits_to_bytes(bit_string: str) -> Tuple[torch.Tensor, int]:
    """将bit字符串转换为uint8张量"""
    padding = (8 - len(bit_string)) % 8
    if padding:
        bit_string += '0' * padding
    
    bytes_list = []
    for i in range(0, len(bit_string), 8):
        bytes_list.append(int(bit_string[i:i+8], 2))
    
    return torch.tensor(bytes_list, dtype=torch.uint8), padding


def _bytes_to_bits(bytes_tensor: torch.Tensor, padding: int) -> str:
    """将uint8张量转换为bit字符串"""
    bit_string = "".join(format(byte.item(), '08b') for byte in bytes_tensor)
    return bit_string[:-padding] if padding > 0 else bit_string


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


def huffman_compress(data: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """压缩多维uint8张量"""
    if data.dtype != torch.uint8:
        raise ValueError("输入数据必须是torch.uint8类型")
    
    original_shape = data.shape
    flat_data = data.flatten().tolist()
    
    freq_counter = Counter(flat_data)
    huffman_tree = _build_huffman_tree(freq_counter)
    huffman_codes = _generate_codes(huffman_tree)
    
    encoded_bits = "".join(huffman_codes[val] for val in flat_data)
    compressed_bytes, padding = _bits_to_bytes(encoded_bits)
    
    # 计算压缩率
    # original_size = data.numel()
    # compressed_size = compressed_bytes.numel()
    # compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
    compression_ratio = get_compression_ratio(data, compressed_bytes)
    
    metadata = {
        'original_shape': original_shape,
        'huffman_codes': huffman_codes,
        'padding': padding,
        'compression_ratio': compression_ratio
    }
    
    return compressed_bytes, metadata


def huffman_decompress(compressed_data: torch.Tensor, metadata: Dict) -> torch.Tensor:
    """解压缩数据"""
    original_shape = metadata['original_shape']
    huffman_codes = metadata['huffman_codes']
    padding = metadata['padding']
    
    decode_dict = {code: char for char, code in huffman_codes.items()}
    bit_string = _bytes_to_bits(compressed_data, padding)
    
    decoded_values = []
    current_code = ""
    for bit in bit_string:
        current_code += bit
        if current_code in decode_dict:
            decoded_values.append(decode_dict[current_code])
            current_code = ""
    
    return torch.tensor(decoded_values, dtype=torch.uint8).reshape(original_shape)


# 示例使用
if __name__ == "__main__":
    # 创建测试数据
    print("创建测试数据...")
    original_data = torch.randint(0, 64, (2, 16, 1024, 32), dtype=torch.uint8)
    print(f"原始数据形状: {original_data.shape}")
    print(f"原始数据大小: {original_data.numel()} 元素")
    
    # 压缩
    print("\n压缩中...")
    compressed_data, metadata = huffman_compress(original_data)
    print(f"压缩后大小: {compressed_data.numel()} 字节")
    
    # 显示压缩率
    compression_ratio = metadata['compression_ratio']
    print(f"压缩比: {compression_ratio:.2f}:1")
    
    # 解压
    print("\n解压中...")
    decompressed_data = huffman_decompress(compressed_data, metadata)
    print(f"解压后形状: {decompressed_data.shape}")
    
    # 验证
    is_equal = torch.equal(original_data, decompressed_data)
    print(f"\n数据完整性检查: {'通过' if is_equal else '失败'}")