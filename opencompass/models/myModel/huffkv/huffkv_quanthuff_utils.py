import os, torch

from .huffkv_quant_utils import quantize, dequantize
from .huffkv_encode_utils import huffman_compress, huffman_decompress

def quanthuff_compress(x: torch.Tensor, dim: int = -1, nbits: int = 4):

    quant_data, quant_metadata = quantize(x, nbits=nbits, dim=dim)
    huff_data, huff_metadata = huffman_compress(quant_data)

    return huff_data, (quant_metadata, huff_metadata)

def quanthuff_decompress(huff_data: torch.Tensor, metadata: tuple):

    quant_metadata, huff_metadata = metadata

    quant_data = huffman_decompress(huff_data, huff_metadata)
    x = dequantize(quant_data, quant_metadata)

    return x