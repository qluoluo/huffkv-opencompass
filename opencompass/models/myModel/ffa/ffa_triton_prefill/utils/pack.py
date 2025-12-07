import torch

def pack_k_hi_lo(k: torch.Tensor):
    """
    示例：如果需要对 K 进行量化或打包为高/低位，请在此处实现。
    """
    k_hi8 = k.to(torch.float8_e5m2).contiguous()
    k_lo8 = k.to(torch.uint8).contiguous()
    return k_hi8, k_lo8