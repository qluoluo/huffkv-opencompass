# utils/layout.py
import torch

__all__ = ["convert_to_triton_layout", "pack_k_hi_lo"]

def convert_to_triton_layout(q_rope_1, k_rope, v):
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == 1 and qlen == 1
    q_triton = q_rope_1[0, :, 0, :].contiguous()                  # [Hq, D]
    k_triton = k_rope[0, :, :, :].contiguous()                    # [Hkv, T, D]
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()        # [T, Hkv, Dv]
    return q_triton, k_triton, v_triton

def pack_k_hi_lo(k_fp16: torch.Tensor):
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8