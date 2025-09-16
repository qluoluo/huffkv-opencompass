import math
import torch
from typing import Optional

import attn_cuda  # 编译后导入

def calc_qk_threshold(q: torch.Tensor, k: torch.Tensor, scale: float):
    # 保持与原实现一致
    HQ, K = q.shape
    HKV, T, _ = k.shape
    G = HQ // HKV
    k0 = k[:, :4, :]
    k1 = k[:, -32:, :]
    k_cat = torch.cat([k0, k1], dim=1)   # [HKV, 36, K]
    k_cat_gqa = k_cat.repeat_interleave(G, dim=0)  # [HQ, 36, K]
    dot = (q.unsqueeze(1) * k_cat_gqa).sum(dim=-1) # [HQ, 36]
    max_val = dot.max(dim=-1).values               # [HQ]
    threshold = max_val * scale - 1000
    return threshold.contiguous()

def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,   # [HQ, K]
    k: torch.Tensor,   # [HKV, T, K]
    v: torch.Tensor,   # [T, HKV, V]
    scale: Optional[float] = None,
    BS: int = 256,
    BK: int = 64,      # CUDA 版本未使用 BK；占位保持接口一致
    BV: int = 64,
    qk_thresholds: Optional[torch.Tensor] = None,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 2 and k.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k.shape
    Tv, HKV2, V = v.shape
    assert Kk == K and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if qk_thresholds is None:
        qk_thresholds = calc_qk_threshold(q, k, scale).to(torch.float32)
    else:
        assert qk_thresholds.shape == (HQ,)
        qk_thresholds = qk_thresholds.to(torch.float32)

    o, lse = attn_cuda.attn_fwd_q1_b1_splitT_cuda(
        q.contiguous(), k.contiguous(), v.contiguous(),
        qk_thresholds.contiguous(),
        float(scale), int(BS), int(BV)
    )
    return o, lse

def to_triton_layout(q_rope_1, k_rope, v):
    # q_rope_1: [B, Hq, 1, D], k_rope: [B, Hkv, T, D], v: [B, Hkv, T, Dv]
    # 返回 q:[HQ,K], k:[HKV,T,K], v:[T,HKV,V]
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv
    assert T == Tv
    assert Dq == Dk, "q/k head_dim 不一致"
    assert Hkv == Hvv, "k/v 的 head 数必须一致"
    assert B == 1, "该 kernel 仅支持 batch=1"
    assert qlen == 1, "该 kernel 仅支持 qlen=1"
    assert Hq % Hkv == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

    # 取 batch=0
    q_triton = q_rope_1[0, :, 0, :].contiguous()            # [HQ, D]
    k_triton = k_rope[0, :, :, :].contiguous()              # [HKV, T, D]
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()  # [T, HKV, Dv]
    return q_triton, k_triton, v_triton