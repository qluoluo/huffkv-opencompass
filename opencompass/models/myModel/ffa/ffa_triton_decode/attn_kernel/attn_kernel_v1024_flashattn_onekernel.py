# attn_kernel_v1022_fused_grid2d_tk.py
import os
import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# ========================
# Layout tools (merged)
# ========================
def pack_k_hi_lo(k_fp16: torch.Tensor):
    """
    Pack fp16 K into two 8-bit halves keeping the same [T, Hkv, D] layout.
    Returns:
    - k_hi8: torch.float8_e5m2 (high 8 bits), shape [T, Hkv, D]
    - k_lo8: torch.uint8        (low  8 bits), shape [T, Hkv, D]
    """
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


def convert_to_triton_layout(
    q_rope_1: torch.Tensor,  # [B=1, Hq, 1, Dq]
    k_rope: torch.Tensor,    # [B=1, Hkv, T, Dk]
    v: torch.Tensor,         # [B=1, Hkv, T, Dv]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert inputs to Triton-friendly tensors with time-major layout for K/V:
    - q_triton: [Hq, Dq]
    - k_triton_fp16: [T, Hkv, Dk]
    - v_triton: [T, Hkv, Dv]
    """
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == 1 and qlen == 1 and Tv == T and Hvv == Hkv

    q_triton = q_rope_1[0, :, 0, :].contiguous()                    # [Hq, D]
    # Time-major for K: [T, Hkv, D]
    k_triton_fp16 = k_rope[0].permute(1, 0, 2).contiguous()
    # Time-major for V: [T, Hkv, Dv]
    v_triton = v[0].permute(1, 0, 2).contiguous()
    return q_triton, k_triton_fp16, v_triton


# ========================
# Single fused kernel
# ========================
@triton.jit
def attn_forward_fused(
    q, k, v, o,
    scale, T, NTB,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
):
    # One program per query head
    pid_hq = tl.program_id(0)
    pid_hkv = pid_hq // G

    # Constants and masks
    RCP_LN2 = 1.4426950408889634
    TRUE_K = tl.full([K], True, tl.int1)
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # Load q
    offs_k = tl.arange(0, K)
    q_ptrs = q + pid_hq * K + offs_k
    q_tile = tl.load(q_ptrs, mask=TRUE_K, other=0.0).to(tl.float16)

    # Streaming softmax accumulators
    neg_inf = tl.full((), float("-inf"), tl.float32)
    g_m = neg_inf
    g_l = tl.zeros((), tl.float32)
    v_offs = tl.arange(0, V)
    g_o = tl.zeros([V], tl.float32)

    # Loop over time in blocks
    for tb in range(0, NTB):
        s0 = tb * BS
        for sb in tl.static_range(NSB):
            offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
            t_mask_sb = offs_t_sb < T

            # K layout = [T, HKV, K] => ptr = t*(HKV*K) + hkv*K + k
            kb_ptrs = k + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
            k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

            # Scores and local softmax within sub-block
            prod = q_tile.to(tl.float32)[:, None] * k_tile.to(tl.float32)
            b_s = tl.sum(prod, axis=0) * scale * RCP_LN2
            b_s_act = tl.where(t_mask_sb, b_s, float("-inf"))
            m_b = tl.max(b_s_act, axis=0)
            b_p = tl.where(t_mask_sb, tl.exp2(b_s - m_b), 0.0)
            l_b = tl.sum(b_p, axis=0)

            # Compute weighted V for this sub-block if it has any valid timesteps
            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_b = tl.zeros([V], tl.float32)
            if need_v:
                # V layout = [T, HKV, V] => ptr = t*(HKV*V) + hkv*V + v
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_b = tl.sum(b_p.to(tl.float32)[:, None] * b_v.to(tl.float32), axis=0)

            # Merge with running numerics (log-sum-exp trick)
            new_m = tl.maximum(g_m, m_b)
            r_prev = tl.exp2(g_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            g_o = g_o * r_prev + o_b * r_blk
            g_l = g_l * r_prev + l_b * r_blk
            g_m = new_m

    # Normalize and store
    out = tl.where(g_l == 0.0, tl.zeros([V], tl.float32), g_o / g_l)
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out.to(o_ptrs.dtype.element_ty))


# ========================
# Host wrapper
# ========================
def attn_forward_decode(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [T, HKV, K], kept for interface compatibility (unused)
    k_lo8: torch.Tensor,  # [T, HKV, K], kept for interface compatibility (unused)
    k_fp16: torch.Tensor, # [T, HKV, K] used
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,   # kept for interface compatibility (unused)
    return_skip_ratio: bool = False,
):
    # Use k_fp16; ignore k_hi8/k_lo8 and delta to keep interface unchanged
    assert q.is_cuda and k_fp16.is_cuda and v.is_cuda
    HQ, K = q.shape
    T, HKV, Kk = k_fp16.shape
    Tv, HKVv, V = v.shape
    assert Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [T, HKV, D]"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)

    # Output
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)

    # Single fused kernel launch: grid = (HQ,)
    attn_forward_fused[(HQ,)](
        q, k_fp16, v, o,
        scale, T, NTB,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        # You may tune:
        # num_warps=4, num_stages=2
    )

    if return_skip_ratio:
        # Analytical skip ratio: sub-blocks that have any valid steps are those with start < T
        kept_sb = triton.cdiv(T, SBS)           # number of valid sub-blocks along time
        total_sb = NTB * NSB                    # total sub-blocks per head
        skip_ratio = 1.0 - (kept_sb / float(total_sb))
        return o, skip_ratio
    else:
        return o
