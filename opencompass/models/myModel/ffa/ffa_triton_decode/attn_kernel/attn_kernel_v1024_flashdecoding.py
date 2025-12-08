# attn_kernel_v1022_fused_grid2d_tk.py
import os
import math
from typing import Tuple

import torch
import triton
import triton.language as tl

# from .attn_kernel_v1023_fused_tk import pack_k_hi_lo, convert_to_triton_layout


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
# Kernels
# ========================
@triton.jit
def attn_forward_stage1_full(
    q, k, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    T_BS: tl.constexpr = 16,  # kept for interface consistency (unused)
):
    # 2D grid = (HQ, NTB):
    #   - program_id(0) => pid_hq
    #   - program_id(1) => pid_tb (large time-block)
    pid_hq = tl.program_id(0)
    pid_tb = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # current large block start
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # corresponding KV head index
    pid_hkv = pid_hq // G

    # load q vector
    offs_k = tl.arange(0, K)
    q_ptrs = q + pid_hq * K + offs_k
    q_tile = tl.load(q_ptrs, mask=TRUE_K, other=0.0).to(tl.float16)

    # iterate sub-blocks within this large block
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # k layout = [T, HKV, K] => ptr = t*(HKV*K) + hkv*K + k
        kb_ptrs = k + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        prod = q_tile.to(tl.float32)[:, None] * k_tile.to(tl.float32)
        b_s = tl.sum(prod, axis=0) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb, b_s, NEG_INF)

        # per-subblock numerics
        m_row = tl.max(b_s_act, axis=0)
        b_p = tl.where(t_mask_sb, tl.exp2(b_s - m_row), 0.0)  # [SBS]
        l_row = tl.sum(b_p, axis=0)

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        # whether this sub-block has any valid time steps
        need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
        o_tile = tl.zeros([V], tl.float32)
        if need_v:
            # v layout = [T, HKV, V] => ptr = t*(HKV*V) + hkv*V + v
            v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
            o_tile = tl.sum(b_p.to(tl.float32)[:, None] * b_v.to(tl.float32), axis=0)

        m_ptrs = m_buf + pid_hq * NTBS + tb_sb
        l_ptrs = l_buf + pid_hq * NTBS + tb_sb
        o_ptrs = o_buf + pid_hq * (NTBS * V) + tb_sb * V + v_offs
        tl.store(m_ptrs, m_row)
        tl.store(l_ptrs, l_row)
        tl.store(o_ptrs, o_tile)
        tl.store(mask_buf + pid_hq * NTBS + tb_sb, need_v.to(tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    HQ: tl.constexpr, V: tl.constexpr,
):
    # 1D grid = (HQ,)
    pid_hq = tl.program_id(0)
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hq * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_hq * (NTBS * V) + tb * V + v_offs)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


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
    precomputed_threshold: torch.Tensor | None = None,  # kept for interface compatibility (unused)
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
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HQ, NTBS), device=q.device, dtype=torch.int8)

    # Stage 1：grid = (HQ, NTB)
    attn_forward_stage1_full[(HQ, NTB)](
        q, k_fp16, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        # This ratio reflects only padding-induced skips (no pruning)
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2：reduce（grid = (HQ,)）
    attn_forward_stage2_masked[(HQ,)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o
