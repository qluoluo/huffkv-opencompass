# attn_kernel_v1022_fused_grid2d_tk_fixed.py
import os
import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# ========================
# Helper Function
# ========================
def next_power_of_2(n):
    """
    Return the smallest power of 2 greater than or equal to n.
    """
    if n == 0:
        return 1
    # Check if n is already a power of 2
    if (n & (n - 1)) == 0:
        return n
    return int(2**math.ceil(math.log2(n)))


# ========================
# Layout tools (merged)
# ========================
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


def pack_k_hi_lo(k_fp16: torch.Tensor):
    """
    Pack fp16 K into two 8-bit halves keeping the same [T, Hkv, D] layout.
    """
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


# ========================
# Kernels
# ========================
@triton.jit
def attn_forward_stage1_fused_threshold(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    th_buf,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    T_BS: tl.constexpr,
    PADDED_BM_DOT: tl.constexpr, # Padded G
    PADDED_K: tl.constexpr,      # Padded K
    PADDED_V: tl.constexpr,      # Padded V
    PADDED_SBS: tl.constexpr,    # Padded SBS
    PADDED_T_BS: tl.constexpr,   # Padded T_BS
):
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    # --- Use PADDED values for arange, and original values for masking ---
    rows = tl.arange(0, PADDED_BM_DOT)
    row_mask = rows < G
    
    offs_k = tl.arange(0, PADDED_K)
    k_mask = offs_k < K
    q_load_mask = row_mask[:, None] & k_mask[None, :]

    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=q_load_mask, other=0.0).to(tl.float16)

    # Check dimensions for tl.dot
    USE_TC_TBS: tl.constexpr = (PADDED_BM_DOT >= 16) and (PADDED_T_BS >= 16) and (PADDED_K >= 16)
    USE_TC_SBS: tl.constexpr = (PADDED_BM_DOT >= 16) and (PADDED_SBS >= 16) and (PADDED_K >= 16)

    # --- Threshold Calculation (tb0 and tb_last) ---
    offs_t0 = tl.arange(0, PADDED_T_BS)
    t_mask0 = (pid_tb * T_BS + offs_t0) < T
    k_load_mask0 = t_mask0[None, :] & k_mask[:, None]
    
    k0_ptrs = k_hi8 + ((pid_tb * T_BS + offs_t0)[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile0 = tl.load(k0_ptrs, mask=k_load_mask0, other=0.0).to(tl.float16)
    
    if USE_TC_TBS:
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32)
    else:
        q_exp = tl.expand_dims(q_tile, 1)
        k_exp = tl.expand_dims(tl.trans(k_tile0), 0)
        b_s0 = tl.sum(q_exp * k_exp, axis=2).to(tl.float32)

    b_s0 = b_s0 * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    tb1 = NTB - 1
    offs_t1 = tl.arange(0, PADDED_T_BS)
    t_mask1 = (tb1 * T_BS + offs_t1) < T
    k_load_mask1 = t_mask1[None, :] & k_mask[:, None]

    k1_ptrs = k_hi8 + ((tb1 * T_BS + offs_t1)[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile1 = tl.load(k1_ptrs, mask=k_load_mask1, other=0.0).to(tl.float16)

    if USE_TC_TBS:
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32)
    else:
        q_exp = tl.expand_dims(q_tile, 1)
        k_exp = tl.expand_dims(tl.trans(k_tile1), 0)
        b_s1 = tl.sum(q_exp * k_exp, axis=2).to(tl.float32)

    b_s1 = b_s1 * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)
    
    th_rows = tl.maximum(m0, m1) - delta
    
    # --- Main Loop ---
    for sb in tl.static_range(NSB):
        offs_t_sb = tl.arange(0, PADDED_SBS)
        t_mask_sb = (s0 + sb * SBS + offs_t_sb) < T
        
        k_load_mask_sb = t_mask_sb[None, :] & k_mask[:, None]
        kb_ptrs = k_hi8 + ((s0 + sb * SBS + offs_t_sb)[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=k_load_mask_sb, other=0.0).to(tl.float16)
        
        if USE_TC_SBS:
            b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32)
        else:
            q_exp = tl.expand_dims(q_tile, 1)
            k_exp = tl.expand_dims(tl.trans(k_tile), 0)
            b_s = tl.sum(q_exp * k_exp, axis=2).to(tl.float32)
        
        b_s = b_s * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
        m_rows_blk = tl.max(b_s_act, axis=1)

        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = (n_below == n_valid)

        if not prune_blk:
            m_rows = m_rows_blk
            b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            offs_v = tl.arange(0, PADDED_V)
            v_mask = offs_v < V
            v_load_mask = t_mask_sb[:, None] & v_mask[None, :]

            v_ptrs = v + ((s0 + sb * SBS + offs_t_sb)[:, None] * (HKV * V)) + (pid_hkv * V) + offs_v[None, :]
            b_v = tl.load(v_ptrs, mask=v_load_mask, other=0.0).to(tl.float16)

            USE_TC_V: tl.constexpr = (PADDED_BM_DOT >= 16) and (PADDED_V >= 16) and (PADDED_SBS >= 16)
            if USE_TC_V:
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)
            else:
                p_exp = tl.expand_dims(b_p, 2)
                v_exp = tl.expand_dims(b_v, 0)
                o_tile = tl.sum(p_exp * v_exp, axis=1).to(tl.float32)

            tb_sb = pid_tb * NSB + sb
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + offs_v[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None] & v_mask[None, :])
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
    PADDED_V: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    g = tl.program_id(1)
    pid_hq = pid_hkv * G + g
    
    offs_v = tl.arange(0, PADDED_V)
    v_mask = offs_v < V
    
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([PADDED_V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_hq * (NTBS * V) + tb * V + offs_v, mask=v_mask, other=0.0)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m

    b_o = b_o / (b_acc + 1e-6) # Add epsilon to avoid division by zero
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([PADDED_V], tl.float32), b_o)
    
    o_ptrs = o + pid_hq * V + offs_v
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=v_mask)


# ========================
# Host wrapper
# ========================
def attn_forward(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [T, HKV, K], float8_e5m2
    k_lo8: torch.Tensor,  # [T, HKV, K], uint8 (可选，不在本实现中使用)
    k_fp16: torch.Tensor, # [T, HKV, K] (可选，仅便于打包/调试)
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda
    HQ, K = q.shape
    T, HKV, Kk = k_hi8.shape
    Tv, HKVv, V = v.shape
    assert Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [T, HKV, D]"
    assert HQ % HKV == 0, "HQ must be divisible by HKV"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS
    
    T_BS = 16

    # --- Pad dimensions to next power of 2 for tl.arange ---
    PADDED_BM_DOT = next_power_of_2(G)
    PADDED_K = next_power_of_2(K)
    PADDED_V = next_power_of_2(V)
    PADDED_SBS = next_power_of_2(SBS)
    PADDED_T_BS = next_power_of_2(T_BS)

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    # Use PADDED_V for buffer to match kernel's o_tile shape
    o_buf = torch.empty((HQ, NTBS, PADDED_V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)
    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    use_tensor_cores = (PADDED_BM_DOT >= 16) and (PADDED_SBS >= 16) and (PADDED_K >= 16)
    num_warps = 4 if use_tensor_cores else 2
    num_stages = 3 if use_tensor_cores else 2

    # Stage 1
    attn_forward_stage1_fused_threshold[(HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS, T_BS=T_BS,
        PADDED_BM_DOT=PADDED_BM_DOT, PADDED_K=PADDED_K, PADDED_V=PADDED_V,
        PADDED_SBS=PADDED_SBS, PADDED_T_BS=PADDED_T_BS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        if total > 0:
            skip_ratio = float((1.0 - (kept.float() / float(total))).item())
        else:
            skip_ratio = 0.0

    # Stage 2
    attn_forward_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V,
        PADDED_V=PADDED_V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o
