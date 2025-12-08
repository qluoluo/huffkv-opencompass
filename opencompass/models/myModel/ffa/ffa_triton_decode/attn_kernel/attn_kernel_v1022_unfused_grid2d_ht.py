# attn_kernel_v1022_split_threshold.py
import os
import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# ========================
# Layout tools (merged)
# ========================
def convert_to_triton_layout(
    q_rope_1: torch.Tensor,  # [B, Hq, 1, Dq]
    k_rope: torch.Tensor,    # [B, Hkv, T, Dk]
    v: torch.Tensor,         # [B, Hkv, T, Dv]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert inputs to Triton-friendly tensors with time-major layout for K/V:
    - q_triton: [B, Hq, Dq]
    - k_triton_fp16: [B, T, Hkv, Dk]
    - v_triton: [B, T, Hkv, Dv]
    """
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv

    q_triton = q_rope_1[:, :, 0, :].contiguous()                    # [B, Hq, D]
    # Time-major for K: [B, T, Hkv, D]
    k_triton_fp16 = k_rope.permute(0, 2, 1, 3).contiguous()
    # Time-major for V: [B, T, Hkv, Dv]
    v_triton = v.permute(0, 2, 1, 3).contiguous()
    return q_triton, k_triton_fp16, v_triton


def pack_k_hi_lo(k_fp16: torch.Tensor):
    """
    Pack fp16 K into two 8-bit halves keeping the same [B, T, Hkv, D] layout.
    Returns:
    - k_hi8: torch.float8_e5m2 (high 8 bits), shape [B, T, Hkv, D]
    - k_lo8: torch.uint8        (low  8 bits), shape [B, T, Hkv, D]
    """
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


# ========================
# New: threshold kernel
# ========================
@triton.jit
def compute_thresholds_kernel(
    q, k_hi8, th_buf,
    scale, T, NTB, delta,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
):
    # 2D grid = (B, HKV). Each program computes thresholds for G query rows of this KV head in batch pid_b.
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    base_hq = pid_hkv * G

    # Load a tile of q (assume BM_DOT >= G)
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    # q layout = [B, HQ, K]
    q_ptrs   = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # First small time-chunk in tb0
    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    # k_hi8 layout = [B, T, HKV, K]
    kb_ptrs0 = k_hi8 + pid_b * (T * HKV * K) + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # Last small time-chunk in tb1 (could equal tb0 if NTB == 1)
    tb1 = NTB - 1
    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_hi8 + pid_b * (T * HKV * K) + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    # Threshold per row in this group: max(m0, m1) - delta
    th_rows = tl.maximum(m0, m1) - delta

    # Store to th_buf[base_hq : base_hq + G]
    # th_buf layout = [B, HQ]
    th_ptrs = th_buf + pid_b * HQ + (base_hq + rows)
    tl.store(th_ptrs, th_rows, mask=row_mask)


# ========================
# Stage-1: pruning + partial reductions (threshold read from buffer)
# ========================
@triton.jit
def attn_forward_stage1_pruning(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    th_buf,  # read-only thresholds of shape [B, HQ]
    scale, T, NTB, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    # 3D grid = (NTB, B, HKV):
    #   - program_id(0) => pid_tb (big time-block)
    #   - program_id(1) => pid_b
    #   - program_id(2) => pid_hkv
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # Current big time-block's start
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # head group base index
    base_hq = pid_hkv * G

    # Load q tile
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    # q layout = [B, HQ, K]
    q_ptrs   = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # Load thresholds per row
    # th_buf layout = [B, HQ]
    th_rows = tl.load(th_buf + pid_b * HQ + (base_hq + rows), mask=row_mask, other=float("-inf"))

    # Iterate small blocks in this big block
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + pid_b * (T * HKV * K) + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                # v layout = [B, T, HKV, V] => ptr = b*(T*HKV*V) + t*(HKV*V) + hkv*V + v
                v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            # m_buf/l_buf/o_buf layout = [B, HQ, NTBS] / [B, HQ, NTBS] / [B, HQ, NTBS, V]
            m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            # mask_buf layout = [B, HKV, NTBS]
            tl.store(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2)
    pid_hq = pid_hkv * G + g
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        # mask_buf layout = [B, HKV, NTBS]
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            # m_buf/l_buf layout = [B, HQ, NTBS]
            m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            # o_buf layout = [B, HQ, NTBS, V]
            o_b = tl.load(o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    # o layout = [B, HQ, V]
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


# ========================
# Host wrapper
# ========================
def attn_forward_decode(
    q: torch.Tensor,      # [B, HQ, K]
    k_hi8: torch.Tensor,  # [B, T, HKV, K], float8_e5m2
    k_lo8: torch.Tensor,  # [B, T, HKV, K], uint8 (可选，不在本实现中使用)
    k_fp16: torch.Tensor, # [B, T, HKV, K] (可选，仅便于打包/调试)
    v: torch.Tensor,      # [B, T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    **kwargs,
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda
    B, HQ, K = q.shape
    Bk, T, HKV, Kk = k_hi8.shape
    Bv, Tv, HKVv, V = v.shape
    assert B == Bk == Bv and Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [B, T, HKV, D]"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((B, HKV, NTBS), device=q.device, dtype=torch.int8)

    # 1) Compute thresholds in a dedicated kernel
    threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
    compute_thresholds_kernel[(B, HKV)](
        q, k_hi8, threshold_buf,
        scale, T, NTB, delta,
        B=B, HKV=HKV, HQ=HQ, K=K, G=G,
        # keep the same micro-params as before
        BM_DOT=16, T_BS=16,
    )

    # 2) Stage 1: prune and compute partial reductions (read thresholds)
    # grid aligned with v1109: (NTB, B, HKV) so time-block dimension varies fastest
    attn_forward_stage1_pruning[(NTB, B, HKV)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        threshold_buf,
        scale, T, NTB, NTBS,
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        BM_DOT=16,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # 3) Stage 2: masked reduce across kept blocks（grid = (B, HKV, G)）
    attn_forward_stage2_masked[(B, HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o
