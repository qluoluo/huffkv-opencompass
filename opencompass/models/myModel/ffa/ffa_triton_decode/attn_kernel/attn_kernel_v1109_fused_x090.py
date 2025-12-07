# attn_kernel_v1029_fused_nothres.py
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
    Returns:
    - k_hi8: torch.float8_e5m2 (high 8 bits), shape [T, Hkv, D]
    - k_lo8: torch.uint8        (low  8 bits), shape [T, Hkv, D]
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
    th_in,                                   # 外部预计算阈值缓冲
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,        # 是否使用外部阈值
):
    # 2D grid = (HKV, NTB):
    #   - program_id(0) => pid_hkv
    #   - program_id(1) => pid_tb (大 time-block)
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # 当前 tb 对应的时间起点（大块）
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 基于当前 HKV 的 head-group
    base_hq = pid_hkv * G

    # 取 q 的一个 tile（假设 BM_DOT >= G）
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    if USE_EXT_TH:
        # 从外部缓冲读取阈值
        th_ptrs = th_in + (base_hq + rows)
        th_rows = tl.load(th_ptrs, mask=row_mask, other=0.0)
    else:
        # 所有 kernel 独立计算阈值（tb0 与 tb_last）
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        # k_hi8 layout = [T, HKV, K] => ptr = t*(HKV*K) + hkv*K + k
        kb_ptrs0 = k_hi8 + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        kb_ptrs1 = k_hi8 + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    # 遍历当前大块内的小块（SBS）
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
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
                # v layout = [T, HKV, V] => ptr = t*(HKV*V) + hkv*V + v
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    g = tl.program_id(1)
    pid_hq = pid_hkv * G + g
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hkv * NTBS + tb).to(tl.int1)
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
# Host-side threshold (optional precompute)
# ========================
def compute_threshold_external(
    q: torch.Tensor,          # [HQ, K]
    k_fp16: torch.Tensor,     # [T, HKV, K]
    scale: float,
    NTB: int,
    delta: float,
    HKV: int,
    HQ: int,
    T_BS: int = 16,
) -> torch.Tensor:
    """
    Replicates the in-kernel threshold computation on host/GPU:
    th_rows = max(max_t dot(q, k[t]) over t in tb0, max_t over t in tb_last) - delta
    where tb0 = [0 .. T_BS-1], tb_last = [(NTB-1)*T_BS .. (NTB-1)*T_BS+T_BS-1], clipped to [0, T).
    """
    assert q.is_cuda and k_fp16.is_cuda
    device = q.device
    dtype = torch.float32
    HQ_, K = q.shape
    T, HKV_, Kk = k_fp16.shape
    assert HQ_ == HQ and HKV_ == HKV and Kk == K
    G = HQ // HKV

    RCP_LN2 = 1.4426950408889634

    # Prepare output
    th = torch.empty((HQ,), device=device, dtype=dtype)

    # tb0 range
    t0_lo = 0
    t0_hi = min(T_BS, T)
    # tb_last range
    t1_lo = max(0, (NTB - 1) * T_BS)
    t1_hi = min(t1_lo + T_BS, T)

    # Compute in float32 for stability
    q_f = q.to(dtype)
    k_f = k_fp16.to(dtype)

    # Iterate heads to respect grouping
    for hkv in range(HKV):
        q_rows = q_f[hkv * G:(hkv + 1) * G]            # [G, K]
        k0 = k_f[t0_lo:t0_hi, hkv]                     # [t0, K]
        k1 = k_f[t1_lo:t1_hi, hkv]                     # [t1, K]

        if k0.numel() > 0:
            s0 = (q_rows @ k0.T) * (scale * RCP_LN2)  # [G, t0]
            m0 = s0.max(dim=1).values
        else:
            m0 = torch.full((G,), float("-inf"), device=device, dtype=dtype)

        if k1.numel() > 0:
            s1 = (q_rows @ k1.T) * (scale * RCP_LN2)  # [G, t1]
            m1 = s1.max(dim=1).values
        else:
            m1 = torch.full((G,), float("-inf"), device=device, dtype=dtype)

        th[hkv * G:(hkv + 1) * G] = torch.maximum(m0, m1) - delta

    return th


# ========================
# Host wrapper
# ========================
def attn_forward(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [T, HKV, K], float8_e5m2
    k_lo8: torch.Tensor,  # [T, HKV, K], uint8 (可选，不在本实现中使用)
    k_fp16: torch.Tensor, # [T, HKV, K] (可选，仅便于打包/调试/外部阈值计算)
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,  # 外部提供的阈值（可选）
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda
    HQ, K = q.shape
    T, HKV, Kk = k_hi8.shape
    Tv, HKVv, V = v.shape
    assert Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [T, HKV, D]"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    # 注意：Stage1 内部使用 T_BS=16 计算阈值的 tb0/tb_last 索引，
    # 这里 NTB 仍按照 BS 计算以匹配原始实现的行为。
    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # 阈值缓冲：使用外部阈值时直接传入；否则仅占位（不会读取）
    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (HQ,)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)
        use_ext_th = False

    # Stage 1：grid 改为 (HKV, NTB)
    attn_forward_stage1_fused_threshold[(HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        USE_EXT_TH=use_ext_th,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2：reduce（grid = (HKV, G) 保持不变）
    attn_forward_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o
