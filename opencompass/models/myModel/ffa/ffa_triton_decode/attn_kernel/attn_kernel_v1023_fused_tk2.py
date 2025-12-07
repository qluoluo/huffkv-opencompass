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
def attn_forward_stage1_multi_hkv_softpath(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    HKV_PER_CTA: tl.constexpr = 4,
):
    # grid(0) = ceil(HKV / HKV_PER_CTA), grid(1) = NTB
    pid_blk = tl.program_id(0)
    pid_tb = tl.program_id(1)
    hkv_start = pid_blk * HKV_PER_CTA
    # 有效 lane 数（最后一个 block 可能不足 HKV_PER_CTA）
    valid_hpc = tl.minimum(HKV - hkv_start, HKV_PER_CTA)
    # 将多个 HKV 的 G 行打包为 BM_DOT 行；有效行是 valid_hpc * G
    rows = tl.arange(0, BM_DOT)
    M_ROWS = valid_hpc * G
    row_valid = rows < M_ROWS
    # 行到 (hkv, g) 的映射
    lane = rows // G                     # [0 .. HKV_PER_CTA-1]
    g    = rows % G                      # [0 .. G-1]
    hkv_rows = hkv_start + lane          # 每行对应的 HKV id
    hq_rows  = hkv_rows * G + g          # 每行对应的 HQ id
    # K 维偏移
    offs_k = tl.arange(0, K)
    # q: [HQ, K]
    q_ptrs = q + hq_rows[:, None] * K + offs_k[None, :]
    # 只加载有效行
    q_tile_f16 = tl.load(q_ptrs, mask=row_valid[:, None], other=0.0).to(tl.float16)
    # q 做浮点乘法累加，用 fp32 累加更稳妥
    q_tile_f32 = q_tile_f16.to(tl.float32)
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    # 大时间块起始
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    # 计算阈值：tb0 与 tb_last
    tb0 = 0
    tb1 = NTB - 1
    def compute_block_max(tb_idx):
        t_offs = tb_idx * T_BS + tl.arange(0, T_BS)
        t_mask = t_offs < T
        # k_hi8: [T, HKV, K]
        # 为每一行（对应不同 HKV）构造 3D 指针 [BM_DOT, K, T_BS]
        k_ptrs_3d = k_hi8 + \
            (t_offs[None, None, :] * (HKV * K)) + \
            (hkv_rows[:, None, None] * K) + \
            offs_k[None, :, None]
        # 只对有效行+有效 t 读；K 维恒定匹配
        k_3d_f16 = tl.load(
            k_ptrs_3d,
            mask=row_valid[:, None, None] & t_mask[None, None, :],
            other=0.0
        ).to(tl.float16)
        # [BM_DOT, T_BS] = sum_k (q[row,k] * k[row,k,t])
        b_s_f32 = tl.sum(q_tile_f32[:, :, None] * k_3d_f16.to(tl.float32), axis=1)
        b_s_ln2 = b_s_f32 * scale * RCP_LN2
        # 无效 t 列设为 -inf
        b_s_ln2 = tl.where(t_mask[None, :], b_s_ln2, NEG_INF)
        # 每行取最大
        m_rows = tl.max(b_s_ln2, axis=1)
        return m_rows, b_s_ln2, t_offs, t_mask
    m0, _, _, _ = compute_block_max(tb0)
    m1, _, _, _ = compute_block_max(tb1)
    # 行阈值
    th_rows = tl.maximum(m0, m1) - delta
    # 遍历小块
    v_offs = tl.arange(0, V)
    for sb in tl.static_range(NSB):
        t_offs = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask = t_offs < T
        # 重新加载该小块的 k
        k_ptrs_3d = k_hi8 + \
            (t_offs[None, None, :] * (HKV * K)) + \
            (hkv_rows[:, None, None] * K) + \
            offs_k[None, :, None]
        k_3d_f16 = tl.load(
            k_ptrs_3d,
            mask=row_valid[:, None, None] & t_mask[None, None, :],
            other=0.0
        ).to(tl.float16)
        # [BM_DOT, SBS]
        b_s_f32 = tl.sum(q_tile_f32[:, :, None] * k_3d_f16.to(tl.float32), axis=1)
        b_s_ln2 = b_s_f32 * scale * RCP_LN2
        # 无效 t 设为 -inf
        b_s_act = tl.where(t_mask[None, :], b_s_ln2, NEG_INF)
        # 每行小块最大
        m_rows_blk = tl.max(b_s_act, axis=1)
        # 判断每行是否低于阈值
        below = (m_rows_blk < th_rows) & row_valid
        tb_sb = pid_tb * NSB + sb
        # 对每个 HKV lane 单独做 prune / 计算 / 写回
        for l in tl.static_range(HKV_PER_CTA):
            # 该 lane 是否有效
            lane_valid = l < valid_hpc
            # 该 lane 对应的行范围 [l*G, l*G+G)
            r0 = l * G
            r1 = r0 + G
            # 该 lane 的 HKV id
            hkv_l = hkv_start + l
            # 行子片
            # Triton 支持 compile-time slice；l 是静态循环常量，G 是 constexpr
            below_slice = below[r0:r1]
            # 有效行数（最后一个 block 不足 HKV_PER_CTA 时，超过 valid_hpc 的 lane 视为 0 行）
            valid_rows_l = tl.where(lane_valid, G, 0)
            # 统计该 lane 的所有行都低于阈值
            n_below_l = tl.sum(below_slice.to(tl.int32), axis=0)
            prune_l = (n_below_l == valid_rows_l) & lane_valid
            if not prune_l:
                # b_p = exp2(b_s - m_rows)
                m_rows_l = m_rows_blk[r0:r1]            # [G]
                b_s_l    = b_s_ln2[r0:r1, :]            # [G, SBS]
                b_p_l    = tl.where(
                    t_mask[None, :],
                    tl.exp2(b_s_l - m_rows_l[:, None]),
                    0.0
                )
                # l_rows = sum_t b_p
                l_rows_l = tl.sum(b_p_l, axis=1)        # [G]
                # 计算 o_tile 行片：逐 t 聚合 v[t, hkv_l, :]
                o_slice = tl.zeros([G, V], tl.float32)
                # 如果该小块有任何有效 t
                need_v = tl.sum(t_mask.to(tl.int32), axis=0) > 0
                if need_v:
                    for t_rel in tl.static_range(SBS):
                        t_valid = t_mask[t_rel]
                        if t_valid:
                            t_idx = t_offs[t_rel]
                            v_ptrs = v + (t_idx * (HKV * V)) + (hkv_l * V) + v_offs
                            v_t = tl.load(v_ptrs).to(tl.float16)  # [V]
                            w = b_p_l[:, t_rel]                   # [G]
                            # o += w[:,None] * v_t[None,:]
                            o_slice += (w[:, None].to(tl.float32)) * v_t[None, :].to(tl.float32)
                # 写回 m/l/o
                # 行对应的 HQ 索引：hq = hkv_l*G + g
                hq_base = hkv_l * G
                g_offs  = tl.arange(0, G)
                # m/l: [HQ, NTBS]；o_buf: [HQ, NTBS, V]
                m_ptrs = m_buf + (hq_base + g_offs) * NTBS + tb_sb
                l_ptrs = l_buf + (hq_base + g_offs) * NTBS + tb_sb
                o_ptrs = o_buf + (hq_base + g_offs)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
                tl.store(m_ptrs, m_rows_l)
                tl.store(l_ptrs, l_rows_l)
                tl.store(o_ptrs, o_slice)
                # 写 mask：该 HKV/tb_sb 保留
                tl.store(mask_buf + hkv_l * NTBS + tb_sb, tl.full((), 1, tl.int8))


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

    # 可选：阈值缓冲（用于调试/分析）
    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    # Stage 1：grid 改为 (HQ, NTB)
    attn_forward_stage1_fused_threshold[(HQ, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
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
