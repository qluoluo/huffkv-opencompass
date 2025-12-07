# attn_kernel_v1121_fused_bsz_cache_static.py
import math
import os
from typing import Tuple

import torch
import triton
import triton.language as tl


# ========================
# Layout tools (merged)
# ========================
def convert_to_triton_layout(
    q_rope_1: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv
    q_triton = q_rope_1[:, :, 0, :].contiguous()
    k_triton_fp16 = k_rope.permute(0, 2, 1, 3).contiguous()
    v_triton = v.permute(0, 2, 1, 3).contiguous()
    return q_triton, k_triton_fp16, v_triton


def pack_k_hi_lo(k_fp16: torch.Tensor):
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


# ========================
# Kernels
# ========================
@triton.jit
def attn_forward_stage1_dynamic_threshold(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    th_buf,        # [B, HQ] 动态阈值buffer
    th_ready,      # [B, HKV] 阈值是否就绪的flag (int32, 0=未就绪, 1=就绪)
    scale, T, NTB, NTBS, delta,
    default_th,    # 默认阈值(静态离线阈值)
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
):
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    pid_tb = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K = tl.full([K], True, tl.int1)

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k = tl.arange(0, K)

    q_ptrs = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # 尝试从buffer读取阈值
    th_ready_ptr = th_ready + pid_b * HKV + pid_hkv
    is_ready = tl.load(th_ready_ptr)

    if is_ready == 1:
        th_ptrs = th_buf + pid_b * HQ + (base_hq + rows)
        th_rows = tl.load(th_ptrs, mask=row_mask, other=default_th)
    else:
        th_rows = tl.full([BM_DOT], default_th, tl.float32)

    # 检查当前block是否为tb0或tb_last，若是则计算并更新阈值
    is_tb0 = pid_tb == 0
    is_tb_last = pid_tb == (NTB - 1)

    if is_tb0 | is_tb_last:
        offs_t_th = pid_tb * T_BS + tl.arange(0, T_BS)
        t_mask_th = offs_t_th < T
        kb_ptrs_th = k_hi8 + pid_b * (T * HKV * K) + (offs_t_th[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile_th = tl.load(kb_ptrs_th, mask=(TRUE_K[:, None] & t_mask_th[None, :]), other=0.0).to(tl.float16)
        b_s_th = tl.dot(q_tile, k_tile_th, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_th = tl.where(t_mask_th[None, :], b_s_th, NEG_INF)
        m_th = tl.max(b_s_th, axis=1)  # [BM_DOT]

        # 原子更新阈值buffer - 用向量化方式
        g_offs = tl.arange(0, BM_DOT)
        g_mask = g_offs < G
        th_ptrs_atomic = th_buf + pid_b * HQ + base_hq + g_offs
        m_vals = m_th - delta
        m_vals = tl.where(g_mask, m_vals, NEG_INF)
        # 逐元素atomic max
        tl.atomic_max(th_ptrs_atomic, m_vals, mask=g_mask)

        # 如果是tb_last，标记阈值就绪
        if is_tb_last:
            tl.atomic_xchg(th_ready_ptr, 1)

        # 重新读取可能被其他kernel更新的阈值
        th_ptrs = th_buf + pid_b * HQ + (base_hq + rows)
        th_rows = tl.load(th_ptrs, mask=row_mask, other=default_th)

    # 遍历当前大块内的小块
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + pid_b * (T * HKV * K) + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # Load V之前再次检查阈值
            th_ptrs_recheck = th_buf + pid_b * HQ + (base_hq + rows)
            th_rows_new = tl.load(th_ptrs_recheck, mask=row_mask, other=default_th)

            below_new = (m_rows_blk < th_rows_new) & row_mask
            n_below_new = tl.sum(below_new.to(tl.int32), axis=0)
            prune_blk_new = n_below_new == n_valid

            if not prune_blk_new:
                m_rows = m_rows_blk
                b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
                l_rows = tl.sum(b_p, axis=1)

                need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
                o_tile = tl.zeros([BM_DOT, V], tl.float32)
                if need_v:
                    v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                    b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                    o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

                m_ptrs = m_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
                l_ptrs = l_buf + pid_b * (HQ * NTBS) + (base_hq + rows) * NTBS + tb_sb
                o_ptrs = o_buf + pid_b * (HQ * NTBS * V) + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
                tl.store(m_ptrs, m_rows, mask=row_mask)
                tl.store(l_ptrs, l_rows, mask=row_mask)
                tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
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
        keep = tl.load(mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_b * (HQ * NTBS) + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_b * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_b * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


# ========================
# Host wrapper
# ========================
def attn_forward_decode(
    q: torch.Tensor,
    k_hi8: torch.Tensor,
    k_lo8: torch.Tensor,
    k_fp16: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    default_th: float = 0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,  # 外部提供的阈值（可选）
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda
    B, HQ, K = q.shape
    Bk, T, HKV, Kk = k_hi8.shape
    Bv, Tv, HKVv, V = v.shape
    assert B == Bk == Bv and Tv == T and HKVv == HKV and Kk == K
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

    # 动态阈值buffer，初始化为-inf以便atomic_max正确工作
    th_buf = torch.full((B, HQ), float('-inf'), device=q.device, dtype=torch.float32)
    th_ready = torch.zeros((B, HKV), device=q.device, dtype=torch.int32)

    attn_forward_stage1_dynamic_threshold[(B, HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        th_buf, th_ready,
        scale, T, NTB, NTBS, delta,
        default_th,
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    attn_forward_stage2_masked[(B, HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    return o
