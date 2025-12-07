import os
import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


@triton.jit
def attn_compute_threshold_two_blocks(
    q, k_mem, threshold_buf, scale, T, NTB, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    # k_mem: 可以是 fp8_e5m2 或 fp16，统一在 kernel 内转为 fp16 再计算
    pid_hkv = tl.program_id(0)
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k = tl.arange(0, K)

    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    # tb = 0
    tb0 = 0
    offs_t0 = tb0 * BS + tl.arange(0, BS)
    t_mask0 = offs_t0 < T
    kb_ptrs0 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(kb_ptrs0, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # tb = NTB-1（当 NTB==1 时等于 0，与 tb0 相同）
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(kb_ptrs1, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    m2 = tl.maximum(m0, m1)
    th = m2 - delta
    tl.store(threshold_buf + (base_hq + rows), th, mask=row_mask)


def compute_attn_thresholds(
    q: torch.Tensor,          # [HQ, K]
    k_mem: torch.Tensor,      # [HKV, T, K], float8_e5m2 或 float16（内部会统一到 fp16 计算）
    scale: float,
    BS: int,
    delta: float = 1000.0,
):
    # 注意：为与 v1019 的 T_BS=16 一致，这里固定阈值计算的块大小为 16。
    BS = 16

    assert q.is_cuda and k_mem.is_cuda
    HQ, K = q.shape
    HKV, T, Kk = k_mem.shape
    assert Kk == K and (HQ % HKV == 0)
    G = HQ // HKV
    NTB = triton.cdiv(T, BS)

    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    grid_th = (HKV, 1)
    attn_compute_threshold_two_blocks[grid_th](
        q, k_mem, threshold_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
    )
    return threshold_buf


@triton.jit
def attn_forward_stage1_pruned(
    q, k_mem, v,            # k_mem 可为 fp8_e5m2 或 fp16，统一在 kernel 内转到 fp16 参与计算
    m_buf, l_buf, o_buf,
    threshold_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    # 本 TB 的起始全局 T 偏移
    s0 = pid_tb * BS

    # 行/列常量与 Q tile
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G

    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # 常量
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # NSB 为编译期常量：每个大块内的子块数（= ceil(BS / SBS)）
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 循环每个子块
    for sb in tl.static_range(NSB):
        # 子块内的 T 偏移与 mask
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # 仅载入 k 的对应 dtype（fp8 或 fp16），并转成 fp16 参与计算
        kb_ptrs = k_mem + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(
            kb_ptrs,
            mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
            other=0.0
        ).to(tl.float16)

        # 注意：对数底为 2 的 softmax 形式
        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2  # [BM_DOT, SBS]
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        # 子块内行最大值
        m_rows_blk = tl.max(b_s_act, axis=1)

        # 阈值与裁剪
        th_rows = tl.load(threshold_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)
        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        # 写入索引：将子块映射为全局块 idx
        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # softmax 规范化（子块内）
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            # V 路
            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            # 写回（注意 stride 里的 NTBS）
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

            # 标记该 (tb, sb) 为有效
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))
        # else: 跳过


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf,     # [HQ, NTBS], [HQ, NTBS], [HQ, NTBS, V]
    mask_buf,                # [HKV, NTBS], int8（每个小块）
    o,                       # [HQ, V], out dtype = q.dtype
    NTBS,                    # int: 总块数 = NTB * NSB
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    g       = tl.program_id(1)
    pid_hq  = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)

    b_m   = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o   = tl.zeros([V], tl.float32)

    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_hq * (NTBS * V) + tb * V + v_offs)

            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk  = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o   = b_o   * r_prev + o_b * r_blk
            b_m   = new_m
        # 无效小块直接跳过

    # 空结果保护
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    # 现在的 mask_buf: [HKV, NTBS], int8, 1=keep, 0=skip
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()     # device 上做 sum，避免 int8 溢出
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())           # 注意：.item() 会触发同步


def attn_forward_decode(
    q: torch.Tensor,         # [HQ, K]
    k_hi8: torch.Tensor,     # [HKV, T, K], float8_e5m2（高 8 位）
    k_lo8: torch.Tensor,     # [HKV, T, K], uint8（低 8 位）
    k_fp16: torch.Tensor,    # [HKV, T, K], float16（完整 k）
    v: torch.Tensor,         # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    threshold_buf: torch.Tensor | None = None,
    return_skip_ratio: bool = False,
    th_use_fp8: bool = False,   # True: 阈值计算用 k_hi8；False: 用 k_fp16
    ker_use_fp8: bool = True,  # True: 主 kernel 用 k_hi8；False: 用 k_fp16
):
    # 同时存储并传入 k_hi8 / k_lo8 / k_fp16；实际计算按开关选择 k_hi8 或 k_fp16。
    assert q.is_cuda and k_hi8.is_cuda and k_lo8.is_cuda and k_fp16.is_cuda and v.is_cuda
    assert q.ndim == 2 and k_hi8.ndim == 3 and k_lo8.ndim == 3 and k_fp16.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    HKV2, T2, Kk2 = k_lo8.shape
    HKV3, T3, Kk3 = k_fp16.shape
    Tv, HKVv, V = v.shape
    assert Kk == K and Kk2 == K and Kk3 == K
    assert T2 == T and T3 == T and Tv == T
    assert HKV2 == HKV and HKV3 == HKV and HKVv == HKV
    # 类型约束（如需要可开启）
    # assert k_hi8.dtype == torch.float8_e5m2
    # assert k_lo8.dtype == torch.uint8
    # assert k_fp16.dtype == torch.float16
    assert HQ % HKV == 0
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o   = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)

    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # 阈值：按开关选择 k_hi8 或 k_fp16
    if threshold_buf is None:
        k_for_th = k_hi8 if th_use_fp8 else k_fp16
        threshold_buf = compute_attn_thresholds(
            q, k_for_th,
            scale=scale, BS=BS, delta=delta,
        )
    else:
        assert threshold_buf.shape == (HQ,) and threshold_buf.dtype == torch.float32 and threshold_buf.device == q.device

    # Stage 1：按开关选择 k_hi8 或 k_fp16
    k_for_kernel = k_hi8 if ker_use_fp8 else k_fp16
    attn_forward_stage1_pruned[(HKV, NTB)](
        q, k_for_kernel, v,
        m_buf, l_buf, o_buf,
        threshold_buf, mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        # num_stages=3, num_warps=8,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

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