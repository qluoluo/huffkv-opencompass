import os
import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


# 辅助：QK 分块乘法（广播乘 + K 维归约），避免 tl.dot 的 16×16 要求
@triton.jit
def qk_block_mul(q_tile, k_tile):
    # q_tile: [M, K]，k_tile: [K, N]
    # 返回: [M, N], fp32
    # q_tile[:, None, :] -> [M, 1, K]
    # k_tile[None, :, :] -> [1, K, N]
    # return tl.sum(q_tile[:, None, :] * k_tile[None, :, :], axis=1).to(tl.float32)
    return tl.sum(q_tile[:, :, None] * k_tile[None, :, :], axis=1).to(tl.float32)


@triton.jit
def attn_compute_threshold_two_blocks(
    q, k_mem, threshold_buf, scale, T, NTB, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr,    # 这里用 4（2 的幂）
):
    pid_hkv = tl.program_id(0)
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)                     # [4]
    row_mask = rows < G                                 # [4]，只保留前 3 行
    offs_k   = tl.arange(0, K)                          # [K]

    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)  # [4, K]

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # tb0
    tb0      = 0
    offs_t0  = tb0 * BS + tl.arange(0, BS)
    t_mask0  = offs_t0 < T
    kb_ptrs0 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0  = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)  # [K, BS]
    b_s0     = qk_block_mul(q_tile, k_tile0) * scale * RCP_LN2                                         # [4, BS], fp32
    b_s0     = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0       = tl.max(b_s0, axis=1)                                                                     # [4]

    # tb1 = NTB-1（NTB==1 时与 tb0 相同）
    tb1      = NTB - 1
    offs_t1  = tb1 * BS + tl.arange(0, BS)
    t_mask1  = offs_t1 < T
    kb_ptrs1 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1  = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)  # [K, BS]
    b_s1     = qk_block_mul(q_tile, k_tile1) * scale * RCP_LN2                                         # [4, BS], fp32
    b_s1     = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1       = tl.max(b_s1, axis=1)                                                                     # [4]

    th = tl.maximum(m0, m1) - delta
    tl.store(threshold_buf + (base_hq + rows), th, mask=row_mask)


@triton.jit
def attn_forward_stage1_pruned(
    q, k_mem, v,
    m_buf, l_buf, o_buf,
    threshold_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr,    # 用 4（2 的幂）
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    s0       = pid_tb * BS
    rows     = tl.arange(0, BM_DOT)                      # [4]
    row_mask = rows < G                                  # [4]
    offs_k   = tl.arange(0, K)

    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)  # [4, K]

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_mem + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile  = tl.load(
            kb_ptrs,
            mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
            other=0.0
        ).to(tl.float16)                                # [K, SBS]

        # QK 打分：广播乘 + K 维归约
        b_s     = qk_block_mul(q_tile, k_tile) * scale * RCP_LN2          # [4, SBS], fp32
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)              # [4, SBS], fp32

        m_rows_blk = tl.max(b_s_act, axis=1)                               # [4]
        th_rows    = tl.load(threshold_buf + (base_hq + rows),
                             mask=row_mask, other=NEG_INF)                 # [4]
        below      = (m_rows_blk < th_rows) & row_mask
        n_below    = tl.sum(below.to(tl.int32), axis=0)
        n_valid    = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk  = n_below == n_valid

        tb_sb   = pid_tb * NSB + sb
        v_offs  = tl.arange(0, V)

        if not prune_blk:
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)  # [4, SBS], fp32
            l_rows = tl.sum(b_p, axis=1)                                                # [4], fp32

            # 计算 o_tile：广播乘 + SBS 维归约，替代 tl.dot(b_p, b_v)
            o_tile = tl.zeros([BM_DOT, V], tl.float32)                                   # [4, V]
            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            if need_v:
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float32)  # [SBS, V], fp32
                # b_p[:, :, None] -> [4, SBS, 1]
                # b_v[None, :, :] -> [1, SBS, V]
                # 结果 [4, SBS, V]，沿 SBS 求和 -> [4, V]
                o_tile = tl.sum(b_p[:, :, None] * b_v[None, :, :], axis=1).to(tl.float32)

            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))
        # else: 整个子块被剪枝，跳过计算与写回


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

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def compute_attn_thresholds(
    q: torch.Tensor,          # [HQ, K]
    k_mem: torch.Tensor,      # [HKV, T, K], float8_e5m2 或 float16（内部会统一到 fp16 计算）
    scale: float,
    BS: int,
    delta: float = 1000.0,
):
    # 阈值计算的 BS 固定为 16（2 的幂）
    BS = 16

    assert q.is_cuda and k_mem.is_cuda
    HQ, K = q.shape
    HKV, T, Kk = k_mem.shape
    assert Kk == K and (HQ % HKV == 0)
    G = HQ // HKV
    NTB = triton.cdiv(T, BS)

    # 本改版假定 G=3
    assert G == 3

    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    grid_th = (HKV, 1)
    attn_compute_threshold_two_blocks[grid_th](
        q, k_mem, threshold_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
        BM_DOT=4,                        # 用 4，避免 arange 报错
        num_warps=2, num_stages=2,
    )
    return threshold_buf


def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())


def attn_forward(
    q: torch.Tensor,         # [HQ, K]
    k_hi8: torch.Tensor,     # [HKV, T, K], float8_e5m2（高 8 位）
    k_lo8: torch.Tensor,     # [HKV, T, K], uint8（低 8 位）
    k_fp16: torch.Tensor,    # [HKV, T, K], float16（完整 k）
    v: torch.Tensor,         # [T, HKV, V]
    scale: float = None,
    BS: int = 128,           # 推荐用 64/128 等 2 的幂
    SBS: int | None = None,  # 推荐与 BS 相同，保持 2 的幂
    delta: float = 5.0,
    threshold_buf: torch.Tensor | None = None,
    return_skip_ratio: bool = False,
    th_use_fp8: bool = False,
    ker_use_fp8: bool = True,
):
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
    assert HQ % HKV == 0
    G = HQ // HKV
    assert G == 3

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    NTB  = triton.cdiv(T, BS)
    NSB  = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o     = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # 阈值：按开关选择 k_hi8 或 k_fp16
    if threshold_buf is None:
        k_for_th = k_hi8 if th_use_fp8 else k_fp16
        threshold_buf = compute_attn_thresholds(
            q, k_for_th,
            scale=scale, BS=16, delta=delta,  # 阈值内部固定 BS=16
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
        BM_DOT=4,                        # 用 4
        num_warps=2, num_stages=2,
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