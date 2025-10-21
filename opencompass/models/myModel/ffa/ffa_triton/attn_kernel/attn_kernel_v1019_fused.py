import os
import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl

@triton.jit
def attn_forward_stage1_fused_threshold(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    th_buf, th_ready, th_lock,          # 新增：阈值、就绪标志、原子锁
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    MAX_SPIN: tl.constexpr = 256,       # 自旋上限
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    ready_ptr = th_ready + pid_hkv
    lock_ptr  = th_lock  + pid_hkv
    th_ptrs   = th_buf   + (base_hq + rows)

    th_rows = tl.full([BM_DOT], NEG_INF, tl.float32)

    # 如果已就绪，直接读
    if (tl.load(ready_ptr).to(tl.int32) == 1):
        th_rows = tl.load(th_ptrs, mask=row_mask, other=NEG_INF)
    else:
        # 原子抢占：第一个把 lock 从 0 -> 1 的成为 owner
        old = tl.atomic_cas(lock_ptr, tl.full((), 0, tl.int32), tl.full((), 1, tl.int32))
        owner = (old == 0)

        if owner:
            # 计算阈值（tb0 与 tb1）
            tb0 = 0
            offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
            t_mask0 = offs_t0 < T
            kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
            k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
            b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
            b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
            m0 = tl.max(b_s0, axis=1)

            tb1 = NTB - 1
            offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
            t_mask1 = offs_t1 < T
            kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
            k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
            b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
            b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
            m1 = tl.max(b_s1, axis=1)

            m2 = tl.maximum(m0, m1)
            th_rows = m2 - delta

            # 先写阈值，再把 ready 置 1（用 CAS 0->1）
            tl.store(th_ptrs, th_rows, mask=row_mask)
            tl.atomic_cas(ready_ptr, tl.full((), 0, tl.int32), tl.full((), 1, tl.int32))
        else:
            # 无 break 的自旋等待
            got = tl.zeros((), tl.int1)                       # 是否已读到 ready
            th_tmp = tl.full([BM_DOT], NEG_INF, tl.float32)   # 临时存放阈值
            for _ in range(0, MAX_SPIN):
                rdy = tl.load(ready_ptr).to(tl.int32) == 1
                need = (~got) & rdy
                loaded = tl.load(th_ptrs, mask=(row_mask & need), other=NEG_INF)
                # 仅在 need 时更新 th_tmp
                th_tmp = tl.where(need, loaded, th_tmp)
                got = got | rdy
            if got:
                th_rows = th_tmp
            else:
                # 超时：本地计算备用阈值（不写回）
                tb0 = 0
                offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
                t_mask0 = offs_t0 < T
                kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
                k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
                b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
                b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
                m0 = tl.max(b_s0, axis=1)

                tb1 = NTB - 1
                offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
                t_mask1 = offs_t1 < T
                kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
                k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
                b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
                b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
                m1 = tl.max(b_s1, axis=1)

                m2 = tl.maximum(m0, m1)
                th_rows = m2 - delta

    # 后续计算保持不变，直接使用 th_rows
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
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

def attn_forward(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [HKV, T, K], float8_e5m2
    k_lo8: torch.Tensor,  # [HKV, T, K], uint8
    k_fp16: torch.Tensor,
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
):
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    Tv, HKVv, V = v.shape
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
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)
    
    # 新增：阈值缓冲、就绪标志、原子锁
    threshold_buf  = torch.empty((HQ,),  device=q.device, dtype=torch.float32)
    threshold_ready = torch.zeros((HKV,), device=q.device, dtype=torch.int32)
    threshold_lock  = torch.zeros((HKV,), device=q.device, dtype=torch.int32)

    attn_forward_stage1_fused_threshold[(HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf, threshold_ready, threshold_lock,   # 传入
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        MAX_SPIN=256,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2: 保持不变
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