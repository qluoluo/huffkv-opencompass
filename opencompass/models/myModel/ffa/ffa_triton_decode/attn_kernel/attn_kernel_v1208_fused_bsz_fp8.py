# attn_kernel_v1109_fused_bsz.py
import math

import torch
import triton
import triton.language as tl


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
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,        # 是否使用外部阈值
):
    # 3D grid = (NTB, B, HKV):
    #   - program_id(0) => pid_tb (大 time-block)
    #   - program_id(1) => pid_b
    #   - program_id(2) => pid_hkv
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

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
    
    # q layout = [B, HQ, K] => ptr = b*(HQ*K) + (base_hq+rows)*K + k
    q_ptrs   = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    if USE_EXT_TH:
        # 从外部缓冲读取阈值
        # th_in layout = [B, HQ] => ptr = b*HQ + (base_hq+rows)
        th_ptrs = th_in + pid_b * HQ + (base_hq + rows)
        th_rows = tl.load(th_ptrs, mask=row_mask, other=0.0)
    else:
        # 所有 kernel 独立计算阈值（tb0 与 tb_last）
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        # k_hi8 layout = [B, T, HKV, K] => ptr = b*(T*HKV*K) + t*(HKV*K) + hkv*K + k
        kb_ptrs0 = k_hi8 + pid_b * (T * HKV * K) + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        kb_ptrs1 = k_hi8 + pid_b * (T * HKV * K) + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    # 遍历当前大块内的小块（SBS）
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
# Host-side threshold (optional precompute)
# ========================
def compute_threshold_external(
    q: torch.Tensor,          # [B, HQ, K]
    k_fp16: torch.Tensor,     # [B, T, HKV, K]
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
    B, HQ_, K = q.shape
    Bk, T, HKV_, Kk = k_fp16.shape
    assert B == Bk and HQ_ == HQ and HKV_ == HKV and Kk == K
    G = HQ // HKV

    RCP_LN2 = 1.4426950408889634

    # Prepare output
    th = torch.empty((B, HQ), device=device, dtype=dtype)

    # tb0 range
    t0_lo = 0
    t0_hi = min(T_BS, T)
    # tb_last range
    t1_lo = max(0, (NTB - 1) * T_BS)
    t1_hi = min(t1_lo + T_BS, T)

    # Compute in float32 for stability
    q_f = q.to(dtype)
    k_f = k_fp16.to(dtype)

    # Iterate over batches and heads
    for b in range(B):
        for hkv in range(HKV):
            q_rows = q_f[b, hkv * G:(hkv + 1) * G]            # [G, K]
            k0 = k_f[b, t0_lo:t0_hi, hkv]                     # [t0, K]
            k1 = k_f[b, t1_lo:t1_hi, hkv]                     # [t1, K]

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

            th[b, hkv * G:(hkv + 1) * G] = torch.maximum(m0, m1) - delta

    return th


# ========================
# Host wrapper
# ========================
def attn_forward_decode(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k: torch.Tensor=None,      # [B, T, HKV, K] (可选，仅便于打包/调试/外部阈值计算)
    v: torch.Tensor=None,      # [B, T, HKV, V]
    k_hi8: torch.Tensor=None,  # [B, T, HKV, K], float8_e5m2
    k_lo8: torch.Tensor=None,  # [B, T, HKV, K], uint8 (可选，不在本实现中使用)
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,  # 外部提供的阈值（可选）
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda and k_hi8.dtype
    B, Tq, HQ, K = q.shape
    Bk, T, HKV, Kk = k_hi8.shape
    Bv, Tv, HKVv, V = v.shape
    assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [B, T, HKV, D]"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # 输出和中间缓冲现在都有batch维度
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((B, HKV, NTBS), device=q.device, dtype=torch.int8)

    # 阈值缓冲：使用外部阈值时直接传入；否则仅占位（不会读取）
    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        use_ext_th = False

    # Stage 1：grid 改为 (NTB, B, HKV) 以使时间块维度成为最快变化维
    attn_forward_stage1_fused_threshold[(NTB, B, HKV)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        USE_EXT_TH=use_ext_th,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2：reduce（grid = (B, HKV, G)）
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
