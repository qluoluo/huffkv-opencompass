# attn_kernel_v1208_fused_bsz_q2.py
# 2-bit quantized K version derived from attn_kernel_v1208_fused_bsz_fp8.py
import math

import torch
import triton
import triton.language as tl


@triton.jit
def attn_forward_stage1_fused_threshold_qbits(
    q, k_q, k_scale, k_zp, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    th_in,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    K_BITS: tl.constexpr = 2,
    SCALE_PER_TOKEN: tl.constexpr = True,      # True: k_scale/k_zp shape [B, T, HKV]
    USE_EXT_TH: tl.constexpr = False,
):
    # 3D grid = (NTB, B, HKV)
    pid_tb = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hkv = tl.program_id(2)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)
    QMAX = (1 << K_BITS) - 1

    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)

    q_ptrs   = q + pid_b * (HQ * K) + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    if USE_EXT_TH:
        th_rows = tl.load(th_in + pid_b * HQ + (base_hq + rows), mask=row_mask, other=0.0)
    else:
        tb0 = 0
        offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
        t_mask0 = offs_t0 < T
        base_tok0 = pid_b * (T * HKV * K) + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K)
        kq_ptrs0 = k_q + base_tok0 + offs_k[:, None]
        kq_tile0 = tl.load(kq_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0).to(tl.int32)
        kq_tile0 = tl.minimum(kq_tile0, tl.full((), QMAX, tl.int32)).to(tl.float32)
        if SCALE_PER_TOKEN:
            scale_ptrs0 = k_scale + pid_b * (T * HKV) + offs_t0 * HKV + pid_hkv
            zp_ptrs0    = k_zp    + pid_b * (T * HKV) + offs_t0 * HKV + pid_hkv
            scale_tile0 = tl.load(scale_ptrs0, mask=t_mask0, other=0.0)[None, :].to(tl.float32)
            zp_tile0    = tl.load(zp_ptrs0,    mask=t_mask0, other=0.0)[None, :].to(tl.float32)
        else:
            scale_ptrs0 = k_scale + base_tok0 + offs_k[:, None]
            zp_ptrs0    = k_zp    + base_tok0 + offs_k[:, None]
            scale_tile0 = tl.load(scale_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float32)
            zp_tile0    = tl.load(zp_ptrs0,    mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float32)
        k_tile0 = (kq_tile0 * scale_tile0 + zp_tile0).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
        m0 = tl.max(b_s0, axis=1)

        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
        t_mask1 = offs_t1 < T
        base_tok1 = pid_b * (T * HKV * K) + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K)
        kq_ptrs1 = k_q + base_tok1 + offs_k[:, None]
        kq_tile1 = tl.load(kq_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0).to(tl.int32)
        kq_tile1 = tl.minimum(kq_tile1, tl.full((), QMAX, tl.int32)).to(tl.float32)
        if SCALE_PER_TOKEN:
            scale_ptrs1 = k_scale + pid_b * (T * HKV) + offs_t1 * HKV + pid_hkv
            zp_ptrs1    = k_zp    + pid_b * (T * HKV) + offs_t1 * HKV + pid_hkv
            scale_tile1 = tl.load(scale_ptrs1, mask=t_mask1, other=0.0)[None, :].to(tl.float32)
            zp_tile1    = tl.load(zp_ptrs1,    mask=t_mask1, other=0.0)[None, :].to(tl.float32)
        else:
            scale_ptrs1 = k_scale + base_tok1 + offs_k[:, None]
            zp_ptrs1    = k_zp    + base_tok1 + offs_k[:, None]
            scale_tile1 = tl.load(scale_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float32)
            zp_tile1    = tl.load(zp_ptrs1,    mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float32)
        k_tile1 = (kq_tile1 * scale_tile1 + zp_tile1).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
        m1 = tl.max(b_s1, axis=1)

        th_rows = tl.maximum(m0, m1) - delta

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        base_toksb = pid_b * (T * HKV * K) + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K)
        kq_ptrssb = k_q + base_toksb + offs_k[:, None]
        kq_tilesb = tl.load(kq_ptrssb, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0).to(tl.int32)
        kq_tilesb = tl.minimum(kq_tilesb, tl.full((), QMAX, tl.int32)).to(tl.float32)
        if SCALE_PER_TOKEN:
            scale_ptrssb = k_scale + pid_b * (T * HKV) + offs_t_sb * HKV + pid_hkv
            zp_ptrssb    = k_zp    + pid_b * (T * HKV) + offs_t_sb * HKV + pid_hkv
            scale_tilesb = tl.load(scale_ptrssb, mask=t_mask_sb, other=0.0)[None, :].to(tl.float32)
            zp_tilesb    = tl.load(zp_ptrssb,    mask=t_mask_sb, other=0.0)[None, :].to(tl.float32)
        else:
            scale_ptrssb = k_scale + base_toksb + offs_k[:, None]
            zp_ptrssb    = k_zp    + base_toksb + offs_k[:, None]
            scale_tilesb = tl.load(scale_ptrssb, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float32)
            zp_tilesb    = tl.load(zp_ptrssb,    mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float32)
        k_tile = (kq_tilesb * scale_tilesb + zp_tilesb).to(tl.float16)
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
                v_ptrs = v + pid_b * (T * HKV * V) + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
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


def _normalize_scale_zero(k_scale: torch.Tensor, k_zero: torch.Tensor, expect_shape3, expect_shape4):
    """
    Ensure scale / zero_point tensors are contiguous and have a supported shape.
    Returns (scale, zero_point, per_token_scale_flag).
    """
    if k_scale.ndim == 4 and k_scale.shape[-1] == 1:
        k_scale = k_scale.squeeze(-1)
    if k_zero.ndim == 4 and k_zero.shape[-1] == 1:
        k_zero = k_zero.squeeze(-1)

    if k_scale.shape == expect_shape3 and k_zero.shape == expect_shape3:
        per_token_scale = True
    elif k_scale.shape == expect_shape4 and k_zero.shape == expect_shape4:
        per_token_scale = False
    else:
        raise ValueError(f"Unsupported k_scale/k_zero shapes: {k_scale.shape=} {k_zero.shape=}, "
                         f"expected {expect_shape3} or {expect_shape4}")

    return k_scale.contiguous(), k_zero.contiguous(), per_token_scale


def attn_forward_decode_quantized(
    q: torch.Tensor,           # [B, 1, HQ, K]
    k_q: torch.Tensor,         # [B, T, HKV, K], quantized ints in [0, 2^k_bits-1]
    k_scale: torch.Tensor,     # [B, T, HKV] or [B, T, HKV, K]
    k_zero: torch.Tensor,      # same shape as k_scale
    v: torch.Tensor,           # [B, T, HKV, V]
    k_bits: int = 2,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    precomputed_threshold: torch.Tensor | None = None,
    **kwargs,
):
    assert q.is_cuda and k_q.is_cuda and v.is_cuda
    if k_bits != 2:
        raise ValueError(f"attn_forward_decode_quantized currently supports 2-bit keys, got k_bits={k_bits}")
    assert k_scale.is_cuda and k_zero.is_cuda, "k_scale/k_zero must be CUDA tensors"
    if not k_scale.is_floating_point() or not k_zero.is_floating_point():
        raise ValueError("k_scale and k_zero must be floating point tensors for dequantization")
    if k_q.is_floating_point():
        raise ValueError("k_q must contain integer quantized values (e.g., uint8/int8)")

    B, Tq, HQ, K = q.shape
    Bk, T, HKV, Kk = k_q.shape
    Bv, Tv, HKVv, V = v.shape
    assert B == Bk == Bv and Tq == 1 and Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [B, T, HKV, D]"
    G = HQ // HKV

    expect_shape3 = (B, T, HKV)
    expect_shape4 = (B, T, HKV, K)
    k_scale, k_zero, per_token_scale = _normalize_scale_zero(k_scale, k_zero, expect_shape3, expect_shape4)

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    q = q.contiguous()
    k_q = k_q.contiguous()
    v = v.contiguous()
    o = torch.empty((B, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((B, HKV, NTBS), device=q.device, dtype=torch.int8)

    if precomputed_threshold is not None:
        assert precomputed_threshold.is_cuda and precomputed_threshold.shape == (B, HQ)
        threshold_buf = precomputed_threshold.contiguous()
        use_ext_th = True
    else:
        threshold_buf = torch.empty((B, HQ), device=q.device, dtype=torch.float32)
        use_ext_th = False

    attn_forward_stage1_fused_threshold_qbits[(NTB, B, HKV)](
        q, k_q, k_scale, k_zero, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf,
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        K_BITS=k_bits, SCALE_PER_TOKEN=per_token_scale, USE_EXT_TH=use_ext_th,
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
    else:
        return o
