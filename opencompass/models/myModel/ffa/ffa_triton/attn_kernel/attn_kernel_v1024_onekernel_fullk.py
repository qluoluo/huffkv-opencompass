# attn_kernel_v1022_fused_grid2d_ht.py
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
    Note: Not used in the fused implementation below; kept for interface compatibility.
    """
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


# ========================
# Fused Kernel: stage1 + stage2 (only k_fp16)
# ========================
@triton.jit
def attn_forward_fused_stage12_fp16(
    q, k_fp16, v, o,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
):
    """
    Fused attention forward:
    - Computes threshold from first/last big time-blocks
    - Prunes blocks below threshold
    - Accumulates stable softmax across kept blocks
    - Produces final output o for all G query heads within the KV head

    Grid: (HKV,)
    """
    pid_hkv = tl.program_id(0)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # Head-group base
    base_hq = pid_hkv * G

    # Load q tile for up to BM_DOT rows in the group
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # Compute threshold using first and last big-block (tb0=0, tb1=NTB-1)
    # First block (take first T_BS tokens)
    tb0 = 0
    offs_t0 = tb0 * BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    kb_ptrs0 = k_fp16 + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile0  = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0     = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0     = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0       = tl.max(b_s0, axis=1)

    # Last block (take first T_BS tokens of last big-block)
    tb1      = NTB - 1
    offs_t1  = tb1 * BS + tl.arange(0, T_BS)
    t_mask1  = offs_t1 < T
    kb_ptrs1 = k_fp16 + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
    k_tile1  = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1     = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1     = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1       = tl.max(b_s1, axis=1)

    th_rows = tl.maximum(m0, m1) - delta

    # Stable accumulation buffers across all blocks
    v_offs = tl.arange(0, V)
    b_m    = tl.full([BM_DOT], NEG_INF, tl.float32)
    b_acc  = tl.zeros([BM_DOT], tl.float32)
    b_o    = tl.zeros([BM_DOT, V], tl.float32)

    # NSB (compile-time) small-blocks per big-block
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # Traverse all big-blocks and their small-blocks
    for tb in range(0, NTB):
        s0 = tb * BS
        for sb in tl.static_range(NSB):
            offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
            t_mask_sb = offs_t_sb < T

            # K block: [SBS, K]
            kb_ptrs = k_fp16 + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
            k_tile  = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

            # Scores in log2 domain
            b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
            b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

            # Per-row block maxima
            m_rows_blk = tl.max(b_s_act, axis=1)

            # Prune decision: prune if all active rows are below threshold
            below    = (m_rows_blk < th_rows) & row_mask
            n_below  = tl.sum(below.to(tl.int32), axis=0)
            n_valid  = tl.sum(row_mask.to(tl.int32), axis=0)
            prune_blk = n_below == n_valid

            tb_sb = tb * NSB + sb

            if not prune_blk:
                # Normalize by per-row block maxima
                m_rows = m_rows_blk
                b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
                l_rows = tl.sum(b_p, axis=1)

                # V block: [SBS, V]
                need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
                o_tile_blk = tl.zeros([BM_DOT, V], tl.float32)
                if need_v:
                    v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                    b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                    o_tile_blk = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

                # Stable accumulation across blocks
                new_m = tl.maximum(b_m, m_rows)
                r_prev = tl.exp2(b_m - new_m)
                r_blk  = tl.exp2(m_rows - new_m)
                b_acc  = b_acc * r_prev + l_rows * r_blk
                b_o    = b_o * r_prev[:, None] + o_tile_blk * r_blk[:, None]
                b_m    = new_m

                # Mark block as kept (for skip-ratio)
                tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))
            # else: pruned block => mask remains 0

    # Final normalization
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty[:, None], tl.zeros([BM_DOT, V], tl.float32), b_o / b_acc[:, None])

    # Write back results
    o_ptrs = o + (base_hq + rows)[:, None] * V + v_offs[None, :]
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=row_mask[:, None])


# ========================
# Host wrapper
# ========================
def attn_forward(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [T, HKV, K], float8_e5m2 (ignored in fused impl; kept for API compatibility)
    k_lo8: torch.Tensor,  # [T, HKV, K], uint8        (ignored in fused impl; kept for API compatibility)
    k_fp16: torch.Tensor, # [T, HKV, K]
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
):
    """
    Fused stage1+stage2 implementation using only k_fp16.

    Interface remains unchanged; k_hi8 and k_lo8 are ignored.
    """
    assert q.is_cuda and k_fp16.is_cuda and v.is_cuda, "q, k_fp16, v must be CUDA tensors"
    HQ, K = q.shape
    T, HKV, Kk = k_fp16.shape
    Tv, HKVv, V = v.shape
    assert Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [T, HKV, D]"
    G = HQ // HKV
    assert HQ == HKV * G, "HQ must be a multiple of HKV"
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    # Mask buffer kept to compute skip_ratio if requested
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # Fused kernel: grid = (HKV,)
    attn_forward_fused_stage12_fp16[(HKV,)](
        q, k_fp16, v, o,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())
        return o, skip_ratio
    else:
        return o
