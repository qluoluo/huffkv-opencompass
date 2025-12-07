# attn_kernel_v1029_fused.py
import os
import math
from typing import Tuple

import torch
import triton
import triton.language as tl


# ========================
# Layout tools (merged)
# (No changes needed here)
# ========================
def convert_to_triton_layout(
    q_rope_1: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == 1 and qlen == 1 and Tv == T and Hvv == Hkv
    q_triton = q_rope_1[0, :, 0, :].contiguous()
    k_triton_fp16 = k_rope[0].permute(1, 0, 2).contiguous()
    v_triton = v[0].permute(1, 0, 2).contiguous()
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
def attn_forward_fused_kernel(
    q, k_hi8, v, o, m_final, l_final, locks,
    scale, T, delta, th_in,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr,
    T_BS: tl.constexpr = 16,
    USE_EXT_TH: tl.constexpr = False,
):
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    base_hq = pid_hkv * G
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)
    
    # ---- 阈值计算 ----
    if USE_EXT_TH:
        th_ptrs = th_in + (base_hq + rows)
        th_rows = tl.load(th_ptrs, mask=row_mask, other=0.0)
    else:
        NTB = T // BS if T % BS == 0 else T // BS + 1
        offs_t0 = 0 * T_BS + tl.arange(0, T_BS); t_mask0 = offs_t0 < T
        kb_ptrs0 = k_hi8 + (offs_t0[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
        b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
        b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF); m0 = tl.max(b_s0, axis=1)
        
        tb1 = NTB - 1
        offs_t1 = tb1 * T_BS + tl.arange(0, T_BS); t_mask1 = offs_t1 < T
        kb_ptrs1 = k_hi8 + (offs_t1[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
        k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
        b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
        b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF); m1 = tl.max(b_s1, axis=1)
        th_rows = tl.maximum(m0, m1) - delta
    # ---- 遍历小块 ----
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T
        
        # FIX: Replace 'if condition: continue' with 'if not condition: ...'
        # 仅当时间块内有有效 token 时才继续
        if tl.sum(t_mask_sb.to(tl.int32)) > 0:
            kb_ptrs = k_hi8 + (offs_t_sb[None, :] * (HKV * K)) + (pid_hkv * K) + offs_k[:, None]
            k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
            b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
            b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
            
            m_rows_blk = tl.max(b_s_act, axis=1)
            below = (m_rows_blk < th_rows) & row_mask
            
            # 只有当至少有一个 head 没有被 prun，才继续计算
            if tl.sum(below.to(tl.int32)) < G:
                v_offs = tl.arange(0, V)
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                
                for g in tl.static_range(G):
                    # Check per-head pruning condition
                    if m_rows_blk[g] >= th_rows[g]:
                        # Recompute per-head scores, l, and o from the shared b_s block
                        b_s_g = tl.load(b_s + g * SBS + tl.arange(0, SBS), mask=t_mask_sb)
                        m_b = m_rows_blk[g] # This is already computed
                        
                        p_g = tl.exp2(b_s_g - m_b)
                        p_g = tl.where(t_mask_sb, p_g, 0.0)
                        l_b = tl.sum(p_g, axis=0)
                        o_b = tl.dot(p_g[None, :].to(tl.float16), b_v, out_dtype=tl.float32)[0, :]
                        
                        # Atomic update for head g
                        current_hq = base_hq + g
                        lock_ptr = locks + current_hq
                        while tl.atomic_cas(lock_ptr, 0, 1) != 0: pass
                        
                        m_old = tl.load(m_final + current_hq)
                        l_old = tl.load(l_final + current_hq)
                        o_old = tl.load(o + current_hq * V + v_offs)
                        m_new = tl.maximum(m_old, m_b)
                        r_old = tl.exp2(m_old - m_new)
                        r_b = tl.exp2(m_b - m_new)
                        l_new = l_old * r_old + l_b * r_b
                        o_new = o_old * r_old + o_b * r_b
                        
                        tl.store(m_final + current_hq, m_new)
                        tl.store(l_final + current_hq, l_new)
                        tl.store(o + current_hq * V + v_offs, o_new)
                        
                        tl.atomic_xchg(lock_ptr, 0)


# ========================
# Host-side threshold (optional precompute)
# (No changes needed here)
# ========================
def compute_threshold_external(
    q: torch.Tensor, k_fp16: torch.Tensor, scale: float, NTB: int, delta: float,
    HKV: int, HQ: int, T_BS: int = 16,
) -> torch.Tensor:
    assert q.is_cuda and k_fp16.is_cuda; device, dtype = q.device, torch.float32
    HQ_, K = q.shape; T, HKV_, Kk = k_fp16.shape; G = HQ // HKV
    assert HQ_ == HQ and HKV_ == HKV and Kk == K
    RCP_LN2 = 1.4426950408889634
    th = torch.empty((HQ,), device=device, dtype=dtype)
    t0_lo, t0_hi = 0, min(T_BS, T)
    t1_lo, t1_hi = max(0, (NTB - 1) * T_BS), min((NTB - 1) * T_BS + T_BS, T)
    q_f, k_f = q.to(dtype), k_fp16.to(dtype)
    for hkv in range(HKV):
        q_rows = q_f[hkv * G:(hkv + 1) * G]
        k0, k1 = k_f[t0_lo:t0_hi, hkv], k_f[t1_lo:t1_hi, hkv]
        m0_val = (q_rows @ k0.T * (scale * RCP_LN2)).max(dim=1).values if k0.numel() > 0 else torch.full((G,), float("-inf"), device=device, dtype=dtype)
        m1_val = (q_rows @ k1.T * (scale * RCP_LN2)).max(dim=1).values if k1.numel() > 0 else torch.full((G,), float("-inf"), device=device, dtype=dtype)
        th[hkv * G:(hkv + 1) * G] = torch.maximum(m0_val, m1_val) - delta
    return th


# ========================
# Host wrapper
# ========================
def attn_forward(
    q: torch.Tensor, k_hi8: torch.Tensor, k_lo8: torch.Tensor, k_fp16: torch.Tensor, v: torch.Tensor,
    scale: float = None, BS: int = 128, SBS: int | None = None, delta: float = 5.0,
    return_skip_ratio: bool = False, precomputed_threshold: torch.Tensor | None = None,
):
    assert q.is_cuda and k_hi8.is_cuda and v.is_cuda
    HQ, K = q.shape; T, HKV, Kk = k_hi8.shape; Tv, HKVv, V = v.shape
    assert Tv == T and HKVv == HKV and Kk == K, "K/V layouts must be [T, HKV, D]"
    G = HQ // HKV
    if scale is None: scale = 1.0 / math.sqrt(K)
    if SBS is None: SBS = BS
    NTB = triton.cdiv(T, BS)

    o = torch.zeros((HQ, V), device=q.device, dtype=torch.float32)
    m_final = torch.full((HQ,), float("-inf"), device=q.device, dtype=torch.float32)
    l_final = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
    locks = torch.zeros((HQ,), device=q.device, dtype=torch.int32)

    if precomputed_threshold is not None:
        threshold_buf, use_ext_th = precomputed_threshold.contiguous(), True
    else:
        threshold_buf, use_ext_th = torch.empty((HQ,), device=q.device, dtype=torch.float32), False
    
    def next_power_of_2(n):
        if n == 0: return 1
        return 1 << (n - 1).bit_length()

    BM_DOT = max(16, next_power_of_2(G))

    attn_forward_fused_kernel[(HKV, NTB)](
        q, k_hi8, v, o, m_final, l_final, locks,
        scale, T, delta, threshold_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        BM_DOT=BM_DOT,
        USE_EXT_TH=use_ext_th,
    )

    final_out = o / l_final.unsqueeze(-1)
    final_out.masked_fill_(l_final.unsqueeze(-1) == 0, 0.0)

    if return_skip_ratio:
        print("Warning: `return_skip_ratio` is an approximation in the fused kernel.")
        skipped_heads = (l_final == 0).sum()
        skip_ratio = (skipped_heads / HQ).item()
        return final_out.to(q.dtype), skip_ratio
    else:
        return final_out.to(q.dtype)

