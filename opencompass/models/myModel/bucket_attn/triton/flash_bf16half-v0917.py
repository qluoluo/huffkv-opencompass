# -*- coding: utf-8 -*-
# K 维 pad 到 4 的倍数；仅传 K 的高 8bit（fp16 高字节）到 Triton；
# 内核以 uint32 读，每个 uint32 拆成 4×uint8，经 LUT 解码为 fp16，进行 q·k 估计。

import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attn_fwd_q1_b1_stage2(
    m_buf,       # [HQ, NTB]
    l_buf,       # [HQ, NTB]
    o_buf,       # [HQ, NTB, V], fp32
    o,           # [HQ, V], out dtype = q.dtype
    lse,         # [HQ], fp32
    NTB,         # int
    HQ: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    # 网格: (pid_v, pid_hq)
    pid_v = tl.program_id(0)
    pid_hq = tl.program_id(1)

    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    b_m = tl.full((), float('-inf'), tl.float32)
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([BV], tl.float32)

    for tb in range(0, NTB):
        m_b = tl.load(m_buf + pid_hq * NTB + tb)
        l_b = tl.load(l_buf + pid_hq * NTB + tb)
        has = l_b > 0.0

        o_b = tl.load(
            o_buf + pid_hq * (NTB * V) + tb * V + v_offs,
            mask=(v_mask & has),
            other=0.0
        )

        m_b_eff = tl.where(has, m_b, tl.full((), float('-inf'), tl.float32))
        new_m = tl.maximum(b_m, m_b_eff)
        r_prev = tl.exp2(b_m - new_m)
        r_blk  = tl.where(has, tl.exp2(m_b - new_m), 0.0)

        b_acc = b_acc * r_prev + l_b * r_blk
        b_o   = b_o   * r_prev + o_b * r_blk
        b_m   = new_m

    out_tile = b_o / b_acc
    if pid_v == 0:
        lse_val = b_m + tl.log2(b_acc)
        tl.store(lse + pid_hq, lse_val)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=v_mask)


@triton.jit
def attn_fwd_q1_b1_stage1_khi_lut_u32(
    q,            # [HQ, KP], e.g. fp16, KP 是 4 的倍数
    k_u32,        # [HKV, T, K4], int32/uint32，每个元素打包了 4 个高字节
    v,            # [T, HKV, V], same dtype as q
    lut,          # [256], fp16 解码 LUT
    m_buf,        # [HQ, NTB]
    l_buf,        # [HQ, NTB]
    o_buf,        # [HQ, NTB, V]
    scale,        # float
    T,            # int
    NTB,          # int
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,      # 未 pad 的原始 K
    KP: tl.constexpr,     # padded K（4 的倍数）
    K4: tl.constexpr,     # KP // 4
    V: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr,
    BK4: tl.constexpr,    # 每次处理的 u32 组数（BK=BK4*4）
    BV: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_tb = tl.program_id(2)

    i_hq = pid_hq
    i_h = i_hq // G

    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    s0 = pid_tb * BS
    offs_t = s0 + tl.arange(0, BS)
    t_mask = offs_t < T
    tl.multiple_of(offs_t, 16)

    b_s = tl.zeros([BS], tl.float32)

    for kk4 in range(0, K4, BK4):
        offs_k4 = kk4 + tl.arange(0, BK4)     # [BK4]
        k4_mask = offs_k4 < K4
        tl.multiple_of(offs_k4, 16)

        base_k = offs_k4 * 4                  # [BK4]
        offs_k0 = base_k + 0
        offs_k1 = base_k + 1
        offs_k2 = base_k + 2
        offs_k3 = base_k + 3

        # 对 q 的 4 个 lane 分别加载（对原始 K 做越界置零）
        q0 = tl.load(q + i_hq * KP + offs_k0, mask=(offs_k0 < K), other=0.0)
        q1 = tl.load(q + i_hq * KP + offs_k1, mask=(offs_k1 < K), other=0.0)
        q2 = tl.load(q + i_hq * KP + offs_k2, mask=(offs_k2 < K), other=0.0)
        q3 = tl.load(q + i_hq * KP + offs_k3, mask=(offs_k3 < K), other=0.0)

        # 以 u32 读 K 的高 8bit
        k_ptrs = k_u32 + i_h * T * K4 + (offs_t[None, :] * K4) + offs_k4[:, None]
        u32 = tl.load(
            k_ptrs,
            mask=(k4_mask[:, None] & t_mask[None, :]),
            other=0,
            cache_modifier=".cg",    # 只保留 .cg
        ).to(tl.uint32)              # [BK4, BS]

        # 拆 4 个字节（小端）
        b0 = (u32 >> 0)  & 0xFF
        b1 = (u32 >> 8)  & 0xFF
        b2 = (u32 >> 16) & 0xFF
        b3 = (u32 >> 24) & 0xFF

        # LUT 解码 -> fp16
        k0 = tl.load(lut + b0.to(tl.int32))  # [BK4, BS], fp16
        k1 = tl.load(lut + b1.to(tl.int32))
        k2 = tl.load(lut + b2.to(tl.int32))
        k3 = tl.load(lut + b3.to(tl.int32))

        # 半精乘 + FP32 累加（广播 q?[:, None]）
        p0 = (q0[:, None] * k0).to(tl.float32)
        p1 = (q1[:, None] * k1).to(tl.float32)
        p2 = (q2[:, None] * k2).to(tl.float32)
        p3 = (q3[:, None] * k3).to(tl.float32)
        b_s += tl.sum(p0 + p1 + p2 + p3, axis=0)

    # base-2 softmax
    RCP_LN2 = 1.4426950408889634
    b_s = b_s * scale * RCP_LN2

    active_t = t_mask
    NEG_INF = float('-inf')
    b_s_act = tl.where(active_t, b_s, NEG_INF)
    m_b = tl.max(b_s_act, axis=0)

    b_p = tl.where(active_t, tl.exp2(b_s - m_b), 0.0)
    l_b = tl.sum(b_p, axis=0)

    num_active = tl.sum(active_t.to(tl.int32), axis=0)
    need_v = num_active > 0

    o_b = tl.zeros([BV], tl.float32)
    if need_v:
        v_ptrs = v + (offs_t[:, None] * (HKV * V)) + (i_h * V) + v_offs[None, :]
        b_v = tl.load(
            v_ptrs,
            mask=(active_t[:, None] & v_mask[None, :]),
            other=0.0,
        ).to(tl.float32)
        o_b = tl.sum(b_p[:, None] * b_v, axis=0)

    o_ptrs = o_buf + i_hq * (NTB * V) + pid_tb * V + v_offs
    tl.store(o_ptrs, o_b, mask=v_mask)

    if pid_v == 0:
        tl.store(m_buf + i_hq * NTB + pid_tb, m_b)
        tl.store(l_buf + i_hq * NTB + pid_tb, l_b)


def attn_fwd_q1_b1_splitT_khi_u32(
    q_pad: torch.Tensor,      # [HQ, KP]
    k_hi_u32: torch.Tensor,   # [HKV, T, K4]
    v: torch.Tensor,          # [T, HKV, V]
    lut: torch.Tensor,        # [256]
    scale: float,
    K: int,                   # 原始未 pad 的 K
    KP: int,                  # pad 后 K（4 的倍数）
    BS: int = 256,
    BK4: int = 32,
    BV: int = 64,
    num_warps: int = 4,
    num_stages: int = 4,
):
    assert q_pad.is_cuda and k_hi_u32.is_cuda and v.is_cuda and lut.is_cuda
    HQ, KP_chk = q_pad.shape
    HKV, T, K4 = k_hi_u32.shape
    Tv, HKV2, V = v.shape
    assert KP_chk == KP and Tv == T and HKV2 == HKV
    assert KP % 4 == 0 and KP // 4 == K4

    G = HQ // HKV
    NTB = triton.cdiv(T, BS)

    o = torch.empty((HQ, V), device=q_pad.device, dtype=q_pad.dtype)
    lse = torch.empty((HQ,), device=q_pad.device, dtype=torch.float32)
    m_buf = torch.empty((HQ, NTB), device=q_pad.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q_pad.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q_pad.device, dtype=torch.float32)

    grid1 = (triton.cdiv(V, BV), HQ, NTB)
    attn_fwd_q1_b1_stage1_khi_lut_u32[grid1](
        q_pad, k_hi_u32, v, lut,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        HKV=HKV, HQ=HQ, K=K, KP=KP, K4=K4, V=V, G=G,
        BS=BS, BK4=BK4, BV=BV,
        num_warps=num_warps, num_stages=num_stages,
    )

    grid2 = (triton.cdiv(V, BV), HQ)
    attn_fwd_q1_b1_stage2[grid2](
        m_buf, l_buf, o_buf,
        o, lse, NTB,
        HQ=HQ, V=V, BV=BV,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o, lse


def to_triton_layout(q_rope_1, k_rope, v):
    # q_rope_1: [B, Hq, 1, D], k_rope: [B, Hkv, T, D], v: [B, Hkv, T, Dv]
    # 返回 q:[HQ,K], k:[HKV,T,K], v:[T,HKV,V]
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv == 1
    assert T == Tv and qlen == 1 and Dq == Dk and Hkv == Hvv
    assert Hq % Hkv == 0

    q_triton = q_rope_1[0, :, 0, :].contiguous()            # [HQ, D]
    k_triton = k_rope[0, :, :, :].contiguous()              # [HKV, T, D]
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()  # [T, HKV, Dv]
    return q_triton, k_triton, v_triton


def fp16_bytes_view(x: torch.Tensor):
    assert x.dtype == torch.float16
    st = x.untyped_storage()
    byte_offset = x.storage_offset() * x.element_size()
    sizes   = list(x.size()) + [2]
    strides = [s * x.element_size() for s in x.stride()] + [1]
    return torch.empty(0, dtype=torch.uint8, device=x.device).set_(st, byte_offset, sizes, strides)


def build_e5m2_hi_lut(dtype=torch.float16, device='cuda'):
    base = torch.arange(256, dtype=torch.uint8, device=device)
    # 解码函数：与 kernel 一致（使用 fp16 LUT）
    u = base.to(torch.int32)
    sign = ((u >> 7) & 1).to(torch.float32)
    exp  = ((u >> 2) & 31).to(torch.int32)
    frac = (u & 3).to(torch.float32)
    sign_fac = 1.0 - 2.0 * sign
    mant = 1.0 + frac * 0.25
    e = (exp - 15).to(torch.int32)
    val_norm = torch.ldexp(mant, e)
    val_sub = frac * (2.0 ** -16)
    is_sub = (exp == 0)
    val = torch.where(is_sub, val_sub, val_norm) * sign_fac
    return val.to(dtype).contiguous()


def pad_q_to_4(q: torch.Tensor, Kp: int) -> torch.Tensor:
    HQ, K = q.shape
    if K == Kp:
        return q
    out = torch.zeros((HQ, Kp), device=q.device, dtype=q.dtype)
    out[:, :K] = q
    return out.contiguous()


def pad_and_pack_k_hi_to_u32(k_rope: torch.Tensor, Kp: int):
    # 输入：k_rope: [B, Hkv, T, D] (fp16)
    # 输出：
    #   k_hi_u32: [HKV, T, K4], int32，K4=Kp//4，每个元素打包 4 个高字节（小端）
    #   以及方便复用的 v_triton（[T,HKV,V]）
    assert k_rope.dtype == torch.float16 and k_rope.is_cuda
    B, Hkv, T, D = k_rope.shape
    assert B == 1
    # 取高字节
    kb = fp16_bytes_view(k_rope)                    # [B,Hkv,T,D,2], uint8
    k_hi = kb[..., 1]                               # [B,Hkv,T,D], uint8
    pad = Kp - D
    if pad > 0:
        k_hi = F.pad(k_hi, (0, pad))                # 最后维度右侧 pad 0
    # 打包为 u32（每 4 个高字节 -> 1 个 u32）
    K4 = Kp // 4
    x = k_hi.view(B, Hkv, T, K4, 4).contiguous()    # [1,Hkv,T,K4,4], uint8
    b0 = x[..., 0].to(torch.int32)
    b1 = x[..., 1].to(torch.int32)
    b2 = x[..., 2].to(torch.int32)
    b3 = x[..., 3].to(torch.int32)
    k_u32 = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).contiguous()   # [1,Hkv,T,K4], int32
    k_u32_triton = k_u32[0]                                           # [Hkv, T, K4]
    return k_u32_triton


def lse_reference_base2_gqa_khi(q_triton, k_rope, scale):
    # 用完整的 k（fp16）做 ref，或按需用高 8bit 解码 ref。这里给完整 k 的 ref（与 Flash 对齐）
    qf = q_triton.float()                     # [HQ, K]
    kf = k_rope[0].float()                    # [HKV, T, K]
    HQ, K = qf.shape
    HKV, T, Kk = kf.shape
    assert Kk == K and HQ % HKV == 0
    G = HQ // HKV
    kf_rep = kf.repeat_interleave(G, dim=0) if G != 1 else kf
    scores = torch.einsum('hk, htk -> ht', qf, kf_rep) * scale
    RCP_LN2 = 1.4426950408889634
    return torch.logsumexp(scores, dim=-1) * RCP_LN2


def flash_compute(q_rope_1, k_rope, v):
    from flash_attn import flash_attn_func
    out = flash_attn_func(
        q_rope_1.transpose(1, 2),
        k_rope.transpose(1, 2),
        v.transpose(1, 2),
        causal=False,
    )
    out = out.squeeze(0).squeeze(0)  # [H, Dv]
    return out


def bench_op(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    # 参数建议：BS 越大 V 带宽越多；BK4*4 是 K 的 tile 大小；可尝试 BK4=32/48，num_warps=4/8
    BS, BK4, BV = 256, 32, 64
    iters, warmup = 50, 10

    # LUT 准备（fp16）
    lut = build_e5m2_hi_lut(dtype=torch.float16, device='cuda')

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # 只取最后一个查询位置
        q_rope_1 = q_rope[:, :, -1:, :]  # [B, Hq, 1, D]

        B, Hq, qlen, D = q_rope_1.shape
        Bk, Hkv, T, Dk = k_rope.shape
        Bv, Hv, Tv, Dv = v.shape
        assert B == 1 and qlen == 1 and Hkv == Hv and D == Dk and T == Tv
        assert Hq % Hkv == 0

        print(f"{T=} {Hq=} {Hkv=} {D=} {Dv=}")

        # Triton 布局（q,v）
        q_triton, _, v_triton = to_triton_layout(q_rope_1, k_rope, v)  # q:[HQ,D], v:[T,HKV,V]

        # K 维 pad 到 4 的倍数
        KP = (D + 3) // 4 * 4
        K4 = KP // 4

        # q pad
        q_pad = pad_q_to_4(q_triton, KP)

        # k 高 8bit pad & 打包为 u32
        k_hi_u32 = pad_and_pack_k_hi_to_u32(k_rope, KP)  # [Hkv, T, K4], int32

        scale = 1.0 / math.sqrt(D)

        # 仅 K 高 8bit（u32 读）+ LUT 的 Triton 实现
        o_triton_khi, lse_triton_khi = attn_fwd_q1_b1_splitT_khi_u32(
            q_pad, k_hi_u32, v_triton, lut,
            scale=scale, BS=BS, BK4=BK4, BV=BV,
            K=D, KP=KP,
            num_warps=4, num_stages=4,
        )

        # 参考（完整精度 Flash）
        o_flash = flash_compute(q_rope_1, k_rope, v)

        # 数值对比
        max_abs = (o_triton_khi.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton_khi.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton_khi.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"[K_hi u32 + LUT] Value diff vs Flash: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton_khi():
            o, _ = attn_fwd_q1_b1_splitT_khi_u32(
                q_pad, k_hi_u32, v_triton, lut,
                scale=scale, BS=BS, BK4=BK4, BV=BV,
                K=D, KP=KP,
                num_warps=4, num_stages=4,
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton_khi = bench_op(run_triton_khi, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton(K_hi u32 + LUT)={ms_triton_khi:.3f} ms, Flash(full)={ms_flash:.3f} ms, ratio={ms_triton_khi/ms_flash:.2f}x")