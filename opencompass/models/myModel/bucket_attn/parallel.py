# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
import time
import warnings
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from einops import reduce

# from fla.ops.utils import prepare_chunk_indices
# from fla.ops.utils.cumsum import chunk_global_cumsum
# from fla.ops.utils.op import exp2, log2
# from fla.utils import autocast_custom_bwd, autocast_custom_fwd, check_shared_mem, contiguous

try:
    from fla_op import exp2, log2
except:
    from .fla_op import exp2, log2


@triton.heuristics({
    'USE_G': lambda args: args['g_cumsum'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def parallel_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    g_cumsum,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # [BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_G:
        p_g = tl.make_block_ptr(g_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_gq = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    else:
        b_gq = None

    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            o_k = i_s + tl.arange(0, BS)
            m_k = o_k < T
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        # [BT, BS]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        # [BT, BS]
        b_p = exp2(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m

    # [BT]
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        # [BS]
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BS]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        if USE_G:
            b_gk = tl.load(g_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_gq[:, None] - b_gk[None, :]

        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float('-inf'))

        # [BT]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        # [BT, BS]
        b_p = exp2(b_s - b_m[:, None])
        # [BT]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [BT, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


def _build_chunk_indices(cu_seqlens: torch.Tensor, BT: int) -> torch.Tensor:
    """
    生成 varlen 情况下的 (n, t_block) 映射表：
    返回 [num_chunks, 2] 的 int32 张量，每行是 (seq_idx, t_block_idx)
    """
    assert cu_seqlens.is_cuda and cu_seqlens.dtype in (torch.int32, torch.int64)
    cu = cu_seqlens.to(torch.int64)
    N = cu.numel() - 1
    chunks = []
    for n in range(N):
        Tn = int(cu[n + 1] - cu[n])
        nblocks = (Tn + BT - 1) // BT
        for tb in range(nblocks):
            chunks.append((n, tb))
    if not chunks:
        return torch.empty((0, 2), dtype=torch.int32, device=cu_seqlens.device)
    return torch.tensor(chunks, dtype=torch.int32, device=cu_seqlens.device)


def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float] = None,   # 典型值 1/sqrt(K)
    causal: bool = True,                # 该 kernel 仅支持 causal=True
    g_cumsum: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,  # varlen 模式传入
    # 下面是 launch 参数（可按需调整/调优）
    BT: int = 128,
    BS: int = 128,
    BK: Optional[int] = None,   # 若为 None 将自动设为 q 的 head_dim
    BV: Optional[int] = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    封装 Triton kernel 的前向函数。

    返回:
      o:  [B, T, HQ, V]（dense）或 [S_total, HQ, V]（varlen）
      lse:[B, T, HQ]（dense）或 [S_total, HQ]（varlen）
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"
    assert q.dtype in (torch.float16, torch.bfloat16), "Support fp16/bf16"
    assert causal, "This kernel currently assumes causal attention."

    # 判别 varlen or dense
    IS_VARLEN = cu_seqlens is not None

    if IS_VARLEN:
        # 约定输入已经 pack 成 [S_total, HQ/H, D] 排布
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3, \
            "Varlen mode expects q:[S, HQ, K], k:[S, H, K], v:[S, H, V]"
        S_total, HQ, Kdim = q.shape
        S_total_k, H, Kdim_k = k.shape
        S_total_v, H_v, Vdim = v.shape
        assert S_total == S_total_k == S_total_v
        assert Kdim == Kdim_k
        assert H == H_v
        G = HQ // H
        assert G * H == HQ, "HQ must be divisible by H (GQA)"
        if BK is None:
            BK = Kdim
        assert BK == Kdim, "BK must equal head_dim (K)."
        BV = Vdim if BV is None else BV

        device = q.device
        dtype = q.dtype

        o = torch.empty((S_total, HQ, Vdim), device=device, dtype=dtype)
        lse = torch.empty((S_total, HQ), device=device, dtype=torch.float32)

        # 生成 varlen 的 chunk_indices，并设置 grid 的第二维为 num_chunks，第三维为 HQ
        chunk_indices = _build_chunk_indices(cu_seqlens, BT)
        num_chunks = chunk_indices.size(0)
        grid = (triton.cdiv(Vdim, BV), num_chunks, HQ)

        # 缩放（自然底 e），kernel 内部会乘以 1/ln2 转为 base-2
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(Kdim)

        # 启动 kernel
        parallel_attn_fwd_kernel[grid](
            q, k, v, o,
            g_cumsum if g_cumsum is not None else q,  # 未使用时传个占位指针（受 USE_G 保护）
            lse,
            sm_scale,
            cu_seqlens,
            chunk_indices,
            0,  # T 对 varlen 分支无用
            # meta
            B=0,                # varlen 分支不会用到 B
            H=H,
            HQ=HQ,
            G=G,
            K=Kdim,
            V=Vdim,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return o, lse

    else:
        # dense：q:[B, T, HQ, K], k:[B, T, H, K], v:[B, T, H, V]
        assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, \
            "Dense mode expects q:[B,T,HQ,K], k:[B,T,H,K], v:[B,T,H,V]"
        B, T, HQ, Kdim = q.shape
        Bk, Tk, H, Kdim_k = k.shape
        Bv, Tv, H_v, Vdim = v.shape
        assert (B, T) == (Bk, Tk) == (Bv, Tv)
        assert Kdim == Kdim_k
        assert H == H_v
        G = HQ // H
        assert G * H == HQ, "HQ must be divisible by H (GQA)"
        if BK is None:
            BK = Kdim
        assert BK == Kdim, "BK must equal head_dim (K)."
        BV = Vdim if BV is None else BV

        device = q.device
        dtype = q.dtype

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o = torch.empty((B, T, HQ, Vdim), device=device, dtype=dtype)
        lse = torch.empty((B, T, HQ), device=device, dtype=torch.float32)

        # grid: (i_v tiles, t-blocks, B*HQ)
        grid = (triton.cdiv(Vdim, BV), triton.cdiv(T, BT), B * HQ)

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(Kdim)

        parallel_attn_fwd_kernel[grid](
            q, k, v, o,
            g_cumsum if g_cumsum is not None else q,  # 未使用时传个占位指针（受 USE_G 保护）
            lse,
            sm_scale,
            None,    # cu_seqlens
            None,    # chunk_indices
            T,
            # meta
            B=B,
            H=H,
            HQ=HQ,
            G=G,
            K=Kdim,
            V=Vdim,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages
        )
        return o, lse


# ========== 基准测试：与 flash_attn 对比 ==========
def _bench_cuda(fn, warmup=10, iters=100):
    # 返回 ms/iter
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def test_and_bench_dense_compare_with_flash_attn(
    B=4, T=2048, H=16, HQ=None, Kdim=64, Vdim=None,
    dtype=torch.float16, causal=True, device="cuda",
    BT=128, BS=128, num_warps=4, num_stages=2
):
    """
    对比数值与速度（dense；GQA=1 以方便与 flash_attn 对齐）。
    """
    HQ = H if HQ is None else HQ
    assert HQ == H, "为了与 flash_attn_func 对齐，这里要求 HQ == H（GQA=1）。"
    Vdim = Kdim if Vdim is None else Vdim
    assert Vdim == Kdim, "为了与 flash_attn_func 对齐，这里要求 V == K。"

    torch.manual_seed(0)
    q = torch.randn(B, T, HQ, Kdim, device=device, dtype=dtype)
    k = torch.randn(B, T, H,  Kdim, device=device, dtype=dtype)
    v = torch.randn(B, T, H,  Vdim, device=device, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(Kdim)

    # 我们的 Triton kernel
    def run_triton():
        o, _ = parallel_attn_fwd(
            q, k, v,
            sm_scale=sm_scale,
            causal=causal,
            g_cumsum=None,
            cu_seqlens=None,
            BT=BT, BS=BS, BK=Kdim, BV=Vdim,
            num_warps=num_warps, num_stages=num_stages
        )
        return o

    o_triton, _ = parallel_attn_fwd(
        q, k, v,
        sm_scale=sm_scale, causal=causal, g_cumsum=None,
        cu_seqlens=None, BT=BT, BS=BS, BK=Kdim, BV=Vdim,
        num_warps=num_warps, num_stages=num_stages
    )

    # flash-attn 对照
    try:
        from flash_attn import flash_attn_func
        q_fa = q  # [B, T, H, K]
        k_fa = k
        v_fa = v
        o_fa = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=0.0, softmax_scale=sm_scale, causal=causal)
    except Exception as e:
        print(f"[WARN] flash_attn 导入或运行失败：{e}")
        print("仅测试 Triton kernel 的速度。")
        o_fa = None

    # 数值对齐
    if o_fa is not None:
        diff = (o_triton - o_fa).float()
        mae = diff.abs().mean().item()
        maxae = diff.abs().max().item()
        rel = (diff.norm() / (o_fa.float().norm() + 1e-8)).item()
        print(f"数值误差: MAE={mae:.3e}, MaxAE={maxae:.3e}, RelL2={rel:.3e}")

    # 速度
    ms_triton = _bench_cuda(run_triton, warmup=10, iters=100)
    print(f"Triton kernel: {ms_triton:.3f} ms/iter (B={B}, T={T}, H={H}, K={Kdim}, dtype={str(dtype).split('.')[-1]})")

    if o_fa is not None:
        def run_fa():
            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=sm_scale, causal=causal)
        ms_fa = _bench_cuda(run_fa, warmup=10, iters=100)
        print(f"flash_attn:     {ms_fa:.3f} ms/iter")
        speedup = ms_fa / ms_triton
        print(f"Speedup (flash_attn / Triton) = {speedup:.3f}x")

    return o_triton, o_fa


if __name__ == "__main__":
    # 基本用法示例（需要 CUDA + Triton，flash-attn 可选）
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = False

    # 1) Dense 对比（HQ=H, V=K）
    o_triton, o_fa = test_and_bench_dense_compare_with_flash_attn(
        B=4, T=2048, H=16, HQ=16, Kdim=64, Vdim=64,
        dtype=torch.float16, causal=True,
        BT=128, BS=128, num_warps=4, num_stages=2
    )

    # 2) 你也可以自行构造 varlen 的输入并调用 parallel_attn_fwd（此处仅给出调用样例）
    # N = 4
    # lengths = torch.tensor([1024, 1536, 2048, 1792], device="cuda", dtype=torch.int32)
    # cu_seqlens = torch.zeros(N + 1, device="cuda", dtype=torch.int32)
    # cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    # S = int(cu_seqlens[-1].item())
    # H, HQ, Kdim, Vdim = 16, 16, 64, 64
    # q = torch.randn(S, HQ, Kdim, device="cuda", dtype=torch.float16)
    # k = torch.randn(S, H,  Kdim, device="cuda", dtype=torch.float16)
    # v = torch.randn(S, H,  Vdim, device="cuda", dtype=torch.float16)
    # o_varlen, lse_varlen = parallel_attn_fwd(q, k, v, sm_scale=1.0/math.sqrt(Kdim), cu_seqlens=cu_seqlens)
    pass