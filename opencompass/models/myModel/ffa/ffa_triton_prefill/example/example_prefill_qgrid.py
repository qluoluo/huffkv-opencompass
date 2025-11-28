import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import argparse
import torch
import triton
import triton.language as tl

from flash_attn import flash_attn_func

h100_autotune_configs = [
    # D <= 64 时通常较优
    triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 64},  num_warps=4, num_stages=2),
    # D <= 128 的通用强配置
    triton.Config({'BLOCK_Q': 64,  'BLOCK_N': 128}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_Q': 64,  'BLOCK_N': 256}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),  # 对于中等长度序列也常好
    # 长序列（Sk 很大）偏好的更大 N 块，或寄存器压力较高时的退让
    triton.Config({'BLOCK_Q': 32,  'BLOCK_N': 256}, num_warps=4, num_stages=4),
    triton.Config({'BLOCK_Q': 32,  'BLOCK_N': 128}, num_warps=4, num_stages=3),
    # 兜底/探索型组合（有时在特定 shape 上更佳）
    triton.Config({'BLOCK_Q': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_Q': 16,  'BLOCK_N': 256},  num_warps=4, num_stages=2),
    triton.Config({'BLOCK_Q': 16,  'BLOCK_N': 512},  num_warps=4, num_stages=2),
]

@triton.autotune(
    configs=h100_autotune_configs,
    key=['SQ', 'SK', 'D', 'CAUSAL', 'DTYPE', 'BLOCK_D'],
    prune_configs_by={
        'type': 'early_config_prune',
        'fn': lambda meta: (meta['BLOCK_Q'] <= 128) and (meta['BLOCK_N'] <= 256)
    }
)
@triton.jit
def _flash_attn_gqa_fwd_qgrid(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, SQ, HQ, D, SK, HK, GROUP_SIZE,
    SM_SCALE,
    DTYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_qblk = tl.program_id(2)

    q_idx = pid_qblk * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_idx = tl.arange(0, BLOCK_D)
    q_mask = q_idx < SQ
    d_mask = d_idx < D

    q_head = pid_hq
    kv_head = q_head // GROUP_SIZE

    offs_q_row = ((pid_b * SQ + q_idx) * HQ + q_head) * D
    q_ptrs = q_ptr + offs_q_row[:, None] + d_idx[None, :]
    q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
    q = q.to(tl.float32)

    m_i = tl.full([BLOCK_Q], -1e9, tl.float32)
    l_i = tl.zeros([BLOCK_Q], tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

    for ko in range(0, SK, BLOCK_N):
        k_idx = ko + tl.arange(0, BLOCK_N)
        k_mask = k_idx < SK

        offs_k_row = ((pid_b * SK + k_idx) * HK + kv_head) * D
        k_ptrs = k_ptr + offs_k_row[:, None] + d_idx[None, :]
        v_ptrs = v_ptr + offs_k_row[:, None] + d_idx[None, :]

        k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
        k = k.to(tl.float32)
        v = v.to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * SM_SCALE

        if CAUSAL:
            qpos = q_idx[:, None]
            kpos = k_idx[None, :]
            causal_mask = kpos > qpos
        else:
            causal_mask = tl.full([BLOCK_Q, BLOCK_N], False, tl.int1)
        invalid_mask = (~k_mask[None, :]) | (~q_mask[:, None]) | causal_mask
        scores = tl.where(invalid_mask, -1e9, scores)

        p_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, p_max)
        p_exp = tl.exp(scores - m_new[:, None])
        p_sum = tl.sum(p_exp, axis=1)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + p_sum
        acc = acc * alpha[:, None] + tl.dot(p_exp, v)
        m_i = m_new

    out_fp32 = acc / l_i[:, None]

    offs_o_row = ((pid_b * SQ + q_idx) * HQ + q_head) * D
    o_ptrs = o_ptr + offs_o_row[:, None] + d_idx[None, :]

    if DTYPE == 0:
        tl.store(o_ptrs, out_fp32.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])
    elif DTYPE == 1:
        tl.store(o_ptrs, out_fp32.to(tl.bfloat16), mask=q_mask[:, None] & d_mask[None, :])
    else:
        tl.store(o_ptrs, out_fp32, mask=q_mask[:, None] & d_mask[None, :])


def flash_attn_gqa_triton_qgrid(q, k, v, causal=False, block_q=64, block_n=128, autotune=True):
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    group_size = Hq // Hk
    o = torch.empty((B, Sq, Hq, D), dtype=q.dtype, device=q.device)
    if D <= 32:
        BLOCK_D = 32
    elif D <= 64:
        BLOCK_D = 64
    elif D <= 128:
        BLOCK_D = 128
    elif D <= 256:
        BLOCK_D = 256
    else:
        raise NotImplementedError("D > 256 not supported in this demo")
    sm_scale = 1.0 / math.sqrt(D)
    if q.dtype == torch.float16:
        DTYPE = 0
    elif q.dtype == torch.bfloat16:
        DTYPE = 1
    elif q.dtype == torch.float32:
        DTYPE = 2
    else:
        raise NotImplementedError("Only fp16/bf16/fp32 dtypes are supported")
    if autotune:
        grid = lambda META: (B, Hq, triton.cdiv(Sq, META['BLOCK_Q']))
        _flash_attn_gqa_fwd_qgrid[grid](
            q, k, v, o,
            B, Sq, Hq, D, Sk, Hk, group_size,
            sm_scale,
            DTYPE,
            CAUSAL=bool(causal),
            BLOCK_D=BLOCK_D,
        )
    else:
        grid = (B, Hq, triton.cdiv(Sq, block_q))
        _flash_attn_gqa_fwd_qgrid[grid](
            q, k, v, o,
            B, Sq, Hq, D, Sk, Hk, group_size,
            sm_scale,
            DTYPE,
            CAUSAL=bool(causal),
            BLOCK_Q=block_q, BLOCK_N=block_n, BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=2,
        )
    return o


def flash_attn_library_gqa(q, k, v, causal=False):
    """
    FlashAttention library baseline (FA2). Implements GQA by repeating K/V to match Hq.
    Shapes:
      q: [B, Sq, Hq, D], k: [B, Sk, Hk, D], v: [B, Sk, Hk, D]
    Returns:
      out: [B, Sq, Hq, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, Dk = k.shape
    assert D == Dk and D == v.shape[-1]
    assert Hq % Hk == 0
    group_size = Hq // Hk
    assert q.dtype in (torch.float16, torch.bfloat16), "flash-attn requires fp16/bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    k_rep = k.repeat_interleave(group_size, dim=2).contiguous()  # [B, Sk, Hq, D]
    v_rep = v.repeat_interleave(group_size, dim=2).contiguous()  # [B, Sk, Hq, D]

    out = flash_attn_func(q.contiguous(), k_rep, v_rep, dropout_p=0.0, softmax_scale=None, causal=causal)
    return out


def benchmark(fn, iters=100, warmup=100):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        _ = fn()
    end_ev.record()
    torch.cuda.synchronize()
    ms = start_ev.elapsed_time(end_ev) / iters
    return ms


def approx_attention_flops(B, Sq, Sk, Hq, D):
    return 4.0 * B * Hq * Sq * Sk * D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Sq", type=int, default=1024)
    parser.add_argument("--Sk", type=int, default=None, help="Defaults to Sq if not set")
    parser.add_argument("--Hq", type=int, default=24)
    parser.add_argument("--Hk", type=int, default=8)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    device = "cuda"
    assert torch.cuda.is_available(), "CUDA is required"

    B = args.B
    Sq = args.Sq
    Sk = args.Sk if args.Sk is not None else args.Sq
    Hq = args.Hq
    Hk = args.Hk
    D = args.D
    assert Hq % Hk == 0, "Hq must be a multiple of Hk for GQA"

    if args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    torch.manual_seed(0)

    q = torch.randn(B, Sq, Hq, D, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, Sk, Hk, D, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, Sk, Hk, D, device=device, dtype=dtype).contiguous()

    # 仅与 FlashAttention 进行正确性比较
    out_triton = flash_attn_gqa_triton_qgrid(q, k, v, causal=args.causal)
    out_flash = None
    if dtype in (torch.float16, torch.bfloat16):
        try:
            out_flash = flash_attn_library_gqa(q, k, v, causal=args.causal)
            max_abs_diff_flash = (out_triton - out_flash).abs().max().item()
            mean_abs_diff_flash = (out_triton - out_flash).abs().mean().item()
            print(f"Correctness vs FlashAttn  - causal={args.causal}: max_abs={max_abs_diff_flash:.6e}, mean_abs={mean_abs_diff_flash:.6e}")
        except Exception as e:
            print(f"FlashAttention baseline skipped due to error: {e}")
            out_flash = None
    else:
        print("FlashAttention baseline skipped: dtype must be fp16/bf16.")

    # 预热编译与自动调优
    print("\nWarming up and running autotuner (this may take a moment)...")
    _ = flash_attn_gqa_triton_qgrid(q, k, v, causal=args.causal)
    torch.cuda.synchronize()
    print("Autotuner finished.")

    iters = args.iters
    warmup = args.warmup

    def run_triton():
        return flash_attn_gqa_triton_qgrid(q, k, v, causal=args.causal)

    def run_flash_attn():
        return flash_attn_library_gqa(q, k, v, causal=args.causal)

    ms_triton = benchmark(run_triton, iters=iters, warmup=warmup)

    if out_flash is not None:
        try:
            ms_flash_attn = benchmark(run_flash_attn, iters=iters, warmup=warmup)
        except Exception:
            ms_flash_attn = None
    else:
        ms_flash_attn = None

    flops = approx_attention_flops(B, Sq, Sk, Hq, D)
    tflops_triton = flops / (ms_triton / 1e3) / 1e12
    tflops_flash = flops / (ms_flash_attn / 1e3) / 1e12 if ms_flash_attn is not None else None

    print("\n==== Speed Benchmark ====")
    print(f"Config: B={B}, Sq={Sq}, Sk={Sk}, Hq={Hq}, Hk={Hk}, D={D}, dtype={args.dtype}, causal={args.causal}")
    print(f"Triton autotuner best config: {_flash_attn_gqa_fwd_qgrid.best_config}")
    print(f"Triton GQA (Autotuned):       {ms_triton:.3f} ms   ~{tflops_triton:.2f} TFLOP/s")
    if ms_flash_attn is not None:
        print(f"FlashAttn (lib) GQA:          {ms_flash_attn:.3f} ms   ~{tflops_flash:.2f} TFLOP/s")
        print(f"Speedup vs FlashAttn:         x{ms_flash_attn / ms_triton:.2f} (FA/Triton)")
    else:
        print("FlashAttention benchmark skipped.")
    print("=========================\n")


if __name__ == "__main__":
    main()
