import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import argparse
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flash_attn import flash_attn_func


@triton.jit
def _flash_attn_gqa_fwd_kvgrid(
    q_ptr, k_ptr, v_ptr, o_ptr,
    B, SQ, HQ, D, SK, HK, GROUP_SIZE,
    SM_SCALE,
    DTYPE,                           # 0: fp16, 1: bf16, 2: fp32
    CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr,           # tile over query sequence
    BLOCK_N: tl.constexpr,           # tile over key sequence
    BLOCK_D: tl.constexpr,           # tile over head dim (must be >= D)
):
    pid_b = tl.program_id(0)         # batch
    pid_hkv = tl.program_id(1)       # kv head index
    pid_qblk = tl.program_id(2)      # query block index

    # Indices for this block
    q_idx = pid_qblk * BLOCK_Q + tl.arange(0, BLOCK_Q)     # [BLOCK_Q]
    d_idx = tl.arange(0, BLOCK_D)                          # [BLOCK_D]
    q_mask = q_idx < SQ
    d_mask = d_idx < D

    kv_head = pid_hkv

    # Iterate over the GROUP_SIZE query heads mapped to this kv_head
    for gh in range(0, GROUP_SIZE):
        q_head = kv_head * GROUP_SIZE + gh

        # Load Q block [BLOCK_Q, BLOCK_D]
        offs_q_row = ((pid_b * SQ + q_idx) * HQ + q_head) * D
        q_ptrs = q_ptr + offs_q_row[:, None] + d_idx[None, :]
        q = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0)
        q = q.to(tl.float32)

        # Initialize streaming softmax stats
        m_i = tl.full([BLOCK_Q], -1e9, tl.float32)
        l_i = tl.zeros([BLOCK_Q], tl.float32)
        acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)

        # Iterate over key blocks
        for ko in range(0, SK, BLOCK_N):
            k_idx = ko + tl.arange(0, BLOCK_N)      # [BLOCK_N]
            k_mask = k_idx < SK

            # Load K/V tiles [BLOCK_N, BLOCK_D] for this kv_head
            offs_k_row = ((pid_b * SK + k_idx) * HK + kv_head) * D
            k_ptrs = k_ptr + offs_k_row[:, None] + d_idx[None, :]
            v_ptrs = v_ptr + offs_k_row[:, None] + d_idx[None, :]

            k = tl.load(k_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
            k = k.to(tl.float32)
            v = v.to(tl.float32)

            # Attention scores = Q @ K^T scaled: [BLOCK_Q, BLOCK_N]
            scores = tl.dot(q, tl.trans(k)) * SM_SCALE

            # Masking (invalid rows/cols and causal)
            if CAUSAL:
                qpos = q_idx[:, None]
                kpos = k_idx[None, :]
                causal_mask = kpos > qpos
            else:
                causal_mask = tl.full([BLOCK_Q, BLOCK_N], False, tl.int1)
            invalid_mask = (~k_mask[None, :]) | (~q_mask[:, None]) | causal_mask
            scores = tl.where(invalid_mask, -1e9, scores)

            # Streaming softmax update
            p_max = tl.max(scores, axis=1)                 # [BLOCK_Q]
            m_new = tl.maximum(m_i, p_max)                 # [BLOCK_Q]
            p_exp = tl.exp(scores - m_new[:, None])        # [BLOCK_Q, BLOCK_N]
            p_sum = tl.sum(p_exp, axis=1)                  # [BLOCK_Q]
            alpha = tl.exp(m_i - m_new)                    # [BLOCK_Q]
            l_i = l_i * alpha + p_sum                      # [BLOCK_Q]
            acc = acc * alpha[:, None] + tl.dot(p_exp, v)  # [BLOCK_Q, BLOCK_D]
            m_i = m_new

        # Normalize
        out_fp32 = acc / l_i[:, None]

        # Store directly in proper dtype to avoid redefining with different dtypes
        offs_o_row = ((pid_b * SQ + q_idx) * HQ + q_head) * D
        o_ptrs = o_ptr + offs_o_row[:, None] + d_idx[None, :]

        if DTYPE == 0:
            tl.store(o_ptrs, out_fp32.to(tl.float16), mask=q_mask[:, None] & d_mask[None, :])
        elif DTYPE == 1:
            tl.store(o_ptrs, out_fp32.to(tl.bfloat16), mask=q_mask[:, None] & d_mask[None, :])
        else:
            tl.store(o_ptrs, out_fp32, mask=q_mask[:, None] & d_mask[None, :])


def flash_attn_gqa_triton_kvgrid(q, k, v, causal=False, block_q=64, block_n=128):
    """
    Triton Flash-Attention forward with GQA.
    q: [B, Sq, Hq, D] (contiguous)
    k: [B, Sk, Hk, D] (contiguous)
    v: [B, Sk, Hk, D] (contiguous)
    Returns:
      o: [B, Sq, Hq, D]
    Grid is (B, Hk, triton.cdiv(Sq, BLOCK_Q)).
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be 4D [B, S, H, D]"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), "q/k/v must be contiguous"
    B, Sq, Hq, D = q.shape
    Bk, Sk, Hk, Dk = k.shape
    Bv, Sv, Hv, Dv = v.shape
    assert B == Bk == Bv, "Batch size mismatch"
    assert D == Dk == Dv, "Head dim mismatch"
    assert Sk == Sv, "K/V sequence length mismatch"
    assert Hk == Hv, "K/V heads mismatch"
    assert Hq % Hk == 0, "Hq must be a multiple of Hk (GQA)"
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

    BLOCK_Q = block_q
    BLOCK_N = block_n
    grid = (B, Hk, triton.cdiv(Sq, BLOCK_Q))
    sm_scale = 1.0 / math.sqrt(D)

    if q.dtype == torch.float16:
        DTYPE = 0
    elif q.dtype == torch.bfloat16:
        DTYPE = 1
    elif q.dtype == torch.float32:
        DTYPE = 2
    else:
        raise NotImplementedError("Only fp16/bf16/fp32 dtypes are supported")

    _flash_attn_gqa_fwd_kvgrid[grid](
        q, k, v, o,
        B, Sq, Hq, D, Sk, Hk, group_size,
        sm_scale,
        DTYPE,
        int(bool(causal)),
        BLOCK_Q=BLOCK_Q, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return o


def torch_reference_attention_gqa(q, k, v, causal=False):
    """
    PyTorch reference attention emulating GQA by repeating K/V along head dimension.
    q: [B, Sq, Hq, D]
    k: [B, Sk, Hk, D]
    v: [B, Sk, Hk, D]
    Returns:
      out: [B, Sq, Hq, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    assert Hq % Hk == 0
    group_size = Hq // Hk

    # Repeat K/V heads to match Hq
    k_rep = k.repeat_interleave(group_size, dim=2)  # [B, Sk, Hq, D]
    v_rep = v.repeat_interleave(group_size, dim=2)  # [B, Sk, Hq, D]

    # Compute logits: [B, Sq, Hq, Sk]
    logits = torch.matmul(
        q.view(B * Hq, Sq, D),
        k_rep.view(B * Hq, Sk, D).transpose(-1, -2),
    ).view(B, Hq, Sq, Sk).transpose(1, 2)
    logits = logits / math.sqrt(D)

    if causal:
        i = torch.arange(Sq, device=q.device).unsqueeze(1)
        j = torch.arange(Sk, device=q.device).unsqueeze(0)
        causal_mask = (j > i)  # [Sq, Sk]
        logits = logits.masked_fill(causal_mask[None, :, None, :], float("-inf"))

    attn = torch.softmax(logits, dim=-1)  # [B, Sq, Hq, Sk]
    out = torch.matmul(
        attn.view(B * Hq, Sq, Sk),
        v_rep.view(B * Hq, Sk, D),
    ).view(B, Hq, Sq, D).transpose(1, 2)
    return out


def torch_sdpa_attention_gqa(q, k, v, causal=False):
    """
    PyTorch SDPA baseline using scaled_dot_product_attention.
    Implement GQA by repeating K/V along head dimension to match Hq.
    Shapes:
      q: [B, Sq, Hq, D], k/v: [B, Sk, Hk, D]
    Returns:
      out: [B, Sq, Hq, D]
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    B, Sq, Hq, D = q.shape
    _, Sk, Hk, _ = k.shape
    assert Hq % Hk == 0
    group_size = Hq // Hk

    k_rep = k.repeat_interleave(group_size, dim=2)  # [B, Sk, Hq, D]
    v_rep = v.repeat_interleave(group_size, dim=2)  # [B, Sk, Hq, D]

    q_flat = q.permute(0, 2, 1, 3).reshape(B * Hq, Sq, D)
    k_flat = k_rep.permute(0, 2, 1, 3).reshape(B * Hq, Sk, D)
    v_flat = v_rep.permute(0, 2, 1, 3).reshape(B * Hq, Sk, D)

    out_flat = F.scaled_dot_product_attention(
        q_flat, k_flat, v_flat, attn_mask=None, dropout_p=0.0, is_causal=causal
    )
    out = out_flat.reshape(B, Hq, Sq, D).permute(0, 2, 1, 3)
    return out


def flash_attn_library_gqa(q, k, v, causal=False):
    """
    FlashAttention library baseline (FA2). Implements GQA by repeating K/V to match Hq.
    Requirements:
      - dtype: fp16 or bf16
      - CUDA sm80+ usually
      - flash-attn package installed
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
    # FlashAttention func expects fp16/bf16
    assert q.dtype in (torch.float16, torch.bfloat16), "flash-attn requires fp16/bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    # Repeat K/V heads to Hq
    k_rep = k.repeat_interleave(group_size, dim=2).contiguous()  # [B, Sk, Hq, D]
    v_rep = v.repeat_interleave(group_size, dim=2).contiguous()  # [B, Sk, Hq, D]

    # flash_attn_func does internal scaling with 1/sqrt(D) if softmax_scale=None
    out = flash_attn_func(q.contiguous(), k_rep, v_rep, dropout_p=0.0, softmax_scale=None, causal=causal)
    return out


def benchmark(fn, iters=100, warmup=100):
    # Warmup
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
    ms = start_ev.elapsed_time(end_ev) / iters  # average ms
    return ms


def approx_attention_flops(B, Sq, Sk, Hq, D):
    # Approximate FLOPs counting only GEMMs (QK^T and P@V):
    # 4 * Sq * Sk * D per head, times B * Hq
    return 4.0 * B * Hq * Sq * Sk * D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--Sq", type=int, default=1024)
    parser.add_argument("--Sk", type=int, default=None, help="Defaults to Sq if not set")
    parser.add_argument("--Hq", type=int, default=24)
    parser.add_argument("--Hk", type=int, default=8)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--block_q", type=int, default=64)
    parser.add_argument("--block_n", type=int, default=128)
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

    # Create inputs (contiguous)
    q = torch.randn(B, Sq, Hq, D, device=device, dtype=dtype).contiguous()
    k = torch.randn(B, Sk, Hk, D, device=device, dtype=dtype).contiguous()
    v = torch.randn(B, Sk, Hk, D, device=device, dtype=dtype).contiguous()

    # Correctness check vs PyTorch reference (matmul+softmax)
    out_triton = flash_attn_gqa_triton_kvgrid(q, k, v, causal=args.causal, block_q=args.block_q, block_n=args.block_n)
    out_ref = torch_reference_attention_gqa(q, k, v, causal=args.causal)
    max_abs_diff = (out_triton - out_ref).abs().max().item()
    mean_abs_diff = (out_triton - out_ref).abs().mean().item()
    print(f"Correctness vs torch (matmul) - causal={args.causal}: max_abs={max_abs_diff:.6e}, mean_abs={mean_abs_diff:.6e}")
    del out_ref

    # Optional correctness check vs SDPA
    try:
        out_sdpa = torch_sdpa_attention_gqa(q, k, v, causal=args.causal)
        max_abs_diff_sdpa = (out_triton - out_sdpa).abs().max().item()
        mean_abs_diff_sdpa = (out_triton - out_sdpa).abs().mean().item()
        print(f"Correctness vs torch SDPA  - causal={args.causal}: max_abs={max_abs_diff_sdpa:.6e}, mean_abs={mean_abs_diff_sdpa:.6e}")
    except Exception as e:
        print(f"SDPA baseline skipped due to error: {e}")
        out_sdpa = None

    # Optional correctness check vs FlashAttention (library)
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

    # Benchmark closures
    def run_triton():
        return flash_attn_gqa_triton_kvgrid(q, k, v, causal=args.causal, block_q=args.block_q, block_n=args.block_n)

    def run_torch_matmul():
        return torch_reference_attention_gqa(q, k, v, causal=args.causal)

    def run_torch_sdpa():
        return torch_sdpa_attention_gqa(q, k, v, causal=args.causal)

    def run_flash_attn():
        return flash_attn_library_gqa(q, k, v, causal=args.causal)

    # Warm up compilation
    _ = run_triton()
    torch.cuda.synchronize()

    iters = args.iters
    warmup = args.warmup

    ms_triton = benchmark(run_triton, iters=iters, warmup=warmup)
    ms_torch_matmul = benchmark(run_torch_matmul, iters=max(10, iters // 4), warmup=warmup // 2 + 1)

    # SDPA benchmark if available
    try:
        ms_torch_sdpa = benchmark(run_torch_sdpa, iters=iters, warmup=warmup)
    except Exception:
        ms_torch_sdpa = None

    # FlashAttention benchmark if available
    if out_flash is not None:
        try:
            ms_flash_attn = benchmark(run_flash_attn, iters=iters, warmup=warmup)
        except Exception:
            ms_flash_attn = None
    else:
        ms_flash_attn = None

    # Compute approximate TFLOP/s
    flops = approx_attention_flops(B, Sq, Sk, Hq, D)  # FLOPs per forward
    tflops_triton = flops / (ms_triton / 1e3) / 1e12
    tflops_torch_matmul = flops / (ms_torch_matmul / 1e3) / 1e12
    tflops_sdpa = flops / (ms_torch_sdpa / 1e3) / 1e12 if ms_torch_sdpa is not None else None
    tflops_flash = flops / (ms_flash_attn / 1e3) / 1e12 if ms_flash_attn is not None else None

    print("\n==== Speed Benchmark ====")
    print(f"Config: B={B}, Sq={Sq}, Sk={Sk}, Hq={Hq}, Hk={Hk}, D={D}, dtype={args.dtype}, causal={args.causal}")
    print(f"Grid: (B, Hkv, cdiv(Sq, BLOCK_Q)) = ({B}, {Hk}, {triton.cdiv(Sq, args.block_q)})")
    print(f"Triton GQA FlashAttn: {ms_triton:.3f} ms   ~{tflops_triton:.2f} TFLOP/s")
    print(f"Torch (matmul) GQA:   {ms_torch_matmul:.3f} ms   ~{tflops_torch_matmul:.2f} TFLOP/s")
    if ms_torch_sdpa is not None:
        print(f"Torch SDPA GQA:       {ms_torch_sdpa:.3f} ms   ~{tflops_sdpa:.2f} TFLOP/s")
        print(f"Speedup vs SDPA:      x{ms_torch_sdpa / ms_triton:.2f} (SDPA/Triton)")
    if ms_flash_attn is not None:
        print(f"FlashAttn (lib) GQA:  {ms_flash_attn:.3f} ms   ~{tflops_flash:.2f} TFLOP/s")
        print(f"Speedup vs FlashAttn: x{ms_flash_attn / ms_triton:.2f} (FA/Triton)")
    print(f"Speedup vs matmul:    x{ms_torch_matmul / ms_triton:.2f} (Matmul/Triton)")
    print("=========================\n")


if __name__ == "__main__":
    main()
