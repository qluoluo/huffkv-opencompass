import math
import os
from tqdm import tqdm

import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Bias,
    Out,
    Lse,
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    softmax_scale,
    prune_threshold,  # 新增：剪枝阈值偏移
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,  # 仅用于编译期循环上界
    CACHE_KEY_SEQLEN_K,  # 仅用于编译期循环上界
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ENABLE_PRUNE: tl.constexpr,  # 新增：是否启用剪枝（编译期常量，避免分支开销）
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Initialize pointers to Q, K, V
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )

    # initialize pointer to m and l
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # load q
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )

    # 可见的 K 上界（右开区间）
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)

    # 计算阈值（仅在启用剪枝时）
    if ENABLE_PRUNE:
        def load_k_block_raw_qk(start_n_block):
            # 加载 k
            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k_blk = tl.load(k_ptrs + start_n_block * stride_kn)
                else:
                    k_blk = tl.load(
                        k_ptrs + start_n_block * stride_kn, mask=offs_d[None, :] < headdim, other=0.0
                    )
            else:
                if EVEN_HEADDIM:
                    k_blk = tl.load(
                        k_ptrs + start_n_block * stride_kn,
                        mask=(start_n_block + offs_n)[:, None] < seqlen_k,
                        other=0.0,
                    )
                else:
                    k_blk = tl.load(
                        k_ptrs + start_n_block * stride_kn,
                        mask=((start_n_block + offs_n)[:, None] < seqlen_k)
                        & (offs_d[None, :] < headdim),
                        other=0.0,
                    )
            # 计算原始 qk（未加 bias/未缩放）
            qk_raw_blk = tl.dot(q, tl.trans(k_blk)).to(tl.float32)
            # 越界/因果掩码（与主循环一致）
            if not EVEN_N:
                qk_raw_blk += tl.where((start_n_block + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
            if IS_CAUSAL:
                qk_raw_blk += tl.where(
                    offs_m[:, None] >= (start_n_block + offs_n)[None, :], 0, float("-inf")
                )
            return qk_raw_blk

        # 第一个 block: start=0
        start_n_first = 0
        qk_raw_first = load_k_block_raw_qk(start_n_first)
        max_first = tl.max(qk_raw_first, 1)  # [BLOCK_M]

        # 最后一个可见 block: start = ((end_n - 1) // BLOCK_N) * BLOCK_N
        last_start = ((end_n - 1) // BLOCK_N) * BLOCK_N
        last_start = tl.max(0, last_start)  # 防止负数
        qk_raw_last = load_k_block_raw_qk(last_start)
        max_last = tl.max(qk_raw_last, 1)

        # 行阈值
        threshold = tl.maximum(max_first, max_last) - prune_threshold  # [BLOCK_M], float32
    else:
        threshold = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    # 主循环：使用编译期上界 CACHE_KEY_SEQLEN_K，运行时按需 break
    for start_n in range(0, CACHE_KEY_SEQLEN_K, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # 运行时越界保护
        if start_n >= end_n:
            break

        # 先计算“未加 bias/未缩放”的 qk_raw，用于与 threshold 比较
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk_raw = tl.dot(q, tl.trans(k)).to(tl.float32)
        if not EVEN_N:
            qk_raw += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk_raw += tl.where(
                offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
            )

        # 提前退出判断（仅开启剪枝时有效）
        if ENABLE_PRUNE:
            block_row_max = tl.max(qk_raw, 1)  # [BLOCK_M]
            should_break = tl.all(block_row_max < threshold)
            if should_break:
                break

        # 走正常 softmax 累计路径
        # 统一缩放与 bias 处理
        qk_scaled = qk_raw
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk_scaled = qk_scaled * softmax_scale + bias
        else:
            qk_scaled = qk_scaled * softmax_scale

        # 归一化 softmax 块
        m_ij = tl.maximum(tl.max(qk_scaled, 1), lse_i)
        p = tl.exp(qk_scaled - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # 缩放累加器
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale)
        acc_o_scale = tl.load(t_ptrs)
        acc_o = acc_o * acc_o_scale[:, None]

        # 读取 V 并累加
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # 更新统计量
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # 写回输出
    o_scale = tl.exp(m_i - lse_i)
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]

    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # write back lse
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)

    # write back Out
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )


def _flash_attn_forward(
    q, k, v, bias=None, causal=False, softmax_scale=None, prune_threshold=0.0, enable_prune=False
):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only supports head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k) or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    else:
        # 传入哑指针避免 Triton 对 None 指针报错（不会被访问，因为 BIAS_TYPE='none'）
        bias = q

    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128.0) * 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    BLOCK = 128
    num_warps = 4 if d <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        bias,
        o,
        lse,
        tmp,
        softmax_scale,
        prune_threshold,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,     # CACHE_KEY_SEQLEN_Q
        seqlen_k // 32,     # CACHE_KEY_SEQLEN_K
        bias_type,
        causal,
        enable_prune,       # ENABLE_PRUNE: tl.constexpr
        BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def flash_official_compute(q_rope, k_rope, v, causal=False):
    from flash_attn import flash_attn_func
    o = flash_attn_func(
        q_rope,
        k_rope,
        v,
        causal=causal,
    )
    return o


def bench_op(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms


if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16  # 建议 fp16/bf16 才能触发 Flash

    # 计时参数
    iters = 100
    warmup = 100

    # 阈值设置（可按需调整或做成自适应）
    prune_threshold = 10.0
    enable_prune = False

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # q_rope = q_rope[:, :, -1:, :]  # 如需 qlen=1，可解开

        B, Hq, qlen, D = q_rope.shape
        Bk, Hkv, T, Dk = k_rope.shape
        Bv, Hv, Tv, Dv = v.shape
        assert B == 1, "该 demo 仅支持 batch=1"
        # assert qlen == 1, "该 demo 仅支持 qlen=1"
        assert Hkv == Hv, "k/v heads 必须一致"
        assert D == Dk, "q/k head_dim 不一致"
        assert T == Tv
        assert Hq % Hkv == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"
        print(f"{T=} {Hq=} {Hkv=} {D=} {Dv=}")

        if Hq != Hkv:
            from transformers.models.llama.modeling_llama import repeat_kv
            k_rope = repeat_kv(k_rope, Hq // Hkv)
            v = repeat_kv(v, Hq // Hkv)

        # 转换为 [B, T, H, D]
        q_rope = q_rope.transpose(1, 2)
        k_rope = k_rope.transpose(1, 2)
        v = v.transpose(1, 2)

        # Triton 实现
        o_triton, _, _ = _flash_attn_forward(
            q_rope, k_rope, v,
            causal=True,
            prune_threshold=prune_threshold,
            enable_prune=enable_prune,
        )

        # 官方实现（对比）
        o_flash = flash_official_compute(q_rope, k_rope, v, causal=True)  # [B, T, H, D]

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton():
            o, _, _ = _flash_attn_forward(
                q_rope, k_rope, v,
                causal=True,
                prune_threshold=prune_threshold,
                enable_prune=enable_prune,
            )
            return o

        def run_flash():
            return flash_official_compute(q_rope, k_rope, v, causal=True)

        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        # break