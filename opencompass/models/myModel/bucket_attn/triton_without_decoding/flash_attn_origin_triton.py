import math
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    TMP,
    L,
    M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    Z,
    H,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_vh + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * N_CTX + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk +=  tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        tl.store(t_ptrs, acc_scale)
        acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs + start_n * stride_vk)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_i)
    tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)


def flash_attn_triton_causal(q, k, v, block_m=64, block_n=64):
    """
    Triton 实现的因果（causal）FlashAttention 前向。
    输入:
      - q, k, v: shape = [B, H, N, D], dtype = fp16/bf16, CUDA tensor，layout 可任意（函数会整理）
      - block_m, block_n: Triton tile 大小，需整除 N
    返回:
      - out: shape = [B, H, N, D]，dtype 与输入相同
    约束:
      - D 必须等于编译期常量 BLOCK_DMODEL（本函数中 = D）
      - N 必须是 block_m 的整数倍
      - q/k/v 会被 view/contiguous 为相同 stride 的 [B*H, N, D]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda, "q/k/v 必须在 CUDA 上"
    assert q.dtype in (torch.float16, torch.bfloat16), "仅支持 fp16/bf16"
    assert q.shape == k.shape == v.shape and len(q.shape) == 4, "期望 q/k/v 形状 [B, H, N, D]"
    B, H, N, D = q.shape
    assert N % block_m == 0, f"N={N} 需被 block_m={block_m} 整除"
    assert D % 16 == 0, "head_dim 需是 16 的倍数以配合张量核心"
    sm_scale = 1.0 / math.sqrt(D)

    # 将三者整理为相同布局与 stride：[B*H, N, D] 且连续
    q_ = q.contiguous().view(B * H, N, D)
    k_ = k.contiguous().view(B * H, N, D)
    v_ = v.contiguous().view(B * H, N, D)

    # 目标输出与辅助张量（acc 在 fp32）
    out = torch.empty((B * H, N, D), dtype=torch.float32, device=q.device)
    L = torch.empty((B * H, N), dtype=torch.float32, device=q.device)
    M = torch.empty((B * H, N), dtype=torch.float32, device=q.device)
    TMP = torch.empty_like(L)

    # 计算 stride（以“元素”为单位）
    stride_qz = 0
    stride_qh = q_.stride(0)
    stride_qm = q_.stride(1)
    stride_qk = q_.stride(2)

    stride_kz = 0
    stride_kh = k_.stride(0)
    stride_kn = k_.stride(1)
    stride_kk = k_.stride(2)

    stride_vz = 0
    stride_vh = v_.stride(0)
    stride_vk = v_.stride(1)
    stride_vn = v_.stride(2)

    stride_oz = 0
    stride_oh = out.stride(0)
    stride_om = out.stride(1)
    stride_on = out.stride(2)

    # 启动配置
    grid = (triton.cdiv(N, block_m), B * H)

    _fwd_kernel[grid](
        q_, k_, v_, sm_scale, TMP, L, M, out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        B, H, N,
        BLOCK_M=block_m, BLOCK_DMODEL=D, BLOCK_N=block_n,
        # 你也可以手动指定 num_warps / num_stages
        # num_warps=4, num_stages=2
    )

    return out.view(B, H, N, D).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = "cuda"

    # 随机示例（确保 N 能被 block_m 整除，D 与 BLOCK_DMODEL 匹配）
    B, H, N, D = 2, 4, 128, 64
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    # Triton 结果
    out_triton = flash_attn_triton_causal(q, k, v, block_m=64, block_n=64)  # [B, H, N, D]
    print(f"{out_triton.shape=}")

    from flash_attn import flash_attn_func
    # 参考实现的输入需要 [B, N, H, D]
    q_bnhd = q.permute(0, 2, 1, 3).contiguous()
    k_bnhd = k.permute(0, 2, 1, 3).contiguous()
    v_bnhd = v.permute(0, 2, 1, 3).contiguous()

    out_ref_bnhd = flash_attn_func(q_bnhd, k_bnhd, v_bnhd, causal=True)
    # 转回 [B, H, N, D] 以与 Triton 输出对齐
    out_ref = out_ref_bnhd.permute(0, 2, 1, 3).contiguous()

    torch.cuda.synchronize()
    max_abs_err = (out_ref - out_triton).abs().max().item()
    mean_abs_err = (out_ref - out_triton).abs().mean().item()
    print(f"max_abs_err: {max_abs_err:.4e}, mean_abs_err: {mean_abs_err:.4e}")