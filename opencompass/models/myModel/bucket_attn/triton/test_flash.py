import os
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_qlen1_lastq_kernel(
    Q, K, V, O,
    stride_q_bh, stride_q_d,
    stride_k_bh, stride_k_n, stride_k_d,
    stride_v_bh, stride_v_n, stride_v_d,
    stride_o_bh, stride_o_d,
    SCALE, N_CTX, D_HEAD,
    # constexpr
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # program over B*H
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    # --- load q (D,) ---
    q_ptrs = Q + pid * stride_q_bh + offs_d * stride_q_d
    q = tl.load(q_ptrs, mask=offs_d < D_HEAD, other=0.0).to(tl.float32)
    # 将 scale 融合进 q，省去内环里的一次乘法
    q = q * SCALE

    # 运行时统计
    m_i = -float("inf")                    # 标量，当前已扫过块中 score 的最大值
    l_i = tl.zeros((), dtype=tl.float32)   # 标量，exp 和
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)  # [D]，加权值和

    # 沿序列分块扫描 K,V
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = (K
                  + pid * stride_k_bh
                  + offs_n[:, None] * stride_k_n
                  + offs_d[None, :] * stride_k_d)
        v_ptrs = (V
                  + pid * stride_v_bh
                  + offs_n[:, None] * stride_v_n
                  + offs_d[None, :] * stride_v_d)

        valid = offs_n < N_CTX

        k = tl.load(k_ptrs,
                    mask=(valid[:, None]) & (offs_d[None, :] < D_HEAD),
                    other=0.0).to(tl.float32)            # [BN, D]
        v = tl.load(v_ptrs,
                    mask=(valid[:, None]) & (offs_d[None, :] < D_HEAD),
                    other=0.0).to(tl.float32)            # [BN, D]

        # 因为 q 在最后一步 => causal 等价于全可见，不需要时间掩码，只需尾部有效性掩码
        # scores: [BN]，用已缩放的 q
        scores = tl.sum(k * q[None, :], axis=1)

        # 仅对越界位置设为 -inf
        scores = tl.where(valid, scores, -float("inf"))

        # 块内 softmax 统计
        m_blk = tl.max(scores, axis=0)             # 标量
        p = tl.exp(scores - m_blk)                 # [BN]
        p = tl.where(valid, p, 0.0)
        l_blk = tl.sum(p, axis=0)                  # 标量
        acc_blk = tl.sum(v * p[:, None], axis=0)   # [D]

        # 归并到运行统计
        m_new = tl.maximum(m_i, m_blk)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_blk - m_new)
        l_i = l_i * alpha + l_blk * beta
        acc = acc * alpha + acc_blk * beta
        m_i = m_new

    # 归一化并写回
    out = acc / l_i
    tl.store(O + pid * stride_o_bh + offs_d * stride_o_d,
             out, mask=offs_d < D_HEAD)


def flash_attn_qlen1_forward_lastq(q, k, v, *, block_n=1024):
    """
    仅前向；仅 qlen=1；仅 causal 且 q 在最后一个位置（因此无需时间掩码）
    q: (B, H, 1, D)
    k: (B, H, N, D)
    v: (B, H, N, D)
    return: (B, H, 1, D)
    """
    assert q.dim() == 4 and q.shape[2] == 1, "只支持 qlen=1（q 形状需为 [B,H,1,D]）"
    B, H, _, D = q.shape
    _, _, N, Dk = k.shape
    assert D == Dk and v.shape == (B, H, N, D)

    # 简化：本 kernel 需要 BLOCK_D >= D；默认上限 128
    MAX_D = 128
    assert D <= MAX_D, f"D={D} 超出本简化实现上限 {MAX_D}；可把 MAX_D 提升到 256/512 并相应调大 num_warps"

    # 选择 BLOCK_D 为不小于 D 的 2 的幂，且不超过 MAX_D
    def next_pow2_ge(x):
        p = 1
        while p < x:
            p <<= 1
        return p
    BLOCK_D = min(MAX_D, next_pow2_ge(D))

    dtype = q.dtype

    # 展平 (B*H, D)/(B*H, N, D)
    q2 = q.reshape(B * H, D).contiguous()
    k2 = k.reshape(B * H, N, D).contiguous()
    v2 = v.reshape(B * H, N, D).contiguous()
    o2 = torch.empty_like(q2)

    # strides
    stride_q_bh, stride_q_d = q2.stride()
    stride_k_bh, stride_k_n, stride_k_d = k2.stride()
    stride_v_bh, stride_v_n, stride_v_d = v2.stride()
    stride_o_bh, stride_o_d = o2.stride()

    scale = 1.0 / math.sqrt(D)

    grid = (B * H,)
    _flash_attn_fwd_qlen1_lastq_kernel[grid](
        q2, k2, v2, o2,
        stride_q_bh, stride_q_d,
        stride_k_bh, stride_k_n, stride_k_d,
        stride_v_bh, stride_v_n, stride_v_d,
        stride_o_bh, stride_o_d,
        scale, N, D,
        BLOCK_N=block_n, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2
    )
    return o2.reshape(B, H, 1, D).to(dtype)


# ---------------------------
# 简单对齐测试
# ---------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 1, 8, 12345, 128
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)

    prof_save_dir = "/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/huffkv-opencompass/opencompass/models/myModel/bucket_attn/prof"

    os.makedirs(prof_save_dir, exist_ok=True)

    with torch.profiler.profile(
            record_shapes=True,  # 记录操作的输入形状
            profile_memory=True,  # 记录内存分配
            activities=[  # 指定分析 CPU 和 GPU（若可用）
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True,
        ) as prof:
        o_triton = flash_attn_qlen1_forward_lastq(q, k, v)
    prof.export_chrome_trace(os.path.join(prof_save_dir, 'flash_attn_triton.json'))

    with torch.profiler.profile(
            record_shapes=True,  # 记录操作的输入形状
            profile_memory=True,  # 记录内存分配
            activities=[  # 指定分析 CPU 和 GPU（若可用）
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            with_stack=True,
        ) as prof:
        # 参考实现（q 在最后一步，causal => 全可见 == 非因果）
        scale = 1.0 / math.sqrt(D)
        att = torch.einsum("bhqd,bhkd->bhqk", q.to(torch.float32), k.to(torch.float32)) * scale
        p = torch.softmax(att, dim=-1)
        o_ref = torch.einsum("bhqk,bhkd->bhqd", p, v.to(torch.float32)).to(q.dtype)
    prof.export_chrome_trace(os.path.join(prof_save_dir, 'torch.json'))

    print("max abs err:", (o_triton - o_ref).abs().max().item())
