import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def convert_to_triton_layout(
    q_rope: torch.Tensor,  # [Bq, Hq, T, Dq]
    k_rope: torch.Tensor,  # [Bk, Hkv, T, Dk]
    v: torch.Tensor,       # [Bv, Hkv, T, Dv]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Bq, Hq, qlen, Dq = q_rope.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert Bq == Bk == Bv
    assert qlen == T == Tv
    assert Hvv == Hkv

    q_triton = q_rope.permute(0, 2, 1, 3).contiguous()
    k_triton = k_rope.permute(0, 2, 1, 3).contiguous()
    v_triton = v.permute(0, 2, 1, 3).contiguous()
    return q_triton, k_triton, v_triton


def pack_k_hi_lo(k: torch.Tensor):
    k = k.contiguous()
    k_hi8 = k.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


@triton.jit
def attn_fwd_prefill_kernel(
    q, k, v, o, scale, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
    delta: tl.constexpr,  # 阈值参数
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_n = i_b
    bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1),
                            (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1),
                            (i_t * BT, 0), (BT, V), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, V], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 第一部分：处理mask块和第一块，计算初始最大值
    # 处理mask块（当前查询块之前的块）
    for i_s in range(0, i_t * BT, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K),
                                (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1),
                                (i_s, 0), (BS, V), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
        b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.exp2(b_m - b_m_new)
        b_m = b_m_new

        b_p = tl.exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    # 第二部分：对于剩余块，先计算最大值判断是否跳过
    o_q = i_t * BT + tl.arange(0, BT)
    
    # 计算当前最大值的阈值
    threshold = b_m - delta
    
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K),
                                (0, i_s), (BK, BS), (0, 1))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T

        # 只加载k来计算最大值
        b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float('-inf'))
        
        # 计算当前块的最大值
        block_max = tl.max(b_s, 1)
        
        # 判断是否跳过这个块
        skip_block = tl.sum(block_max < threshold) == BT  # 所有行的最大值都小于阈值
        
        if not skip_block:
            # 不跳过，加载v并计算
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1),
                                    (i_s, 0), (BS, V), (1, 0))
            b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

            b_m_new = tl.maximum(b_m, block_max)
            b_r = tl.exp2(b_m - b_m_new)
            b_m = b_m_new

            b_p = tl.exp2(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    eps = 1e-12
    b_o = b_o / tl.maximum(b_acc[:, None], eps)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def attn_forward_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_hi8=None, k_lo8=None,
    scale: Optional[float] = None,
    causal: bool = True,
    BT: Optional[int] = None,
    BS: Optional[int] = None,
    BK: Optional[int] = None,
    delta=5.0, return_skip_ratio=False,
    precomputed_threshold=False,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert causal

    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    Bq, T, HQ, Kdim = q.shape
    Bk, Tk, H, Kdim_k = k.shape
    Bv, Tv, H_v, Vdim = v.shape

    assert Bq == Bk == Bv
    assert T == Tk == Tv
    assert Kdim == Kdim_k and H == H_v

    G = HQ // H
    assert G * H == HQ

    if BK is None:
        BK = Kdim
    assert BK == Kdim, "BK must equal head_dim (K)."

    device = q.device
    dtype = q.dtype
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty((Bq, T, HQ, Vdim), device=device, dtype=dtype)

    if scale is None:
        scale = 1.0 / math.sqrt(Kdim)

    # 设置默认的块大小
    if BT is None:
        BT = 128
    if BS is None:
        BS = 128

    grid = lambda meta: (
        triton.cdiv(T, meta['BT']),
        Bq * HQ,
    )

    # 调用修改后的kernel
    if return_skip_ratio:
        # 这里需要修改kernel来返回跳过的块数
        # 由于Triton的限制，我们可能需要使用额外的输出张量
        # 简化版本：只返回输出，跳过比例在Python端计算
        attn_fwd_prefill_kernel[grid](
            q=q, k=k, v=v, o=o, scale=scale, T=T,
            B=Bq, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
            BT=BT, BS=BS, BK=BK, delta=delta,
            num_warps=8, num_stages=3,
        )
        # 返回一个估计的跳过比例（这里需要kernel支持准确计数）
        return o, 0.0  # 暂时返回0，需要完善kernel来准确计数
    
    attn_fwd_prefill_kernel[grid](
        q=q, k=k, v=v, o=o, scale=scale, T=T,
        B=Bq, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
        BT=BT, BS=BS, BK=BK, delta=delta,
        num_warps=8, num_stages=3,
    )
    
    return o