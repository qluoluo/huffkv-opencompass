# 计算逻辑有缺陷，该脚本只计算了q重要且k重要的位置对应的attn结果，但是可以当做之后的对比baseline

import math
from typing import Optional, Tuple, List

import torch
import triton
import triton.language as tl


def convert_to_triton_layout(
    q_rope: torch.Tensor,  # [Bq, Hq, T, Dq]
    k_rope: torch.Tensor,  # [Bk, Hkv, T, Dk]
    v: torch.Tensor,       # [Bv, Hkv, T, Dv]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将输入从 [B, H, T, D] 转换为 Triton 友好的 [B, T, H, D] 布局。
    """
    Bq, Hq, qlen, Dq = q_rope.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert Bq == Bk == Bv
    assert qlen == T == Tv
    assert Hvv == Hkv

    q_triton = q_rope.permute(0, 2, 1, 3).contiguous()  # [B, T, HQ, Dq]
    k_triton = k_rope.permute(0, 2, 1, 3).contiguous()  # [B, T, H, Dk]
    v_triton = v.permute(0, 2, 1, 3).contiguous()       # [B, T, H, Dv]
    return q_triton, k_triton, v_triton


@triton.jit
def attn_fwd_prefill_kernel(
    q, k, v, o, selected_mask, scale, threshold, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr,
):
    """
    只计算 Q 的最后一个 block 的注意力输出，并统计每个 KV block 的最大分数；
    以 first_block_max - threshold 为阈值写入 selected_mask（形状 [B*HQ, NB]，0/1）。
    """
    i_bh = tl.program_id(0)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G  # KV head 映射

    RCP_LN2: tl.constexpr = 1.4426950216

    # 仅处理 Q 的最后一个 block
    q_start = tl.maximum(0, T - BT)

    # Q 的 block ptr
    p_q = tl.make_block_ptr(
        base=q + ((i_b * T * HQ + i_hq) * K),
        shape=(T, K),
        strides=(HQ * K, 1),
        offsets=(q_start, 0),
        block_shape=(BT, BK),
        order=(1, 0),
    )

    # O 的 block ptr
    p_o = tl.make_block_ptr(
        base=o + (i_b * T * HQ + i_hq) * V,
        shape=(T, V),
        strides=(HQ * V, 1),
        offsets=(q_start, 0),
        block_shape=(BT, V),
        order=(1, 0),
    )

    # 读取 Q 的最后一个 block
    b_q = tl.load(p_q, boundary_check=(0, 1))

    # softmax 累积缓存
    b_o = tl.zeros([BT, V], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 记录第一个 KV block 的最大分数
    first_block_max = -float('inf')

    # 遍历所有 K/V 的时间块
    for i_s in range(0, T, BT):
        p_k = tl.make_block_ptr(
            base=k + ((i_b * T * H + i_h) * K),
            shape=(K, T),
            strides=(1, H * K),
            offsets=(0, i_s),
            block_shape=(BK, BT),
            order=(0, 1),
        )

        p_v = tl.make_block_ptr(
            base=v + ((i_b * T * H + i_h) * V),
            shape=(T, V),
            strides=(H * V, 1),
            offsets=(i_s, 0),
            block_shape=(BT, V),
            order=(1, 0),
        )

        b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
        b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

        # QK^T 分数
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

        # Mask
        o_q = q_start + tl.arange(0, BT)
        o_k = i_s + tl.arange(0, BT)
        m_q = o_q < T
        m_k = o_k < T
        causal = (o_q[:, None] >= o_k[None, :])
        mask = causal & m_q[:, None] & m_k[None, :]
        b_s = tl.where(mask, b_s, float('-inf'))

        # Update stats
        b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
        b_r = tl.exp2(b_m - b_m_new)
        b_m = b_m_new

        b_p = tl.exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.float32), b_v.to(tl.float32))

        # 统计 block max
        row_max = tl.max(b_s, axis=1)
        block_max = tl.max(row_max, axis=0)

        if i_s == 0:
            first_block_max = block_max

        # 阈值比较
        thr_cmp_val = first_block_max - threshold
        cond = block_max >= thr_cmp_val

        # 写选中掩码
        i_blk = i_s // BT
        write_idx = i_bh * NB + i_blk
        tl.store(selected_mask + write_idx, (1 if cond else 0), mask=None)

    # 归一化并写回
    eps = 1e-12
    b_den = tl.maximum(b_acc, eps)
    b_o = b_o / b_den[:, None]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def attn_forward_prefill_lastQblock(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    threshold: float = 0.0,
    causal: bool = True,
    BT: Optional[int] = None,
    BK: Optional[int] = None,
):
    """
    计算 Q 的最后一个 block 的注意力输出，同时统计各 KV block 的注意力分数最大值。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    Bq, T, HQ, Kdim = q.shape
    Bk, Tk, H, Kdim_k = k.shape
    Bv, Tv, H_v, Vdim = v.shape

    assert Bq == Bk == Bv
    assert T == Tk == Tv
    assert Kdim == Kdim_k
    assert H == H_v

    G = HQ // H
    if BK is None:
        BK = Kdim
    
    device = q.device
    dtype = q.dtype
    o_full = torch.empty((Bq, T, HQ, Vdim), device=device, dtype=dtype)

    if scale is None:
        scale = 1.0 / math.sqrt(Kdim)

    if BT is None:
        BT = 128

    NB = (T + BT - 1) // BT
    NQ = NB

    selected_mask_dev = torch.empty((Bq * HQ, NB), device=device, dtype=torch.int32)

    grid = (Bq * HQ,)
    attn_fwd_prefill_kernel[grid](
        q=q, k=k, v=v, o=o_full,
        selected_mask=selected_mask_dev,
        scale=scale, threshold=float(threshold), T=T,
        B=Bq, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
        BT=BT, BK=BK, NB=NB,
        num_warps=8,
        num_stages=2,
        # num_stages=3,
    )

    selected_mask = selected_mask_dev.view(Bq, HQ, NB).bool().cpu()

    selected_indices: List[List[int]] = []
    for b in range(Bq):
        for hq in range(HQ):
            row = selected_mask[b, hq]
            idxs = torch.nonzero(row, as_tuple=False).view(-1).tolist()
            selected_indices.append(idxs)

    meta = dict(B=Bq, T=T, HQ=HQ, H=H, G=G, K=Kdim, V=Vdim, BT=BT, BK=BK, NB=NB, NQ=NQ, dtype=dtype, device=device, scale=scale)

    return o_full, selected_indices, selected_mask, selected_mask_dev, meta


# ================== 修改：稀疏计算 Triton 内核 (共享 Mask) ==================

@triton.jit
def attn_fwd_sparse_kernel(
    q, k, v, o, 
    block_mask_dev,  # [B*HQ, NB]，Q和KV共用此Mask
    scale, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    NB: tl.constexpr, NQ: tl.constexpr,
):
    """
    稀疏块注意力内核。
    修改说明：q_mask 和 kv_mask 现在共用同一个 mask 指针 (block_mask_dev)。
    """
    i_bh = tl.program_id(0)
    i_q = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    RCP_LN2: tl.constexpr = 1.4426950216

    # Q block 起始位置
    q_start = i_q * BT

    # Q block ptr
    p_q = tl.make_block_ptr(
        base=q + ((i_b * T * HQ + i_hq) * K),
        shape=(T, K),
        strides=(HQ * K, 1),
        offsets=(q_start, 0),
        block_shape=(BT, BK),
        order=(1, 0),
    )

    p_o = tl.make_block_ptr(
        base=o + (i_b * T * HQ + i_hq) * V,
        shape=(T, V),
        strides=(HQ * V, 1),
        offsets=(q_start, 0),
        block_shape=(BT, V),
        order=(1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))

    # softmax 累积缓存
    b_o = tl.zeros([BT, V], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 1. 检查当前 Q block 是否重要 (从共享 Mask 读取)
    # 因为 NQ == NB，且 Q 和 KV 在时间上对齐，所以可以直接用 i_q 索引 mask
    mask_idx_q = i_bh * NB + i_q
    q_imp_i32 = tl.load(block_mask_dev + mask_idx_q)
    q_imp = q_imp_i32 != 0

    # 遍历 KV 块
    for i_blk in range(0, NB):
        # 强制计算首尾块
        is_edge = (i_blk == 0) or (i_blk == NB - 1)

        # 2. 检查当前 KV block 是否重要 (从共享 Mask 读取)
        mask_idx_kv = i_bh * NB + i_blk
        kv_imp_i32 = tl.load(block_mask_dev + mask_idx_kv)
        kv_imp = kv_imp_i32 != 0

        # 如果 Q 重要且 KV 也重要，或者是边缘块，则计算
        do_blk = is_edge or (q_imp and kv_imp)

        if do_blk:
            i_s = i_blk * BT

            p_k = tl.make_block_ptr(
                base=k + ((i_b * T * H + i_h) * K),
                shape=(K, T),
                strides=(1, H * K),
                offsets=(0, i_s),
                block_shape=(BK, BT),
                order=(0, 1),
            )

            p_v = tl.make_block_ptr(
                base=v + ((i_b * T * H + i_h) * V),
                shape=(T, V),
                strides=(H * V, 1),
                offsets=(i_s, 0),
                block_shape=(BT, V),
                order=(1, 0),
            )

            b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
            b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

            b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

            o_q = q_start + tl.arange(0, BT)
            o_k = i_s + tl.arange(0, BT)
            m_q = o_q < T
            m_k = o_k < T
            causal = (o_q[:, None] >= o_k[None, :])
            mask = causal & m_q[:, None] & m_k[None, :]
            b_s = tl.where(mask, b_s, float('-inf'))

            b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
            b_r = tl.exp2(b_m - b_m_new)
            b_m = b_m_new

            b_p = tl.exp2(b_s - b_m[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.float32), b_v.to(tl.float32))

    eps = 1e-12
    b_den = tl.maximum(b_acc, eps)
    b_o = b_o / b_den[:, None]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ================== 修改：对外 API (移除 q_mask 入参) ==================

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
    delta=3.0, return_skip_ratio=False,
    precomputed_threshold=False,
):
    """
    流程：
    1) 先调用 prefill 内核，仅计算最后一个 Q block，并生成 kv_mask (标识重要块)。
    2) 再调用稀疏内核，为其余 Q block 计算。
       此处 Q 的 block Mask 和 KV 的 block Mask 相同，均使用第一步生成的 kv_mask。

    返回：
    - o_full: 结果张量
    - block_mask: 计算得到的块掩码 (CPU bool)
    - block_mask_dev: 计算得到的块掩码 (GPU int32)
    - meta: 元数据
    """
    # 第一步：计算最后一个 Q 块 + 获得 block Mask
    o, _, block_mask, block_mask_dev, meta = attn_forward_prefill_lastQblock(
        q, k, v, scale=scale, threshold=delta, causal=causal, BT=BT, BK=BK
    )

    B = meta["B"]
    H = meta["H"]
    HQ = meta["HQ"]
    G = meta["G"]
    Kdim = meta["K"]
    Vdim = meta["V"]
    BT = meta["BT"]
    BK = meta["BK"]
    NB = meta["NB"]
    NQ = meta["NQ"]
    scale = meta["scale"]

    # 稀疏计算其余 Q block（不覆盖最后一个 Q block）
    NQ_to_compute = max(0, NQ - 1)
    
    if NQ_to_compute > 0:
        grid = (B * HQ, NQ_to_compute)
        attn_fwd_sparse_kernel[grid](
            q=q, k=k, v=v, o=o,
            block_mask_dev=block_mask_dev, # Q 和 KV 使用同一个 Mask
            scale=scale, T=meta["T"],
            B=B, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
            BT=BT, BK=BK, NB=NB, NQ=NQ,
            num_warps=8,
            num_stages=2,
            # num_stages=3,
        )

    # return o_full, block_mask, block_mask_dev, meta
    if return_skip_ratio:
        return o, 0
    
    return o

    


# -------------------------- 演示/测试 --------------------------

def test_sparse_shared_mask():
    torch.manual_seed(0)
    B = 2
    T = 1024
    HQ = 8
    H = 4
    D = 64
    V = 64

    q = torch.randn(B, T, HQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, H, V, device="cuda", dtype=torch.float16)

    scale = 1.0 / math.sqrt(D)
    BT = 128
    threshold = 0.1

    # 调用修改后的接口
    o_full, block_mask, _, meta = attn_forward_prefill(
        q, k, v, scale=scale, delta=threshold, BT=BT, BK=None
    )

    print(f"稀疏输出形状: {o_full.shape}")
    print(f"生成的 Block Mask (CPU) 形状: {block_mask.shape}")
    
    # 验证 mask 内容 (打印第一个 batch 第一个 head 的 mask)
    print(f"Sample Mask [B=0, HQ=0]: {block_mask[0, 0].int().tolist()}")
    
    return o_full


if __name__ == "__main__":
    o = test_sparse_shared_mask()
    
    print(f"{o.shape=}")