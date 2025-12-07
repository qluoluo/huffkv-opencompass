import math
import os
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ================== 辅助函数 ==================

def convert_to_triton_layout(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    v: torch.Tensor,
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

# ================== Last Block Importance Kernel (用于生成 Mask) ==================

@triton.jit
def attn_fwd_prefill_kernel(
    q, k, v, o, selected_mask, scale, threshold, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, NB: tl.constexpr,
):
    i_bh = tl.program_id(0)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    RCP_LN2: tl.constexpr = 1.4426950216
    q_start = tl.maximum(0, T - BT)

    p_q = tl.make_block_ptr(
        base=q + ((i_b * T * HQ + i_hq) * K),
        shape=(T, K), strides=(HQ * K, 1), offsets=(q_start, 0),
        block_shape=(BT, BK), order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        base=o + (i_b * T * HQ + i_hq) * V,
        shape=(T, V), strides=(HQ * V, 1), offsets=(q_start, 0),
        block_shape=(BT, V), order=(1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, V], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    first_block_max = -float('inf')

    for i_s in range(0, T, BT):
        p_k = tl.make_block_ptr(
            base=k + ((i_b * T * H + i_h) * K),
            shape=(K, T), strides=(1, H * K), offsets=(0, i_s),
            block_shape=(BK, BT), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=v + ((i_b * T * H + i_h) * V),
            shape=(T, V), strides=(H * V, 1), offsets=(i_s, 0),
            block_shape=(BT, V), order=(1, 0),
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

        row_max = tl.max(b_s, axis=1)
        block_max = tl.max(row_max, axis=0)

        if i_s == 0:
            first_block_max = block_max

        thr_cmp_val = first_block_max - threshold
        cond = block_max >= thr_cmp_val

        i_blk = i_s // BT
        write_idx = i_bh * NB + i_blk
        tl.store(selected_mask + write_idx, (1 if cond else 0), mask=None)

    eps = 1e-12
    b_den = tl.maximum(b_acc, eps)
    b_o = b_o / b_den[:, None]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def attn_forward_prefill_lastQblock(
    q, k, v, block_mask, scale=None, threshold=0.0, causal=True, BT=128, BK=None
):
    Bq, T, HQ, Kdim = q.shape
    Bk, Tk, H, Kdim_k = k.shape
    Bv, Tv, H_v, Vdim = v.shape
    G = HQ // H
    if BK is None: BK = Kdim
    o = torch.empty((Bq, T, HQ, Vdim), device=q.device, dtype=q.dtype)
    if scale is None: scale = 1.0 / math.sqrt(Kdim)
    NB = (T + BT - 1) // BT
    grid = (Bq * HQ,)

    attn_fwd_prefill_kernel[grid](
        q=q, k=k, v=v, o=o,
        selected_mask=block_mask,
        scale=scale, threshold=float(threshold), T=T,
        B=Bq, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
        BT=BT, BK=BK, NB=NB,
        num_warps=4, num_stages=2,
    )
    return o

# ================== 稀疏计算 Kernel (逻辑已修改) ==================

@triton.jit
def attn_fwd_sparse_kernel(
    q, k, v, o,
    block_mask_dev,
    total_count_ptr,
    viz_mask_ptr,  
    scale, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr,
    NB: tl.constexpr, NQ: tl.constexpr,
    COUNT_BLOCKS: tl.constexpr,
    SAVE_VIZ: tl.constexpr, 
    PATTERN_MODE: tl.constexpr # 新增：接收 pattern 模式 ID
):
    i_bh = tl.program_id(0)
    i_q = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    RCP_LN2: tl.constexpr = 1.4426950216
    q_start = i_q * BT

    p_q = tl.make_block_ptr(
        base=q + ((i_b * T * HQ + i_hq) * K),
        shape=(T, K), strides=(HQ * K, 1), offsets=(q_start, 0),
        block_shape=(BT, BK), order=(1, 0),
    )
    p_o = tl.make_block_ptr(
        base=o + (i_b * T * HQ + i_hq) * V,
        shape=(T, V), strides=(HQ * V, 1), offsets=(q_start, 0),
        block_shape=(BT, V), order=(1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, V], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 预加载当前 query block 的 importance mask (仅在 ffa 模式下需要)
    mask_idx_q = i_bh * NB + i_q
    q_imp_i32 = tl.load(block_mask_dev + mask_idx_q)
    q_imp = q_imp_i32 != 0

    local_computed_count = 0

    # 循环范围：只计算 Causal 部分 (0 到 i_q)
    for i_blk in range(0, i_q + 1):
        
        do_blk = False

        # ==================== Pattern Logic ====================
        
        # 模式 0: "dense" - 全计算 (Baseline)
        if PATTERN_MODE == 0:
            do_blk = True

        # 模式 1: "local" - Sink (Block 0) + Sliding Window (Last 4 blocks)
        elif PATTERN_MODE == 1:
            is_first = (i_blk == 0)
            dist = i_q - i_blk
            is_local = dist < 4  # Window size = 4 blocks
            do_blk = is_first or is_local

        # 模式 2: "ffa" - 原始逻辑 (Sink + Local + Importance)
        elif PATTERN_MODE == 2:
            # 1. Anchor/Sink
            is_first = (i_blk == 0)
            # 2. Local/Diagonal
            is_current = (i_blk == i_q)
            # 3. Mask Importance
            mask_idx_kv = i_bh * NB + i_blk
            kv_imp_i32 = tl.load(block_mask_dev + mask_idx_kv)
            kv_imp = kv_imp_i32 != 0
            
            do_blk = is_first or is_current or q_imp or kv_imp
        
        # =======================================================

        if do_blk:
            if COUNT_BLOCKS:
                local_computed_count += 1

            if SAVE_VIZ:
                viz_offset = i_bh * NQ * NB + i_q * NB + i_blk
                tl.store(viz_mask_ptr + viz_offset, 1) 

            i_s = i_blk * BT
            p_k = tl.make_block_ptr(
                base=k + ((i_b * T * H + i_h) * K),
                shape=(K, T), strides=(1, H * K), offsets=(0, i_s),
                block_shape=(BK, BT), order=(0, 1),
            )
            p_v = tl.make_block_ptr(
                base=v + ((i_b * T * H + i_h) * V),
                shape=(T, V), strides=(H * V, 1), offsets=(i_s, 0),
                block_shape=(BT, V), order=(1, 0),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
            b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

            b_s = tl.dot(b_q, b_k) * scale * RCP_LN2

            # Causal Mask
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

    if COUNT_BLOCKS:
        tl.atomic_add(total_count_ptr, local_computed_count)

    eps = 1e-12
    b_den = tl.maximum(b_acc, eps)
    b_o = b_o / b_den[:, None]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


# ================== 可视化函数 ==================

def plot_attention_heatmap(viz_tensor, filename="attention_heatmap_color.png"):
    if isinstance(viz_tensor, torch.Tensor):
        viz_tensor = viz_tensor.cpu().numpy()

    num_heads_total = viz_tensor.shape[0]
    NQ = viz_tensor.shape[1]
    NB = viz_tensor.shape[2]

    cols = int(math.ceil(math.sqrt(num_heads_total)))
    rows = int(math.ceil(num_heads_total / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten() if num_heads_total > 1 else [axes]

    # 0: Skipped/Masked, 1: Computed
    colors = ['#f0f0f0', '#3b4cc0']
    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for i in range(num_heads_total):
        ax = axes[i]
        im = ax.imshow(viz_tensor[i], cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
        ax.set_title(f"Head {i} Block Computation")
        ax.set_xlabel("Key Blocks")
        ax.set_ylabel("Query Blocks")
        ax.set_xticks(range(0, NB, max(1, NB//5)))
        ax.set_yticks(range(0, NQ, max(1, NQ//5)))

    for i in range(num_heads_total, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[Viz] Saved to {filename}")
    plt.close()


# ================== 主要入口 ==================

def attn_forward_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
    BT: int = 128,
    BK: Optional[int] = None,
    delta=5.0,
    return_skip_ratio=False,
    return_viz=False,
    pattern: str = "ffa",  # 新增 pattern 参数
    **kwargs,
):
    q, k, v = convert_to_triton_layout(q, k, v)
    
    Bq, T, HQ, Kdim = q.shape
    Bk, Tk, H, Kdim_k = k.shape
    Vdim = v.shape[-1]

    if BK is None: BK = Kdim
    if scale is None: scale = 1.0 / math.sqrt(Kdim)

    # 1. 验证并映射 Pattern
    # 0: dense, 1: local, 2: ffa
    valid_patterns = {
        "dense": 0,       # 全计算
        "local": 1,       # 仅 Sink + Local Window
        "ffa": 2,  # 原始的基于 Mask 的重要性稀疏
    }

    if pattern not in valid_patterns:
        raise ValueError(f"Invalid pattern '{pattern}'. Supported patterns: {list(valid_patterns.keys())}")
    
    pattern_mode_id = valid_patterns[pattern]

    G = HQ // H
    NB = (T + BT - 1) // BT
    NQ = NB

    block_mask_dev = torch.empty((Bq * HQ, NB), device=q.device, dtype=torch.int32)

    # 2. 计算 Last Block (Dense计算，用于决策)
    # 即使 pattern="dense" 或 "local"，如果为了保持接口一致或可视化，这步可以保留，
    # 但如果是 pure dense 其实这步可以跳过。这里为了逻辑简单我们总是计算它。
    o = attn_forward_prefill_lastQblock(
        q, k, v,
        block_mask=block_mask_dev,
        scale=scale, threshold=delta, causal=causal, BT=BT, BK=BK
    )

    computed_blocks_count = None
    if return_skip_ratio:
        computed_blocks_count = torch.zeros((1,), dtype=torch.int32, device=q.device)

    viz_tensor_dev = None
    if return_viz:
        viz_tensor_dev = torch.zeros((Bq * HQ, NQ, NB), dtype=torch.int32, device=q.device)

    # 3. 稀疏 Kernel 计算前 NQ-1 个 Block
    NQ_to_compute = max(0, NQ - 1)

    if NQ_to_compute > 0:
        grid = (Bq * HQ, NQ_to_compute)
        attn_fwd_sparse_kernel[grid](
            q, k, v, o,
            block_mask_dev,
            computed_blocks_count,  # 传入计数器指针
            viz_tensor_dev,
            scale, T,
            Bq, H, HQ, G, Kdim, Vdim,
            BT, BK, NB, NQ,
            return_skip_ratio,     # COUNT_BLOCKS flag
            return_viz,            # SAVE_VIZ flag
            PATTERN_MODE=pattern_mode_id, # 传入 Pattern ID
            num_warps=4, num_stages=2,
        )

    # ================== 返回值处理 ==================
    ret = [o]

    if return_skip_ratio:
        blocks_per_head_causal = NQ * (NQ + 1) // 2
        total_valid_blocks = (Bq * HQ) * blocks_per_head_causal
        
        sparse_blocks = computed_blocks_count.item()
        dense_blocks_count = (Bq * HQ) * NB 
        
        actual_computed = dense_blocks_count + sparse_blocks
        
        if total_valid_blocks > 0:
            skip_ratio = 1.0 - (actual_computed / total_valid_blocks)
        else:
            skip_ratio = 0.0
            
        ret.append(skip_ratio)

    if return_viz:
        if viz_tensor_dev is not None:
            viz_tensor_dev[:, -1, :] = 1
            pass 
        ret.append(viz_tensor_dev)

    if len(ret) == 1:
        return ret[0]
    return tuple(ret)


# -------------------------- 测试与 Flash Attention 对比 --------------------------

def test_sparse_viz_and_compare():
    print("Initializing test...")
    
    # 1. 参数设置
    B = 1
    T = 1024 * 4
    HQ = 32
    H = 16
    D = 64
    V = 64
    BT = 128
    
    # 2. 构造数据
    # torch.manual_seed(42)
    q = torch.randn(B, HQ, T, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, T, V, device="cuda", dtype=torch.float16)

    # 3. 运行 Triton Sparse Attention
    # 测试三种 pattern 之一，这里演示 "ffa"
    # 你可以尝试改为 "dense" 或 "local" 来验证不同逻辑
    test_pattern = "local" 
    threshold = 0
    
    print(f"Running Custom Sparse Attention (Pattern='{test_pattern}', Threshold={threshold})...")

    output_triton, ratio, viz_mask = attn_forward_prefill(
        q, k, v,
        BT=BT,
        delta=threshold,
        return_skip_ratio=True,
        return_viz=True,
        pattern=test_pattern # 传入 pattern
    )

    # 4. 运行 PyTorch Flash Attention (Ground Truth)
    print("Running Torch Flash Attention (Ground Truth)...")
    
    q_ref = q.permute(0, 2, 1, 3) 
    k_ref = k.permute(0, 2, 1, 3) 
    v_ref = v.permute(0, 2, 1, 3) 
    
    k_ref = k_ref.repeat_interleave(HQ // H, dim=1)
    v_ref = v_ref.repeat_interleave(HQ // H, dim=1)
    
    output_ref = F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True 
    )
    
    output_ref = output_ref.permute(0, 2, 1, 3)

    # 5. 验证正确性
    diff = (output_triton - output_ref).abs().max().item()
    print("-" * 40)
    print(f"Results Comparison:")
    print(f"Pattern Used: {test_pattern}")
    print(f"Skip Ratio: {ratio:.2%} (Theoretical reduction)")
    print(f"Max Difference vs Flash Attn: {diff:.6f}")
    
    if test_pattern == "dense" and diff < 0.1:
        print(">> VERIFICATION PASSED for dense (Matches standard output)")
    elif test_pattern != "dense":
        print(">> Note: For sparse/local patterns, deviation is expected.")
    else:
        print(">> VERIFICATION FAILED or High Deviation")
    print("-" * 40)

    # 6. 保存可视化
    print(f"Viz Mask shape: {viz_mask.shape}")
    plot_attention_heatmap(viz_mask, filename=f"test_pattern_{test_pattern}.png")


if __name__ == "__main__":
    test_sparse_viz_and_compare()