import os
import math
import torch
import triton
import triton.language as tl
from tqdm import tqdm

@triton.jit
def parallel_attn_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
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
):
    # 获取三维网格中的线程块ID
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # 从组合索引中分离出批次和注意力头索引
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G  # 计算对应的KV头索引（用于GQA）
    
    # === 序列范围处理 ===
    # 定长序列：直接计算序列范围
    i_n = i_b
    bos, eos = i_n * T, i_n * T + T
    
    # 数学常量：1/ln(2)，用于log2和exp2计算
    RCP_LN2: tl.constexpr = 1.4426950216
    
    # === 创建内存访问指针 ===
    # 查询指针：指向当前批次的查询数据
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    # 输出指针：指向输出位置
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    # LSE指针：存储对数求和指数
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    
    # === 初始化缓存块 ===
    # 加载查询块到共享内存，大小为[BT, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    # 初始化输出块为全零，大小为[BT, BV]
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    
    # 初始化最大值和累加器，用于数值稳定的softmax计算
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)  # 每行最大值
    b_acc = tl.zeros([BT], dtype=tl.float32)  # 指数和
    
    # === 第一遍循环：处理因果注意力之前的块（无需掩码）===
    for i_s in range(0, i_t * BT, BS):
        # 创建键值指针
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        
        # 加载键值块
        b_k = tl.load(p_k, boundary_check=(0, 1))  # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))  # [BS, BV]
        
        # 计算注意力分数：Q*K^T * scale / ln(2)（为log2转换准备）
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2  # [BT, BS]
        
        # === 数值稳定的softmax计算 ===
        # 更新每行最大值
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        # 计算缩放因子（用于之前累积值的调整）
        b_r = exp2(b_mp - b_m)
        # 计算指数化的注意力权重
        b_p = exp2(b_s - b_m[:, None])  # [BT, BS]
        
        # 更新累加器
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # 更新输出：加权求和值向量
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        
        b_mp = b_m  # 保存旧的最大值
    
    # === 第二遍循环：处理当前块（应用因果掩码）===
    o_q = i_t * BT + tl.arange(0, BT)  # 当前查询的位置索引
    
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        # 创建键值指针（与第一遍类似）
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        
        # 计算键的位置和掩码
        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        
        # 加载键值块
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # 计算注意力分数
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        
        # === 应用因果掩码 ===
        # 只允许查询关注之前的键（包括当前位置）
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float('-inf'))
        
        # 数值稳定的softmax计算（与第一遍相同）
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = exp2(b_mp - b_m)
        b_p = exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_mp = b_m
    
    # === 最终归一化和存储 ===
    # 归一化输出：除以注意力权重和
    b_o = b_o / b_acc[:, None]
    # 计算最终的对数求和指数（用于反向传播）
    b_m += log2(b_acc)
    
    # 存储结果回全局内存
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))

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

def flash_compute(q_rope, k_rope, v, causal):
    from flash_attn import flash_attn_func
    # Input q: [B, Hq, Tq, D], k: [B, Hkv, Tk, D], v: [B, Hkv, Tk, Dv]
    # flash_attn_func expects [B, T, H, D]
    out = flash_attn_func(
        q_rope.transpose(1, 2),  # [B, Tq, Hq, D]
        k_rope.transpose(1, 2),  # [B, Tk, Hkv, D]
        v.transpose(1, 2),       # [B, Tk, Hkv, Dv]
        causal=causal,
    )
    # Convert output from [B, Tq, Hq, Dv] back to [B, Hq, Tq, Dv]
    return out.transpose(1, 2)

if __name__ == "__main__":
    from utils import load_qkvh

    # torch.set_float32_matmul_precision("high")

    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'

    # exp_root_subdir = 'Llama-3_2-3B/longbench_narrativeqa_42'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_46'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_54'
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_57'

    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16

    # 计时参数
    iters = 100
    warmup = 100

    # iters = 1
    # warmup = 0

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        # if layer_idx == 0:
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v      = layer_qkvh_data["v"].to('cuda')

        B, Hq, T, D = q_rope.shape
        _, Hkv, _, Dv = k_rope.shape
        print(f"B={B} {T=} {Hq=} {Hkv=} {D=} {Dv=}")

        scale = 1.0 / math.sqrt(D)
        
        # --- Triton Kernel Call ---
        # Pass original tensors directly. Layout conversion is handled inside.
        o_triton, skip_ratio = flash_attn_prefill_gqa(
            q=q_rope,
            k=k_rope,
            v=v,
            scale=scale,
            BLOCK_TQ=q_block_size,
            BLOCK_TK=k_block_size,
            causal=causal,
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (Note: This kernel doesn't skip blocks)")

        # --- FlashAttention Baseline Call ---
        # Pass original tensors directly.
        o_flash = flash_compute(q_rope, k_rope, v, causal)

        # --- Numerical Comparison ---
        # Both o_triton and o_flash now have the shape [B, Hq, T, Dv]
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # --- Performance Benchmark ---
        def run_triton():
            o, _ = flash_attn_prefill_gqa(q_rope, k_rope, v, scale=scale, causal=causal, BLOCK_TQ=q_block_size, BLOCK_TK=k_block_size)
            return o

        def run_flash():
            return flash_compute(q_rope, k_rope, v, causal=causal)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")