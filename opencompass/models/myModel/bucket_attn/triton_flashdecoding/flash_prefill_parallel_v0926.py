import os
import math
import torch
import triton
import triton.language as tl
from tqdm import tqdm
from typing import Optional, Tuple

os.environ['TRITON_PRINT_AUTOTUNING'] = '1'

# ====== 新增：autotune 配置 ======
TUNE_CONFIGS = [
    # 轻量形状 / 小 V 瓶颈
    triton.Config({'BT': 64,  'BS': 128, 'BV': 64},  num_warps=4, num_stages=2),
    triton.Config({'BT': 128, 'BS': 128, 'BV': 64},  num_warps=4, num_stages=2),

    # 中等形状
    triton.Config({'BT': 128, 'BS': 128, 'BV': 128}, num_warps=4, num_stages=2),
    triton.Config({'BT': 128, 'BS': 256, 'BV': 64},  num_warps=8, num_stages=2),

    # 序列更长 / 更深流水（H100 更友好）
    triton.Config({'BT': 128, 'BS': 128, 'BV': 128}, num_warps=8, num_stages=3),
    triton.Config({'BT': 256, 'BS': 128, 'BV': 64},  num_warps=8, num_stages=3),

    # 保底小块
    triton.Config({'BT': 64,  'BS': 64,  'BV': 64},  num_warps=4, num_stages=2),
]

# 关键：对不同问题规模分别缓存最佳配置（T/K/V/HQ 变化会重新搜索一次后缓存）
AUTOTUNE_KEY = ['T', 'K', 'V', 'HQ']

@triton.autotune(configs=TUNE_CONFIGS, key=AUTOTUNE_KEY)
@triton.jit
def parallel_attn_fwd_kernel(
    q, k, v, o, scale, T,
    B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    i_n = i_b
    bos, eos = i_n * T, i_n * T + T
    RCP_LN2: tl.constexpr = 1.4426950216

    # Q：会被高复用，默认缓存策略即可
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1),
                            (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1),
                            (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_o = tl.zeros([BT, BV], dtype=tl.float32)

    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # 第一遍：完整过去块（无需因果掩码）
    for i_s in range(0, i_t * BT, BS):
        # K/V：顺序扫描，建议 L2-only（.cg）减少 L1 抖动
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K),
                                (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1),
                                (i_s, i_v * BV), (BS, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")  # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")  # [BS, BV]

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2  # [BT, BS]

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.exp2(b_m - b_m_new)
        b_m = b_m_new

        b_p = tl.exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    # 当前块：带因果掩码
    o_q = i_t * BT + tl.arange(0, BT)
    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K),
                                (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1),
                                (i_s, i_v * BV), (BS, BV), (1, 0))

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T

        b_k = tl.load(p_k, boundary_check=(0, 1), cache_modifier=".cg")
        b_v = tl.load(p_v, boundary_check=(0, 1), cache_modifier=".cg")

        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2
        b_s = tl.where((o_q[:, None] >= o_k[None, :]) & m_k[None, :], b_s, float('-inf'))

        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r = tl.exp2(b_m - b_m_new)
        b_m = b_m_new

        b_p = tl.exp2(b_s - b_m[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

    eps = 1e-12
    b_o = b_o / tl.maximum(b_acc[:, None], eps)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def parallel_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    causal: bool = True,
    BT: Optional[int] = None,
    BS: Optional[int] = None,
    BK: Optional[int] = None,   # 需等于 head_dim
    BV: Optional[int] = None,
) -> torch.Tensor:
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert causal

    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, T, HQ, Kdim = q.shape
    Bk, Tk, H, Kdim_k = k.shape
    Bv, Tv, H_v, Vdim = v.shape
    assert (B, T) == (Bk, Tk) == (Bv, Tv)
    assert Kdim == Kdim_k and H == H_v
    G = HQ // H
    assert G * H == HQ

    if BK is None:
        BK = Kdim
    assert BK == Kdim, "BK must equal head_dim (K)."

    device = q.device
    dtype = q.dtype
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty((B, T, HQ, Vdim), device=device, dtype=dtype)

    if scale is None:
        scale = 1.0 / math.sqrt(Kdim)

    # 关键：使用 lambda meta 定义 grid，适配 autotune 的 BT/BV
    grid = lambda meta: (
        triton.cdiv(Vdim, meta['BV']),
        triton.cdiv(T,    meta['BT']),
        B * HQ,
    )

    # 仅当你想强行固定 BT/BS/BV 时才传入；否则交给 autotune
    launch_kwargs = dict(
        q=q, k=k, v=v, o=o, scale=scale, T=T,
        B=B, H=H, HQ=HQ, G=G, K=Kdim, V=Vdim,
        BK=BK,
    )
    if BT is not None: launch_kwargs['BT'] = BT
    if BS is not None: launch_kwargs['BS'] = BS
    if BV is not None: launch_kwargs['BV'] = BV

    parallel_attn_fwd_kernel[grid](**launch_kwargs)
    return o

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

def bench_op(func, iters=100, warmup=10):
    """基准测试函数"""
    # Warmup
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / iters

if __name__ == "__main__":
    # 设置默认的块大小
    q_block_size = 128
    k_block_size = 128
    BK = 128
    BV = 128
    causal = True
    iters = 100
    warmup = 10

    sample_length = 16 * 1024
    sample_length = -1
    # 计时参数
    iters = 10
    warmup = 10

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

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype)       # [B, Hkv, T, Dv]

        if sample_length > 0:
            q_rope = q_rope[..., :sample_length, :]
            k_rope = k_rope[..., :sample_length, :]
            v      = v[..., :sample_length, :]

        # 统一转为 kernel 期望的 [B, T, H, D]
        q_t = q_rope.transpose(1, 2).contiguous()  # [B, T, Hq, D]
        k_t = k_rope.transpose(1, 2).contiguous()  # [B, T, Hkv, D]
        v_t = v.transpose(1, 2).contiguous()       # [B, T, Hkv, Dv]

        B, T, Hq, Kdim = q_t.shape
        _, _, Hkv, Kdim_k = k_t.shape
        _, _, Hkv_v, Vdim = v_t.shape
        assert Hkv == Hkv_v, "k/v 的头数必须一致"
        assert Kdim == Kdim_k, "q/k 的 head_dim 必须一致"

        print(f"B={B} T={T} Hq={Hq} Hkv={Hkv} Kdim={Kdim} Vdim={Vdim}")

        scale = 1.0 / math.sqrt(Kdim)

        # --- Triton Kernel Call ---
        o_triton = parallel_attn_fwd(
            q=q_t,              # [B, T, Hq, K]
            k=k_t,              # [B, T, Hkv, K]
            v=v_t,              # [B, T, Hkv, V]
            scale=scale, causal=causal,
            # BT=q_block_size,    # 注意参数名是 BT/BS
            # BS=k_block_size,
            BK=BK,              # kernel 要求 BK == Kdim
            # BV=BV,              # 可设为 Vdim，或保留默认
        )                       # 返回 [B, T, Hq, V]

        # --- FlashAttention Baseline Call ---
        o_flash = flash_compute(q_rope, k_rope, v, causal)  # [B, Hq, T, V]

        # --- Numerical Comparison ---
        # 将 Triton 输出转成与 Flash 相同布局 [B, Hq, T, V]
        o_triton_hqt = o_triton.transpose(1, 2)
        diff = (o_triton_hqt.float() - o_flash.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel = diff.max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # --- Performance Benchmark ---
        def run_triton():
            return parallel_attn_fwd(
                q=q_t, k=k_t, v=v_t,
                scale=scale, causal=causal,
                # BT=q_block_size, 
                # BS=k_block_size,
                BK=BK, 
                # BV=BV,
            )

        def run_flash():
            return flash_compute(q_rope, k_rope, v, causal=causal)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")
        break