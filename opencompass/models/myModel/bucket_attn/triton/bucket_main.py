import os
import math
from typing import Tuple

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
# from torch.backends.cuda import sdp_kernel
from torch.nn.attention import sdpa_kernel
from tqdm import tqdm

try:
    from fla_op import exp2, log2
except:
    from .fla_op import exp2, log2

@triton.jit
def qlen1_attn_fwd_kernel(
    q, k, v, out, lse,
    scale, seq_len,
    n_kv_heads: tl.constexpr,
    n_q_heads: tl.constexpr,
    group_size: tl.constexpr,
    d_k: tl.constexpr,
    d_v: tl.constexpr,
    seq_block: tl.constexpr,
    k_block: tl.constexpr,
    v_block: tl.constexpr,
):
    v_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // group_size

    INV_LN2: tl.constexpr = 1.4426950408889634

    # Q指针访问
    q_ptr = tl.make_block_ptr(
        base=q + q_head_idx * d_k,
        shape=(1, d_k),
        strides=(d_k, 1),
        offsets=(0, 0),
        block_shape=(1, k_block),
        order=(1, 0)
    )

    # 输出指针
    out_ptr = tl.make_block_ptr(
        base=out + q_head_idx * d_v,
        shape=(1, d_v),
        strides=(d_v, 1),
        offsets=(0, v_idx * v_block),
        block_shape=(1, v_block),
        order=(1, 0)
    )

    # LSE指针
    lse_ptr = lse + q_head_idx

    # 加载查询（建议显式零填充以防越界）
    q_block = tl.load(q_ptr, boundary_check=(0, 1), padding_option='zero')

    # 初始化累加器
    out_acc = tl.zeros([1, v_block], dtype=tl.float32)
    max_val = tl.full([1], float('-inf'), dtype=tl.float32)
    exp_sum = tl.zeros([1], dtype=tl.float32)

    # 主循环
    for seq_start in range(0, seq_len, seq_block):
        # K指针
        k_ptr = tl.make_block_ptr(
            base=k + kv_head_idx * d_k,
            shape=(seq_len, d_k),
            strides=(n_kv_heads * d_k, 1),
            offsets=(seq_start, 0),
            block_shape=(seq_block, k_block),
            order=(1, 0)
        )

        # V指针
        v_ptr = tl.make_block_ptr(
            base=v + kv_head_idx * d_v,
            shape=(seq_len, d_v),
            strides=(n_kv_heads * d_v, 1),
            offsets=(seq_start, v_idx * v_block),
            block_shape=(seq_block, v_block),
            order=(1, 0)
        )

        # 加载数据（显式零填充）
        k_block_data = tl.load(k_ptr, boundary_check=(0, 1), padding_option='zero')  # [seq_block, k_block]
        v_block_data = tl.load(v_ptr, boundary_check=(0, 1), padding_option='zero')  # [seq_block, v_block]

        # ----- 用逐元素乘+求和替代 tl.dot -----
        # attn_scores: [1, seq_block] = sum_k (q[1, k] * K[seq_block, k])
        q_f32 = q_block.to(tl.float32)               # [1, k_block]
        k_f32 = k_block_data.to(tl.float32)          # [seq_block, k_block]
        attn_scores = tl.sum(k_f32 * q_f32, axis=1)[None, :]  # [1, seq_block]
        attn_scores = attn_scores * (scale * INV_LN2)

        # 在线 softmax
        new_max = tl.maximum(max_val, tl.max(attn_scores, 1))          # [1]
        scale_factor = tl.exp2(max_val - new_max)                      # [1]
        exp_weights = tl.exp2(attn_scores - new_max[:, None])          # [1, seq_block]

        exp_sum = exp_sum * scale_factor + tl.sum(exp_weights, 1)      # [1]

        # out_acc += exp_weights @ V
        # [1, seq_block] x [seq_block, v_block] -> [1, v_block]
        vw = v_block_data.to(tl.float32)                               # [seq_block, v_block]
        ew = exp_weights.to(tl.float32)                                # [1, seq_block]
        o_update = tl.sum(vw * tl.trans(ew), axis=0)[None, :]          # [1, v_block]
        out_acc = out_acc * scale_factor[:, None] + o_update

        max_val = new_max

    # 归一化和写回
    out_acc = out_acc / exp_sum[:, None]
    final_lse = max_val + tl.log2(exp_sum)

    tl.store(out_ptr, out_acc.to(out_ptr.dtype.element_ty), boundary_check=(0, 1))
    # tl.store(lse_ptr, final_lse[0].to(tl.float32))
    tl.store(lse_ptr, final_lse.to(tl.float32))

def launch_attention_q1(q, k, v, scale, group_size, seq_block=128, k_block=64, v_block=64):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    
    n_q_heads, d_k = q.shape
    seq_len, n_kv_heads, d_k2 = k.shape
    seq_len2, n_kv_heads2, d_v = v.shape
    
    assert seq_len == seq_len2 and n_kv_heads == n_kv_heads2 and d_k == d_k2
    assert n_q_heads % group_size == 0 and (n_q_heads // group_size) == n_kv_heads
    
    out = torch.empty((n_q_heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.empty((n_q_heads,), dtype=torch.float32, device=q.device)
    
    grid = (triton.cdiv(d_v, v_block), n_q_heads)
    
    qlen1_attn_fwd_kernel[grid](
        q, k, v, out, lse,
        scale, seq_len,
        n_kv_heads=n_kv_heads,
        n_q_heads=n_q_heads,
        group_size=group_size,
        d_k=d_k, d_v=d_v,
        seq_block=seq_block,
        k_block=k_block,
        v_block=v_block,
        num_warps=4,
        num_stages=2,
    )
    
    return out, lse


# ======================
# 基准与对比工具函数
# ======================

def expand_kv_for_gqa(k: torch.Tensor, v: torch.Tensor, G: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # 输入: [B, H_kv, T, D]; 返回: [B, H_q (=H_kv*G), T, D]
    if G == 1:
        return k, v
    k_exp = k.repeat_interleave(G, dim=1).contiguous()
    v_exp = v.repeat_interleave(G, dim=1).contiguous()
    return k_exp, v_exp


@torch.inference_mode()
def run_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, backend: str) -> torch.Tensor:
    # q/k/v: [B, Hq, Tq/Tk, D], 同 dtype/device
    assert backend in ("math", "flash")
    if backend == "flash":
        # ctx = sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False)
        ctx = sdpa_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False)
    else:
        # ctx = sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
        ctx = sdpa_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    with ctx:
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return o


def time_cuda(fn, warmup=10, iters=50):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        _ = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sum(times) / len(times)


def compare_outputs(o_ref: torch.Tensor, o_test: torch.Tensor, name_ref="ref", name_test="test"):
    a = o_ref.float()
    b = o_test.float()
    diff = (a - b).abs()
    rel = diff / (a.abs() + 1e-6)
    print(f"[Compare] {name_test} vs {name_ref}:")
    print(f"  max_abs: {diff.max().item():.6e}, mean_abs: {diff.mean().item():.6e}")
    print(f"  max_rel: {rel.max().item():.6e}, mean_rel: {rel.mean().item():.6e}")


# ======================
# 主逻辑：仅 qlen=1
# ======================

if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16  # 建议 fp16/bf16 才能触发 Flash
    BS, BK, BV = 128, 64, 64  # Triton tile，可按需调参

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # 只取最后一个查询位置 -> qlen=1
        q_rope_1 = q_rope[:, :, -1:, :]  # [B, Hq, 1, D]

        B, Hq, Tq, D  = q_rope_1.shape
        _, Hkv, Tk, Dk = k_rope.shape
        _, Hkv2, Tv, Dv = v.shape
        assert B == 1, "当前 Triton 实现 batch 固定为 1"
        assert Tq == 1, "qlen 必须为 1"
        assert Tk == Tv, "K/V 的时间维需一致"
        assert Dk == D and Dv == D, "默认假设 K/V 维度与 Q 相同"
        assert Hq % Hkv == 0, "GQA: Hq 必须是 Hkv 的整数倍"
        G = Hq // Hkv
        scale = 1.0 / math.sqrt(D)

        # PyTorch SDPA 路径（需要扩展 KV 到 Hq）
        k_sdpa, v_sdpa = expand_kv_for_gqa(k_rope, v, G)  # [B, Hq, T, D]
        # 1) torch-math
        o_torch = run_sdpa(q_rope_1, k_sdpa, v_sdpa, backend="math")  # [B, Hq, 1, D]
        # 2) torch-flash（若不可用将抛错，按需 try/except）
        try:
            o_flash = run_sdpa(q_rope_1, k_sdpa, v_sdpa, backend="flash")
        except Exception as e:
            print(f"Flash SDP 不可用，降级到 Math。原因: {e}")
            o_flash = o_torch

        # Triton 路径（GQA 原生）
        # q1: [HQ, D]
        q1 = q_rope_1[0, :, 0, :].contiguous()
        # k: [T, Hkv, D]；v: [T, Hkv, D]
        k_tri = k_rope[0].permute(2, 0, 1).contiguous()  # [T, Hkv, D] (等价写法，确保最后一维为 D)
        k_tri = k_rope[0].permute(2, 0, 1)  # 纠正：上一行错误，多余变换，下面统一采用正确顺序
        k_tri = k_rope[0].permute(2, 0, 1)  # 实际上应为 [T, Hkv, D] -> 但 k_rope[0] 是 [Hkv, T, D]
        # 更清晰的写法如下：
        k_tri = k_rope[0].permute(1, 0, 2).contiguous()  # [T, Hkv, D]
        v_tri = v[0].permute(1, 0, 2).contiguous()       # [T, Hkv, D]

        o_tri, lse = launch_attention_q1(q1, k_tri, v_tri, scale, G, BS, BK, BV)  # [Hq, D]
        o_tri_bhtd = o_tri.unsqueeze(0).unsqueeze(2).contiguous()  # [B=1, Hq, 1, D]

        # 数值对比
        compare_outputs(o_torch, o_flash, name_ref="torch-math", name_test="torch-flash")
        compare_outputs(o_torch, o_tri_bhtd, name_ref="torch-math", name_test="triton(q=1)")

        # 计时
        warmup, iters = 10, 50

        def fn_torch_math():
            return run_sdpa(q_rope_1, k_sdpa, v_sdpa, backend="math")

        def fn_torch_flash():
            return run_sdpa(q_rope_1, k_sdpa, v_sdpa, backend="flash")

        def fn_triton():
            return launch_attention_q1(q1, k_tri, v_tri, scale, G, BS, BK, BV)[0]

        t_math  = time_cuda(fn_torch_math, warmup=warmup, iters=iters)
        try:
            t_flash = time_cuda(fn_torch_flash, warmup=warmup, iters=iters)
        except Exception:
            t_flash = float('nan')
        t_triton = time_cuda(fn_triton, warmup=warmup, iters=iters)

        print(f"[Latency ms @ qlen=1] torch-math: {t_math:.3f} | torch-flash: {t_flash:.3f} | triton: {t_triton:.3f}")

        # 为防止 OOM，可按需仅测前若干层
        # if layer_idx >= 2: break