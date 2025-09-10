import os
import math
import time
from typing import Tuple

import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from tqdm import tqdm

try:
    from op import exp2, log2
except:
    from .op import exp2, log2


@triton.jit
def parallel_attn_fwd_kernel(
    q, k, v, o, lse,   # pointers
    scale,             # float
    T,                 # int
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
    # grid = (i_v, i_t, i_hq)；batch 固定为 1
    i_v = tl.program_id(0)
    i_t = tl.program_id(1)
    i_hq = tl.program_id(2)
    i_h = i_hq // G  # GQA: 查询头归到对应 KV 头

    RCP_LN2: tl.constexpr = 1.4426950216  # 1/ln(2)

    # Q / O / LSE block pointers
    p_q  = tl.make_block_ptr(q + i_hq * K, (T, K), (HQ*K, 1), (i_t * BT, 0),         (BT, BK), (1, 0))
    p_o  = tl.make_block_ptr(o + i_hq * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV),  (BT, BV), (1, 0))
    p_ls = tl.make_block_ptr(lse + i_hq,  (T,),   (HQ,),      (i_t * BT,),           (BT,),    (0,))

    # 片上寄存/共享累积
    b_q   = tl.load(p_q, boundary_check=(0, 1))             # [BT, BK]
    b_o   = tl.zeros([BT, BV], dtype=tl.float32)            # [BT, BV]
    b_m   = tl.full([BT], float('-inf'), dtype=tl.float32)  # [BT] rolling max
    b_acc = tl.zeros([BT], dtype=tl.float32)                # [BT] rolling sum(exp)

    # 无因果遮罩：直接扫全量 key-block
    for i_s in range(0, T, BS):
        p_k = tl.make_block_ptr(k + i_h * K, (K, T), (1, H*K), (0, i_s),        (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + i_h * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))           # [BK, BS]
        b_v = tl.load(p_v, boundary_check=(0, 1))           # [BS, BV]
        b_s = tl.dot(b_q, b_k) * scale * RCP_LN2            # [BT, BS]，换底到 base-2

        # online log-sum-exp 合并
        b_m_new = tl.maximum(b_m, tl.max(b_s, 1))
        b_r     = exp2(b_m - b_m_new)                       # [BT]
        b_p     = exp2(b_s - b_m_new[:, None])              # [BT, BS]
        b_acc   = b_acc * b_r + tl.sum(b_p, 1)
        b_o     = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)
        b_m     = b_m_new

    # 归一化并写回
    b_o = b_o / b_acc[:, None]
    b_m = b_m + log2(b_acc)  # 保存 base-2 的 lse

    tl.store(p_o,  b_o.to(p_o.dtype.element_ty),  boundary_check=(0, 1))
    tl.store(p_ls, b_m.to(p_ls.dtype.element_ty), boundary_check=(0,))


def launch_parallel_attn_fwd(q, k, v, scale, G, BT=128, BS=64, BK=64, BV=64):
    # 期望：
    # q: [T, HQ, K], k: [T, H, K], v: [T, H, V]，都为 contiguous，device 相同
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    T, HQ, Kdim = q.shape
    Tk, H, Kdim2 = k.shape
    Tv, H2, Vdim = v.shape
    assert Tk == T and Tv == T and H == H2 and Kdim == Kdim2
    assert HQ % G == 0 and (HQ // G) == H  # HQ = H * G

    o   = torch.empty((T, HQ, Vdim), dtype=q.dtype, device=q.device)
    lse = torch.empty((T, HQ),       dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(Vdim, BV), triton.cdiv(T, BT), HQ)

    parallel_attn_fwd_kernel[grid](
        q, k, v, o, lse,
        scale,
        T,
        H=H, HQ=HQ, G=G,
        K=Kdim, V=Vdim,
        BT=BT, BS=BS, BK=BK, BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return o, lse


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
    # q/k/v: [B, Hq, T, D], 同 dtype/device
    assert backend in ("math", "flash")
    if backend == "flash":
        ctx = sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False)
    else:
        ctx = sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False)
    with ctx:
        # 注意：scale=None 则使用 1/sqrt(D)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    return o


def time_cuda(fn, warmup=10, iters=50):
    # 返回平均毫秒
    # 预热
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
    # 都转 float32 做误差评估
    a = o_ref.float()
    b = o_test.float()
    diff = (a - b).abs()
    rel = diff / (a.abs() + 1e-6)
    print(f"[Compare] {name_test} vs {name_ref}:")
    print(f"  max_abs: {diff.max().item():.6e}, mean_abs: {diff.mean().item():.6e}")
    print(f"  max_rel: {rel.max().item():.6e}, mean_rel: {rel.mean().item():.6e}")


# ======================
# 主逻辑：加载数据并对比
# ======================

if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # 路径自行修改
    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16  # 建议 fp16/bf16 才能触发 Flash
    BT, BS, BK, BV = 128, 64, 64, 64  # Triton tile，可按需调参

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # 注意：为了保证维度一致，这里不再只取最后一个 query 位置
        # 如果一定要只测最后一个位置，需要改 Triton kernel 支持 Lq != Lk/V
        B, Hq, T, D  = q_rope.shape
        _, Hkv, Tk, Dk = k_rope.shape
        _, Hkv2, Tv, Dv = v.shape
        assert B == 1, "当前 Triton 实现 batch 固定为 1"
        assert Tk == T and Tv == T, "q/k/v 的时间维需一致"
        assert Dk == D and Dv == D, "默认假设 K/V 维度与 Q 相同"
        assert Hq % Hkv == 0, "GQA: Hq 必须是 Hkv 的整数倍"
        G = Hq // Hkv

        # PyTorch SDPA 需要 q/k/v 的 head 数相同 -> 扩展 K/V 到 Hq
        k_sdpa, v_sdpa = expand_kv_for_gqa(k_rope, v, G)  # [B, Hq, T, D]
        assert k_sdpa.shape[1] == Hq and v_sdpa.shape[1] == Hq

        # 自定义 Triton 路径需要 [T, H, D] 排列
        q_tri = q_rope[0].permute(1, 0, 2).contiguous()  # [T, Hq, D]
        k_tri = k_rope[0].permute(1, 0, 2).contiguous()  # [T, Hkv, D]
        v_tri = v[0].permute(1, 0, 2).contiguous()       # [T, Hkv, Dv]
        scale = 1.0 / math.sqrt(D)

        # 计算输出
        # 1) PyTorch-Math
        o_torch = run_sdpa(q_rope, k_sdpa, v_sdpa, backend="math")  # [B, Hq, T, D]

        # 2) PyTorch-Flash（如果不可用会抛错，必要时可 try/except 降级）
        try:
            o_flash = run_sdpa(q_rope, k_sdpa, v_sdpa, backend="flash")
        except Exception as e:
            print(f"Flash SDP 不可用，降级到 Math。原因: {e}")
            o_flash = o_torch

        # 3) 自定义 Triton
        o_tri, lse = launch_parallel_attn_fwd(
            q_tri, k_tri, v_tri, scale, G, BT=BT, BS=BS, BK=BK, BV=BV
        )  # [T, Hq, Dv]
        o_tri_bhtd = o_tri.permute(1, 0, 2).unsqueeze(0).contiguous()  # [B=1, Hq, T, Dv]

        # 数值对比（都转到相同 dtype）
        compare_outputs(o_torch, o_flash, name_ref="torch-math", name_test="torch-flash")
        compare_outputs(o_torch, o_tri_bhtd, name_ref="torch-math", name_test="triton")

        # 计时对比
        iters, warmup = 50, 10

        def fn_torch_math():
            return run_sdpa(q_rope, k_sdpa, v_sdpa, backend="math")

        def fn_torch_flash():
            return run_sdpa(q_rope, k_sdpa, v_sdpa, backend="flash")

        def fn_triton():
            return launch_parallel_attn_fwd(q_tri, k_tri, v_tri, scale, G, BT=BT, BS=BS, BK=BK, BV=BV)[0]

        t_math  = time_cuda(fn_torch_math, warmup=warmup, iters=iters)
        try:
            t_flash = time_cuda(fn_torch_flash, warmup=warmup, iters=iters)
        except Exception:
            t_flash = float('nan')
        t_triton = time_cuda(fn_triton, warmup=warmup, iters=iters)

        print(f"[Latency ms] torch-math: {t_math:.3f} | torch-flash: {t_flash:.3f} | triton: {t_triton:.3f}")

        break

        # 为防止 OOM，可按需仅测前若干层
        # if layer_idx >= 2: break