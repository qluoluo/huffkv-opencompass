import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F

from attn_cuda_wrapper import (
    attn_fwd_q1_b1_splitT as attn_cuda_splitT,
    calc_qk_threshold,         # 可选：你也可以沿用自己现有的同名函数
    to_triton_layout,          # 保持数据布局转换一致
)

def flash_compute(q_rope_1, k_rope, v):
    from flash_attn import flash_attn_func
    # q_rope_1: [B=1, H, 1, D], k_rope: [1, H, T, D], v: [1, H, T, Dv]
    out = flash_attn_func(
        q_rope_1.transpose(1, 2),
        k_rope.transpose(1, 2),
        v.transpose(1, 2),
        causal=False,
    )
    out = out.squeeze(0).squeeze(0)  # [H, Dv]
    return out


def lse_reference_base2_gqa(q_cuda, k_cuda, scale):
    # 计算 lse 的 reference（以 2 为底），用于与 cuda kernel 的 lse 对比（支持 GQA）
    # q_cuda: [HQ,K], k_cuda: [HKV,T,K]
    qf = q_cuda.float()
    kf = k_cuda.float()
    HQ, K = qf.shape
    HKV, T, Kk = kf.shape
    assert Kk == K
    assert HQ % HKV == 0
    G = HQ // HKV
    # 扩展 k 到 HQ 个 head（仅用于参考数值）
    if G != 1:
        kf_rep = kf.repeat_interleave(G, dim=0)  # [HQ, T, K]
    else:
        kf_rep = kf
    # scores[hq, t] = (q[hq] · kf_rep[hq, t]) * scale
    scores = torch.einsum('hk, htk -> ht', qf, kf_rep) * scale
    # 以 e 为底的 logsumexp -> 转成以 2 为底
    RCP_LN2 = 1.4426950408889634
    lse_e = torch.logsumexp(scores, dim=-1)         # [HQ]
    lse_base2 = lse_e * RCP_LN2                     # [HQ]
    return lse_base2


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
    BS, BK, BV = 256, 64, 64  # cuda tile，可按需调参

    # 计时参数
    iters = 50
    warmup = 10

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # 只取最后一个查询位置 -> qlen=1
        q_rope_1 = q_rope[:, :, -1:, :]  # [B, Hq, 1, D]

        B, Hq, qlen, D = q_rope_1.shape
        Bk, Hkv, T, Dk = k_rope.shape
        Bv, Hv, Tv, Dv = v.shape
        assert B == 1, "该 demo 仅支持 batch=1"
        assert qlen == 1, "该 demo 仅支持 qlen=1"
        assert Hkv == Hv, "k/v heads 必须一致"
        assert D == Dk, "q/k head_dim 不一致"
        assert T == Tv
        assert Hq % Hkv == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

        print(f"{T=}")

        # 准备给 cuda 内核的布局（支持 GQA）
        q_cuda, k_cuda, v_cuda = to_triton_layout(q_rope_1, k_rope, v)

        # 运行 cuda 实现
        scale = 1.0 / math.sqrt(D)

        # 关键：预计算阈值（不参与时间计算）
        qk_thresholds = calc_qk_threshold(q_cuda, k_cuda, scale).contiguous()

        o_cuda, lse_cuda = attn_cuda_splitT(
            q_cuda, k_cuda, v_cuda,
            scale=scale, BS=256, BK=64, BV=64,
            qk_thresholds=qk_thresholds,  # 或者传 None 让内部计算
        )  # o:[HQ,V], lse:[HQ] (以 2 为底)

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # 数值对比（与 Flash 输出）
        max_abs = (o_cuda.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_cuda.float() - o_flash.float()).abs().mean().item()
        rel = (o_cuda.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        # LSE 参考（高精度，用于 sanity check）
        lse_ref2 = lse_reference_base2_gqa(q_cuda, k_cuda, scale)  # [HQ], base-2
        lse_max_abs = (lse_cuda.float() - lse_ref2).abs().max().item()
        lse_rel = (lse_cuda.float() - lse_ref2).abs().max() / (lse_ref2.abs().max().clamp_min(1e-6))
        lse_rel = lse_rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")
        print(f"LSE (base-2) diff vs FP32 ref: max_abs={lse_max_abs:.3e}, rel={lse_rel:.3e}")

        # 性能对比：计时不包含阈值计算
        def run_cuda():
            o, _ = attn_cuda_splitT(
                q_cuda, k_cuda, v_cuda,
                scale=scale, BS=BS, BK=BK, BV=BV,
                qk_thresholds=qk_thresholds,  # 使用已计算好的阈值
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_cuda = bench_op(run_cuda, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: cuda={ms_cuda:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_cuda/ms_flash:.2f}x")