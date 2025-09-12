import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def attn_fwd_q1_b1_kernel(
    q,         # [HQ, K]
    k,         # [HKV, T, K]
    v,         # [T, HKV, V]
    o,         # [HQ, V]
    lse,       # [HQ]
    scale,     # float
    T,         # int
    HKV: tl.constexpr,  # num kv heads
    HQ: tl.constexpr,   # num q heads
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,    # group size = HQ // HKV
    BS: tl.constexpr,   # tile size for time
    BK: tl.constexpr,   # tile size for K dim
    BV: tl.constexpr,   # tile size for V dim
):
    # 网格：pid_v 切 V 维；pid_hq 切 HQ 维
    pid_v = tl.program_id(0)
    pid_hq = tl.program_id(1)

    # GQA：将 query head -> kv head
    i_hq = pid_hq
    i_h = i_hq // G  # 同一组的 G 个 q 头共享一个 kv 头

    # 本程序块负责的 V 切片
    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    # 常量
    RCP_LN2 = 1.4426950408889634  # 1/ln(2)

    # 在线 softmax 状态（单查询位置）
    b_m = tl.full((), float('-inf'), tl.float32)  # 当前最大值（log2 域）
    b_acc = tl.zeros((), tl.float32)              # 归一化分母（base-2 域）
    b_o = tl.zeros([BV], tl.float32)              # 输出累积（float32）

    # 遍历时间维（Key/Value 的长度 T），每次处理 BS 个 token
    for s in range(0, T, BS):
        offs_t = s + tl.arange(0, BS)
        t_mask = offs_t < T

        # 先计算分数 b_s: [BS]，使用 K 维分块 BK 做点积
        b_s = tl.zeros([BS], tl.float32)
        for kk in range(0, K, BK):
            offs_k = kk + tl.arange(0, BK)
            k_mask = offs_k < K

            # q_chunk: [BK]
            q_ptrs = q + i_hq * K + offs_k
            q_chunk = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # k_chunk: [BK, BS]
            # k[hkv, t, k] 的地址：base + hkv*T*K + t*K + k
            k_ptrs = (
                k
                + i_h * T * K
                + (offs_t[None, :] * K)
                + offs_k[:, None]
            )
            k_chunk = tl.load(
                k_ptrs, mask=(k_mask[:, None] & t_mask[None, :]), other=0.0
            ).to(tl.float32)

            # 累加到分数（还未乘缩放）
            # [BS] += sum_{BK}(q_chunk * k_chunk)
            b_s += tl.sum(q_chunk[:, None] * k_chunk, axis=0)

        # 乘缩放并转为以 2 为底的指数域
        b_s = b_s * scale * RCP_LN2

        # 数值稳定在线 softmax 更新（base-2）
        b_s_masked = tl.where(t_mask, b_s, float('-inf'))
        m_tile = tl.max(b_s_masked, axis=0)

        new_m = tl.maximum(b_m, m_tile)
        # 旧和的缩放因子
        r = tl.exp2(b_m - new_m)
        # 本块概率（base-2 域），无效位置置 0
        b_p = tl.exp2(b_s - new_m)
        b_p = tl.where(t_mask, b_p, 0.0)

        # 先缩放历史，再加上本块
        b_acc = b_acc * r + tl.sum(b_p, axis=0)

        # 加载 V 并累积到输出
        # v[t, hkv, v] 的地址：base + t*HKV*V + hkv*V + v
        v_ptrs = (
            v
            + (offs_t[:, None] * (HKV * V))
            + (i_h * V)
            + v_offs[None, :]
        )
        b_v = tl.load(
            v_ptrs, mask=(t_mask[:, None] & v_mask[None, :]), other=0.0
        ).to(tl.float32)

        # b_o: [BV]
        b_o = b_o * r + tl.sum(b_p[:, None] * b_v, axis=0)

        # 更新当前最大值
        b_m = new_m

    # 归一化得到最终输出；lse = log2(sum(exp2(scores))) = b_m + log2(b_acc)
    b_o = b_o / b_acc
    lse_val = b_m + tl.log2(b_acc)

    # 写回
    o_ptrs = o + i_hq * V + v_offs
    tl.store(o_ptrs, b_o.to(o_ptrs.dtype.element_ty), mask=v_mask)
    tl.store(lse + i_hq, lse_val.to((lse + i_hq).dtype.element_ty))


def attn_fwd_q1_b1(
    q: torch.Tensor,  # [HQ, K], fp16/bf16/fp32
    k: torch.Tensor,  # [HKV, T, K], same dtype as q
    v: torch.Tensor,  # [T, HKV, V], same dtype as q
    scale: float = None,
    BS: int = 64,
    BK: int = 64,
    BV: int = 64,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 2 and k.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k.shape
    Tv, HKV2, V = v.shape
    assert Kk == K and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0, "GQA 需要 HQ 是 HKV 的整数倍"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # 输出张量
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(V, BV), HQ)
    attn_fwd_q1_b1_kernel[grid](
        q, k, v, o, lse,
        scale, T,
        HKV=HKV, HQ=HQ, K=K, V=V,
        G=G,
        BS=BS, BK=BK, BV=BV
    )
    return o, lse


def to_triton_layout(q_rope_1, k_rope, v):
    # q_rope_1: [B, Hq, 1, D], k_rope: [B, Hkv, T, D], v: [B, Hkv, T, Dv]
    # 返回 q:[HQ,K], k:[HKV,T,K], v:[T,HKV,V]
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv
    assert T == Tv
    assert Dq == Dk, "q/k head_dim 不一致"
    assert Hkv == Hvv, "k/v 的 head 数必须一致"
    assert B == 1, "该 kernel 仅支持 batch=1"
    assert qlen == 1, "该 kernel 仅支持 qlen=1"
    assert Hq % Hkv == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

    # 取 batch=0
    q_triton = q_rope_1[0, :, 0, :].contiguous()            # [HQ, D]
    k_triton = k_rope[0, :, :, :].contiguous()              # [HKV, T, D]
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()  # [T, HKV, Dv]
    return q_triton, k_triton, v_triton


def flash_compute(q_rope_1, k_rope, v):
    from flash_attn import flash_attn_func
    # q_rope_1: [B=1, H, 1, D], k_rope: [1, H, T, D], v: [1, H, T, Dv]
    out = flash_attn_func(
        q_rope_1.transpose(1, 2),
        k_rope.transpose(1, 2),
        v.transpose(1, 2),
        causal=False,
    )
    # out = out.transpose(1, 2)
    out = out.squeeze(0).squeeze(0)
    return out


def lse_reference_base2_gqa(q_triton, k_triton, scale):
    # 计算 lse 的 reference（以 2 为底），用于与 triton kernel 的 lse 对比（支持 GQA）
    # q_triton: [HQ,K], k_triton: [HKV,T,K]
    qf = q_triton.float()
    kf = k_triton.float()
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
    BS, BK, BV = 128, 64, 64  # Triton tile，可按需调参

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

        # 准备给 Triton 内核的布局（支持 GQA）
        q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)

        # 运行 Triton 实现
        scale = 1.0 / math.sqrt(D)
        o_triton, lse_triton = attn_fwd_q1_b1(
            q_triton, k_triton, v_triton,
            scale=scale, BS=BS, BK=BK, BV=BV
        )  # o:[HQ,V], lse:[HQ] (以 2 为底)

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # print(f"{o_triton.shape=}, {o_flash.shape=}")

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        # LSE 参考（高精度，用于 sanity check）
        lse_ref2 = lse_reference_base2_gqa(q_triton, k_triton, scale)  # [HQ], base-2
        lse_max_abs = (lse_triton.float() - lse_ref2).abs().max().item()
        lse_rel = (lse_triton.float() - lse_ref2).abs().max() / (lse_ref2.abs().max().clamp_min(1e-6))
        lse_rel = lse_rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, rel={rel:.3e}")
        print(f"LSE (base-2) diff vs FP32 ref: max_abs={lse_max_abs:.3e}, rel={lse_rel:.3e}")

        # 性能对比
        def run_triton():
            o, _ = attn_fwd_q1_b1(q_triton, k_triton, v_triton, scale=scale, BS=BS, BK=BK, BV=BV)
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")