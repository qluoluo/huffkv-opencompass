# 根据阈值筛选qk内积过小的，不进行后续操作
# 计时不包括筛选时间

import math
import os
from tqdm import tqdm

import torch
import triton
import triton.language as tl



@triton.jit
def attn_find_threshold_two_blocks(
    q, k, thres_buf, scale, T, NTB, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    base_hq = pid_hkv * G

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k = tl.arange(0, K)

    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    # tb = 0
    tb0 = 0
    offs_t0 = tb0 * BS + tl.arange(0, BS)
    t_mask0 = offs_t0 < T
    k_ptrs0 = k + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(k_ptrs0, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # tb = NTB-1（当 NTB==1 时等于 0，与 tb0 相同）
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    k_ptrs1 = k + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(k_ptrs1, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    m2 = tl.maximum(m0, m1)
    th = m2 - delta
    tl.store(thres_buf + (base_hq + rows), th, mask=row_mask)


def precompute_attn_thresholds(
    q: torch.Tensor,    # [HQ, K]
    k: torch.Tensor,    # [HKV, T, K]
    scale: float,
    BS: int,
    delta: float = 1000.0,
):
    assert q.is_cuda and k.is_cuda
    HQ, K = q.shape
    HKV, T, Kk = k.shape
    assert Kk == K and (HQ % HKV == 0)
    G = HQ // HKV
    NTB = triton.cdiv(T, BS)

    thres_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    grid_th = (HKV, 1)
    attn_find_threshold_two_blocks[grid_th](
        q, k, thres_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
    )
    return thres_buf


@triton.jit
def attn_fwd_stage1_pruned(
    q, k, v,
    m_buf, l_buf, o_buf,
    thres_buf, mask_buf,          # 新增：mask_buf [HKV, NTB]，int8
    scale, T, NTB,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)
    base_hq = pid_hkv * G

    s0 = pid_tb * BS
    offs_t = s0 + tl.arange(0, BS)
    t_mask = offs_t < T

    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    offs_k = tl.arange(0, K)
    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    k_ptrs = k + pid_hkv * T * K + (offs_t[None, :] * K) + offs_k[:, None]
    k_tile = tl.load(k_ptrs, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask[None, :]), other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
    b_s_act = tl.where(t_mask[None, :], b_s, NEG_INF)

    m_rows_blk = tl.max(b_s_act, axis=1)

    th_rows = tl.load(thres_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)
    below = (m_rows_blk < th_rows) & row_mask
    n_below = tl.sum(below.to(tl.int32), axis=0)
    n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
    prune_blk = n_below == n_valid

    v_offs = tl.arange(0, V)

    if not prune_blk:
        # 只有在有效块时才计算并写回中间结果
        m_rows = m_rows_blk
        b_p = tl.where(t_mask[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
        l_rows = tl.sum(b_p, axis=1)

        # V 路
        need_v = tl.sum(t_mask.to(tl.int32), axis=0) > 0
        o_tile = tl.zeros([BM_DOT, V], tl.float32)
        if need_v:
            v_ptrs = v + (offs_t[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

        m_ptrs = m_buf + (base_hq + rows) * NTB + pid_tb
        l_ptrs = l_buf + (base_hq + rows) * NTB + pid_tb
        o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTB * V) + pid_tb * V + v_offs[None, :]

        tl.store(m_ptrs, m_rows, mask=row_mask)
        tl.store(l_ptrs, l_rows, mask=row_mask)
        tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

        # 标记该 tb 为有效
        tl.store(mask_buf + pid_hkv * NTB + pid_tb, tl.full((), 1, tl.int8))
    else:
        # 标记该 tb 为无效（阶段2会直接跳过）
        tl.store(mask_buf + pid_hkv * NTB + pid_tb, tl.full((), 0, tl.int8))


@triton.jit
def attn_fwd_stage2_masked(
    m_buf, l_buf, o_buf,     # [HQ, NTB], [HQ, NTB], [HQ, NTB, V]
    mask_buf,                # [HKV, NTB], int8
    o,                       # [HQ, V], out dtype = q.dtype
    lse,                     # [HQ], fp32
    NTB,                     # int
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    g       = tl.program_id(1)
    pid_hq  = pid_hkv * G + g

    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)

    b_m   = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o   = tl.zeros([V], tl.float32)

    for tb in range(0, NTB):
        keep = tl.load(mask_buf + pid_hkv * NTB + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTB + tb)
            l_b = tl.load(l_buf + pid_hq * NTB + tb)
            o_b = tl.load(o_buf + pid_hq * (NTB * V) + tb * V + v_offs)

            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk  = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o   = b_o   * r_prev + o_b * r_blk
            b_m   = new_m
        # 否则：直接跳过，无任何 V 维计算

    # 空结果保护
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    lse_val  = tl.where(is_empty, neg_inf, b_m + tl.log2(b_acc))

    tl.store(lse + pid_hq, lse_val)
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))

def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    # mask_buf: [HKV, NTB], int8, 1=keep, 0=skip
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()     # device 上做 sum，避免 int8 溢出
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())           # 注意：.item() 会触发同步

def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,      # [HQ, K]
    k: torch.Tensor,      # [HKV, T, K]
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    delta: float = 5.0,
    thres_buf: torch.Tensor | None = None,
    return_skip_ratio: bool = False,
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

    NTB = triton.cdiv(T, BS)

    # 输出与中间缓冲
    o   = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)
    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    # 每 (HKV, tb) 的有效性掩码
    mask_buf = torch.empty((HKV, NTB), device=q.device, dtype=torch.int8)

    # 阈值缓冲：如果外部没给，就在这里算
    if thres_buf is None:
        thres_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)
        attn_find_threshold_two_blocks[(HKV, 1)](
            q, k, thres_buf,
            scale, T, NTB, delta,
            HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
        )
    else:
        assert thres_buf.shape == (HQ,) and thres_buf.dtype == torch.float32 and thres_buf.device == q.device

    # Stage 1：计算、裁剪并写 mask_buf
    attn_fwd_stage1_pruned[(HKV, NTB)](
        q, k, v,
        m_buf, l_buf, o_buf,
        thres_buf, mask_buf,
        scale, T, NTB,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS,
    )

    # 如果需要，统计跳过的 block 比率（不建议在计时循环里开启）
    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2：按 mask 跳过无效块
    attn_fwd_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, lse, NTB,
        HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, lse, skip_ratio
    else:
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
    out = out.squeeze(0).squeeze(0)  # [H, Dv]
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

    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'

    # exp_root_subdir = 'Llama-3_2-3B/longbench_narrativeqa_42'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_46'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_54'
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_57'

    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256
    delta = 5.0

    print(f"{BS=}")

    # 计时参数
    iters = 100
    warmup = 100

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

        print(f"{T=} {Hq=} {Hkv=} {D=} {Dv=}")

        # 准备给 Triton 内核的布局（支持 GQA）
        q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)
        scale = 1.0 / math.sqrt(D)

        thres_buf = precompute_attn_thresholds(
            q_triton, k_triton,
            scale=scale, BS=BS, delta=delta,
        )
        torch.cuda.synchronize()

        o_triton, lse_triton, skip_ratio = attn_fwd_q1_b1_splitT(
            q_triton, k_triton, v_triton,
            scale=scale, BS=BS,
            thres_buf=thres_buf,
            return_skip_ratio=True,   # 仅在这里拿统计；计时时不要开
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        # LSE 参考（高精度，用于 sanity check）
        lse_ref2 = lse_reference_base2_gqa(q_triton, k_triton, scale)  # [HQ], base-2
        lse_max_abs = (lse_triton.float() - lse_ref2).abs().max().item()
        lse_rel = (lse_triton.float() - lse_ref2).abs().max() / (lse_ref2.abs().max().clamp_min(1e-6))
        lse_rel = lse_rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")
        print(f"LSE (base-2) diff vs FP32 ref: max_abs={lse_max_abs:.3e}, rel={lse_rel:.3e}")

        # 性能对比
        def run_triton():
            o, _ = attn_fwd_q1_b1_splitT(
                q_triton, k_triton, v_triton,
                scale=scale, BS=BS,

                thres_buf=thres_buf,
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        # break

        # if layer_idx > 0:
        #     break