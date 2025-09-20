# 不包含计算最大值的时间

import math
import os
from tqdm import tqdm

import torch
import triton
import triton.language as tl


@triton.jit
def attn_fwd_stage1_pruned(
    q, k, v,
    m_buf, l_buf, o_buf,
    thres_buf,       # [HQ], fp32
    scale, T, NTB,   # ints / float
    delta,           # fp32：阈值 = max - delta
    # compile-time switches
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, BM_DOT: tl.constexpr = 16,
    FIRST_LAST_PASS: tl.constexpr = False,  # 首尾两块 pass
    ENABLE_PRUNE: tl.constexpr = True,      # 是否剪枝
    UPDATE_THRES: tl.constexpr = False,     # 是否把 (max - delta) 写入 thres_buf
    TB_OFFSET: tl.constexpr = 0,            # 遍历块起点（第二次 pass 用 1）
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)

    # 选择要处理的时间块:
    # - FIRST_LAST_PASS=True 时，n_y=2，对应 tb = 0 与 tb = NTB-1
    # - 否则 tb = TB_OFFSET + pid_tb
    tb = TB_OFFSET + pid_tb
    if FIRST_LAST_PASS:
        # pid_tb ∈ {0,1}，映射到 {0, NTB-1}
        tb = pid_tb * (NTB - 1)

    base_hq = pid_hkv * G
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G

    offs_k = tl.arange(0, K)
    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    s0 = tb * BS
    offs_t = s0 + tl.arange(0, BS)
    t_mask = offs_t < T

    k_ptrs = k + pid_hkv * T * K + (offs_t[None, :] * K) + offs_k[:, None]
    k_tile = tl.load(k_ptrs, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask[None, :]), other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")

    # qk/log2
    b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
    b_s_act = tl.where(t_mask[None, :], b_s, NEG_INF)

    # 每行在该块的最大值
    m_rows_blk = tl.max(b_s_act, axis=1)

    # 是否更新阈值：阈值 = max(first,last) - delta（用原子 max 聚合两个块）
    if UPDATE_THRES:
        th_update = m_rows_blk - delta
        tl.atomic_max(thres_buf + (base_hq + rows), th_update, mask=row_mask)

    # 剪枝判定（第一次 pass 关闭剪枝）
    prune_blk = False
    if ENABLE_PRUNE:
        th_rows = tl.load(thres_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)
        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

    v_offs = tl.arange(0, V)
    o_tile = tl.zeros([BM_DOT, V], tl.float32)
    m_rows_store = tl.full([BM_DOT], NEG_INF, tl.float32)
    l_rows = tl.zeros([BM_DOT], tl.float32)

    if not prune_blk:
        # 标准 Softmax-归一化的分块统计
        m_rows = m_rows_blk
        b_p = tl.where(t_mask[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
        l_rows = tl.sum(b_p, axis=1)
        m_rows_store = m_rows

        need_v = tl.sum(t_mask.to(tl.int32), axis=0) > 0
        if need_v:
            v_ptrs = v + (offs_t[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0).to(tl.float16)
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

    # 写回中间结果
    m_ptrs = m_buf + (base_hq + rows) * NTB + tb
    tl.store(m_ptrs, m_rows_store, mask=row_mask)

    l_ptrs = l_buf + (base_hq + rows) * NTB + tb
    tl.store(l_ptrs, l_rows, mask=row_mask)

    o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTB * V) + tb * V + v_offs[None, :]
    tl.store(o_ptrs, o_tile, mask=row_mask[:, None])


# stage2 原样
@triton.jit
def attn_fwd_stage2(
    m_buf, l_buf, o_buf, o, lse, NTB,
    HQ: tl.constexpr, V: tl.constexpr,
):
    pid_hq = tl.program_id(1)
    v_offs = tl.arange(0, V)
    b_m = tl.full((), float('-inf'), tl.float32)
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)

    for tb in range(0, NTB):
        m_b = tl.load(m_buf + pid_hq * NTB + tb)
        l_b = tl.load(l_buf + pid_hq * NTB + tb)
        has = l_b > 0.0

        o_b = tl.load(o_buf + pid_hq * (NTB * V) + tb * V + v_offs, mask=has, other=0.0)
        m_b_eff = tl.where(has, m_b, tl.full((), float('-inf'), tl.float32))

        new_m = tl.maximum(b_m, m_b_eff)
        r_prev = tl.exp2(b_m - new_m)
        r_blk  = tl.where(has, tl.exp2(m_b - new_m), 0.0)

        b_acc = b_acc * r_prev + l_b * r_blk
        b_o   = b_o   * r_prev + o_b * r_blk
        b_m   = new_m

    out_tile = b_o / b_acc
    lse_val = b_m + tl.log2(b_acc)
    tl.store(lse + pid_hq, lse_val)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,  # [HQ, K]
    k: torch.Tensor,  # [HKV, T, K]
    v: torch.Tensor,  # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    delta: float = 3.0,
):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 2 and k.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k.shape
    Tv, HKV2, V = v.shape
    assert Kk == K and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    NTB = triton.cdiv(T, BS)

    # 输出与中间缓冲
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    # 阈值缓冲：先置为 -inf
    thres_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)
    thres_buf.fill_(-float('inf'))

    # 1) Stage1 首次 launch：只跑首尾两块，正常计算 + 更新阈值
    grid_first = (HKV, 1 if NTB == 1 else 2)
    attn_fwd_stage1_pruned[grid_first](
        q, k, v,
        m_buf, l_buf, o_buf,
        thres_buf,
        scale, T, NTB,
        delta,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS,
        # 下面这些是 constexpr 参数
        FIRST_LAST_PASS=True,
        ENABLE_PRUNE=False,   # 首次不剪枝，确保首尾块一定被计算
        UPDATE_THRES=True,    # 把(max - delta)聚合到 thres_buf
        TB_OFFSET=0,
        # num_warps=4, num_stages=3,
    )

    # 2) Stage1 第二次 launch：遍历剩余块（tb=1..NTB-2），启用剪枝
    if NTB > 2:
        grid_rest = (HKV, NTB - 2)
        attn_fwd_stage1_pruned[grid_rest](
            q, k, v,
            m_buf, l_buf, o_buf,
            thres_buf,
            scale, T, NTB,
            delta,
            HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS,
            FIRST_LAST_PASS=False,
            ENABLE_PRUNE=True,
            UPDATE_THRES=False,
            TB_OFFSET=1,        # 从 tb=1 开始
            # num_warps=4, num_stages=3,
        )

    # 3) Stage2 合并
    grid2 = (1, HQ)
    attn_fwd_stage2[grid2](
        m_buf, l_buf, o_buf,
        o, lse, NTB,
        HQ=HQ, V=V,
        # num_warps=4, num_stages=3,
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

    # exp_root = os.path.join(exp_root_dir, 'Llama-3_2-3B/longbench_narrativeqa_42')
    exp_root = os.path.join(exp_root_dir, 'Llama-3_2-3B/longbench_gov_report_46')
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256 # 只保留时间分块
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

        # 运行 Triton 实现
        scale = 1.0 / math.sqrt(D)

        o_triton, lse_triton = attn_fwd_q1_b1_splitT(
            q_triton, k_triton, v_triton,
            scale=scale, BS=BS,
        )

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