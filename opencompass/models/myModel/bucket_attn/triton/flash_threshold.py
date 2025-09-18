import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def calc_qk_threshold(q: torch.Tensor, k: torch.Tensor, scale: float):
    # q: [HQ, K]
    # k: [HKV, T, K]
    HQ, K = q.shape
    HKV, T, _ = k.shape
    G = HQ // HKV
    k0 = k[:, :4, :]              # 前4个 key
    k1 = k[:, -32:, :]            # 后32个 key
    k_cat = torch.cat([k0, k1], dim=1)  # [HKV, 36, K]

    # 扩展为 [HQ, 36, K]
    k_cat_gqa = k_cat.repeat_interleave(G, dim=0)  # 每个 query head 对应一组 key

    q_expand = q.unsqueeze(1)                      # [HQ, 1, K]
    dot = (q_expand * k_cat_gqa).sum(dim=-1)       # [HQ, 36]
    max_val = dot.max(dim=-1).values               # [HQ]
    threshold = max_val
    threshold = threshold * scale
    threshold = threshold - 5
    return threshold.contiguous()  # [HQ]


@triton.jit
def attn_fwd_stage1(
    q,           # [HQ, K]
    k,           # [HKV, T, K]
    v,           # [T, HKV, V]
    m_buf,       # [HQ, NTB]
    l_buf,       # [HQ, NTB]
    o_buf,       # [HQ, NTB, V] (fp32)
    scale,       # float
    T,           # int
    NTB,         # int
    qk_thresholds,  # [HQ], 与 q 同设备，dtype 任意可转 fp32
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,     # 每个 KV 头对应的 Q 头个数（组大小）
    BS: tl.constexpr,    # 要求: % 16 == 0
    BK: tl.constexpr,    # 要求: % 16 == 0
    BV: tl.constexpr,    # 要求: % 16 == 0
    BM_DOT: tl.constexpr = 16,  # dot 的 M 维补到 16
):
    # 网格: (pid_hkv, pid_tb)
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)

    base_hq = pid_hkv * G

    # 时间块
    s0 = pid_tb * BS
    offs_t = s0 + tl.arange(0, BS)
    t_mask = offs_t < T

    # 行补到 16
    rows = tl.arange(0, BM_DOT)          # [0..15]
    row_mask = rows < G                  # 只前 G 行有效（真实 Q 行）

    # 1) b_s = Q·K（[BM_DOT, BK] @ [BK, BS]）
    b_s = tl.zeros([BM_DOT, BS], tl.float32)

    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        k_mask = offs_k < K

        # Q tile: [BM_DOT, BK]
        q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
        q_tile = tl.load(
            q_ptrs,
            mask=(row_mask[:, None] & k_mask[None, :]),
            other=0.0
        ).to(tl.float16)

        # K tile: [BK, BS]
        k_ptrs = k + pid_hkv * T * K + (offs_t[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(
            k_ptrs,
            mask=(k_mask[:, None] & t_mask[None, :]),
            other=0.0
        ).to(tl.float16)

        b_s += tl.dot(q_tile, k_tile, out_dtype=tl.float32)

    # 以 2 为底的指数域
    RCP_LN2 = 1.4426950408889634
    b_s = b_s * scale * RCP_LN2  # [BM_DOT, BS], base-2 logits

    # 阈值: 传入的是 (q·k)*scale - 5 (base-e)，这里换成 base-2
    thr_rows = tl.load(
        qk_thresholds + (base_hq + rows),
        mask=row_mask,
        other=float('inf'),
    ).to(tl.float32)                        # [BM_DOT]
    thr_rows_2 = thr_rows * RCP_LN2         # [BM_DOT] base-2

    # 计算该时间位置是否需要加载 v（任一有效行超过各自阈值即可）
    cmp = b_s >= thr_rows_2[:, None]        # [BM_DOT, BS] bool
    cmp = tl.where(row_mask[:, None], cmp, False)  # 只考虑有效行
    keep_any = tl.max(tl.where(cmp, 1, 0), axis=0) # [BS] int32
    keep_t = (keep_any > 0) & t_mask                 # [BS] bool

    # 计算每行块内最大值与分母（注意：这里仍使用全部 t；如果想“硬剪枝”，把 keep_t 并入 mask）
    NEG_INF = float('-inf')
    # b_s_act = tl.where(t_mask[None, :], b_s, NEG_INF)  # [BM_DOT, BS]
    # m_rows = tl.max(b_s_act, axis=1)                   # [BM_DOT]

    # 若希望“硬剪枝”分母一起裁掉，请用下面两行替换上面两行：
    valid_mask = (keep_t)[None, :]
    b_s_act = tl.where(valid_mask, b_s, NEG_INF); m_rows = tl.max(b_s_act, axis=1)

    # b_p = tl.where(t_mask[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)  # [BM_DOT, BS]
    # l_rows = tl.sum(b_p, axis=1)                                          # [BM_DOT]

    # 若使用“硬剪枝”，改为：
    b_p = tl.where(valid_mask, tl.exp2(b_s - m_rows[:, None]), 0.0)
    l_rows = tl.sum(b_p, axis=1)

    # 写回 m/l（只写前 G 行）
    m_ptrs = m_buf + (base_hq + rows) * NTB + pid_tb
    tl.store(m_ptrs, m_rows, mask=row_mask)

    l_ptrs = l_buf + (base_hq + rows) * NTB + pid_tb
    tl.store(l_ptrs, l_rows, mask=row_mask)

    # 该时间块是否需要加载 v（至少有一个 t 被保留）
    need_v = tl.sum(keep_t.to(tl.int32)) > 0

    # 3) 分子：o_b = b_p · V（[BM_DOT, BS] @ [BS, BV]）
    for v0 in range(0, V, BV):
        v_offs = v0 + tl.arange(0, BV)
        v_mask = v_offs < V

        o_tile = tl.zeros([BM_DOT, BV], tl.float32)
        if need_v:
            # 只对 keep_t=True 的时间位置加载 V
            v_ptrs = v + (offs_t[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v = tl.load(
                v_ptrs,
                mask=(keep_t[:, None] & v_mask[None, :]),
                other=0.0
            ).to(tl.float16)  # [BS, BV]

            # 对被裁掉的 t，b_v 为 0，从而其对 o_tile 的贡献为 0
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

        # 只写前 G 行
        o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTB * V) + pid_tb * V + v_offs[None, :]
        tl.store(o_ptrs, o_tile, mask=(row_mask[:, None] & v_mask[None, :]))

@triton.jit
def attn_fwd_stage2(
    m_buf,       # [HQ, NTB]
    l_buf,       # [HQ, NTB]
    o_buf,       # [HQ, NTB, V], fp32
    o,           # [HQ, V], out dtype = q.dtype
    lse,         # [HQ], fp32
    NTB,         # int
    HQ: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    # 网格: (pid_v, pid_hq)
    pid_v = tl.program_id(0)
    pid_hq = tl.program_id(1)

    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    # 在线合并 across tb（base-2 域）
    b_m = tl.full((), float('-inf'), tl.float32)  # 当前全局最大值
    b_acc = tl.zeros((), tl.float32)              # 当前全局分母
    b_o = tl.zeros([BV], tl.float32)              # 当前全局分子向量

    # 沿 tb 做稳定合并
    for tb in range(0, NTB):
        # 读取该块的统计量
        m_b = tl.load(m_buf + pid_hq * NTB + tb)
        l_b = tl.load(l_buf + pid_hq * NTB + tb)

        # 是否为非空块（有贡献）
        has = l_b > 0.0

        # 只在 has=True 时读分子；否则用 0
        o_b = tl.load(
            o_buf + pid_hq * (NTB * V) + tb * V + v_offs,
            mask=(v_mask & has),
            other=0.0
        )

        # 对空块将 m_b 视为 -inf，使其对全局缩放“无操作”
        m_b_eff = tl.where(has, m_b, tl.full((), float('-inf'), tl.float32))

        new_m = tl.maximum(b_m, m_b_eff)
        r_prev = tl.exp2(b_m - new_m)
        r_blk  = tl.where(has, tl.exp2(m_b - new_m), 0.0)

        b_acc = b_acc * r_prev + l_b * r_blk
        b_o   = b_o   * r_prev + o_b * r_blk
        b_m   = new_m

    # 归一化与 lse（base-2）
    out_tile = b_o / b_acc
    if pid_v == 0:
        lse_val = b_m + tl.log2(b_acc)
        tl.store(lse + pid_hq, lse_val)

    # 写回输出
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=v_mask)


def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    BS: int = 128,
    BK: int = 64,
    BV: int = 64,
    qk_thresholds: torch.Tensor = None,
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

    NTB = triton.cdiv(T, BS)  # 时间维被分成 NTB 个分块

    # 输出
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    # 中间缓冲（stage1 -> stage2）
    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    # 阈值：若未传入则内部计算；建议外部预先计算并传入以避免计时包含这一步
    if qk_thresholds is None:
        qk_thresholds = calc_qk_threshold(q, k, scale).contiguous()
    else:
        assert qk_thresholds.shape == (HQ,)
        assert qk_thresholds.device == q.device

    grid1 = (HKV, NTB)
    attn_fwd_stage1[grid1](
        q, k, v,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        qk_thresholds,          # 新增参数
        HKV=HKV, HQ=HQ, K=K, V=V, G=G,
        BS=BS, BK=BK, BV=BV,
        # num_warps=4,
        # num_stages=3,
    )

    # Stage 2: 跨时间分块做稳定合并 + 归一化
    grid2 = (triton.cdiv(V, BV), HQ)
    attn_fwd_stage2[grid2](
        m_buf, l_buf, o_buf,
        o, lse, NTB,
        HQ=HQ, V=V, BV=BV,
        # num_warps=4,
        # num_stages=3,
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

    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16  # 建议 fp16/bf16 才能触发 Flash
    BS, BK, BV = 256, 64, 64  # Triton tile，可按需调参

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

        print(f"{T=} {Hq=} {Hkv=} {D=} {Dv=}")

        # 准备给 Triton 内核的布局（支持 GQA）
        q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)

        # 运行 Triton 实现
        scale = 1.0 / math.sqrt(D)

        # 关键：预计算阈值（不参与时间计算）
        qk_thresholds = calc_qk_threshold(q_triton, k_triton, scale).contiguous()

        o_triton, lse_triton = attn_fwd_q1_b1_splitT(
            q_triton, k_triton, v_triton,
            scale=scale, BS=BS, BK=BK, BV=BV,
            qk_thresholds=qk_thresholds,   # 传入预计算阈值
        )  # o:[HQ,V], lse:[HQ] (以 2 为底)

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

        # 性能对比：计时不包含阈值计算
        def run_triton():
            o, _ = attn_fwd_q1_b1_splitT(
                q_triton, k_triton, v_triton,
                scale=scale, BS=BS, BK=BK, BV=BV,
                qk_thresholds=qk_thresholds,  # 使用已计算好的阈值
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")