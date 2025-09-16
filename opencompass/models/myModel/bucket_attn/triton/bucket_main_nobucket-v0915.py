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
    threshold = threshold - 10
    return threshold.contiguous()  # [HQ]


@triton.jit
def attn_fwd_q1_b1_stage1(
    q,           # [HQ, K]
    k,           # [HKV, T, K]
    v,           # [T, HKV, V]
    m_buf,       # [HQ, NTB]
    l_buf,       # [HQ, NTB]
    o_buf,       # [HQ, NTB, V]
    scale,       # float
    T,           # int
    NTB,         # int = ceil(T / BS)
    qk_thresholds,  # [HQ], 每个head一个阈值（已 scaled）
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # 网格: (pid_v, pid_hq, pid_tb)
    pid_v = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_tb = tl.program_id(2)

    i_hq = pid_hq
    i_h = i_hq // G

    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    s0 = pid_tb * BS
    offs_t = s0 + tl.arange(0, BS)
    t_mask = offs_t < T

    # 加载对应 threshold（已 scaled），并转成 fp32
    th_ptr = qk_thresholds + i_hq
    threshold = tl.load(th_ptr).to(tl.float32)

    # 块内 q·k 累加（使用 block_ptr + tl.dot）
    b_s = tl.zeros([BS], tl.float32)

    # Q 的 block_ptr：将每个 head 的 q 视为 [1, K]，沿着 K 维分块
    q_base = q + i_hq * K
    q_mat_shape = (1, K)
    q_mat_strides = (K, 1)

    # K 的 block_ptr：将 k 视为 [K, T] 的 2D 视图（固定 head），按 [BK, BS] 分块
    k_base = k + i_h * T * K + s0 * K
    # 这里取一个 2D 视图：行=K，列=BS（位于原张量 T 维的一段），行步长=1，列步长=K
    k_mat_shape = (K, BS)
    k_mat_strides = (1, K)

    for kk in range(0, K, BK):
        # Q tile: [1, BK]
        q_bp = tl.make_block_ptr(
            base=q_base,
            shape=q_mat_shape,
            strides=q_mat_strides,
            offsets=(0, kk),
            block_shape=(16, BK),
            order=(1, 0),
        )
        q_tile = tl.load(
            q_bp,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)  # [1, BK]

        # K tile: [BK, BS]
        k_bp = tl.make_block_ptr(
            base=k_base,
            shape=k_mat_shape,
            strides=k_mat_strides,
            offsets=(kk, 0),
            block_shape=(BK, BS),
            order=(1, 0),
        )
        k_tile = tl.load(
            k_bp,
            boundary_check=(0, 1),  # K 维和 T(切片) 维都做边界检查
            padding_option="zero",
        ).to(tl.float32)  # [BK, BS]

        # b_s += q_tile @ k_tile  -> [1, BS]，随后压到 [BS]
        b_s += tl.dot(q_tile, k_tile, out_dtype=tl.float32)[0, :]

    # 注意：为了与 e 底 softmax 等价，这里把分数换算到以 2 为底的指数域：
    # 2^(s / ln 2) = e^s，因此在用 exp2 前先乘以 1/ln 2
    RCP_LN2 = 1.4426950408889634
    b_s = b_s * scale * RCP_LN2  # 现在 b_s 处于“以 2 为底的对数域”

    # 判定阈值与边界
    skip = b_s < threshold               # [BS], 被阈值屏蔽
    active_t = (~skip) & t_mask          # [BS], 实际参与 softmax 的位置

    # 仅用活跃位置计算块内最大值
    NEG_INF = float('-inf')
    b_s_act = tl.where(active_t, b_s, NEG_INF)
    m_b = tl.max(b_s_act, axis=0)        # 标量（若整块无活跃，值为 -inf）

    # 计算块内分母与分子：只累加活跃位置
    # 警告：即便 m_b=-inf，下面 exp2 的无穷大也会通过 active_t 掩码归零，不会影响结果
    b_p = tl.where(active_t, tl.exp2(b_s - m_b), 0.0)  # [BS]

    # 使用 tl.dot 来求和分母 l_b
    # l_b = sum_j b_p[j]
    b_p_row = tl.reshape(b_p, (1, BS))               # [1, BS]
    ones_col = tl.full((BS, 1), 1.0, tl.float32)     # [BS, 1]
    l_b = tl.dot(b_p_row, ones_col, out_dtype=tl.float32)[0, 0]  # 标量

    # 仅当该块存在至少一个活跃位置时，才从 v 读取
    num_active = tl.sum(active_t.to(tl.int32), axis=0)
    need_v = num_active > 0

    o_b = tl.zeros([BV], tl.float32)
    if need_v:
        # V 的 block_ptr：将 v 视为 [BS, V]（沿时间分块行、V 列），再按 [BS, BV] 取块
        v_base = v + s0 * (HKV * V) + i_h * V
        v_mat_shape = (BS, V)
        v_mat_strides = (HKV * V, 1)
        v_bp = tl.make_block_ptr(
            base=v_base,
            shape=v_mat_shape,
            strides=v_mat_strides,
            offsets=(0, pid_v * BV),
            block_shape=(BS, BV),
            order=(1, 0),
        )
        b_v = tl.load(
            v_bp,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)  # [BS, BV]

        # 仅活跃时间步参与：对不活跃位置清零
        b_v = tl.where(active_t[:, None], b_v, 0.0)

        # 使用 tl.dot 代替 sum：o_b = b_p^T @ b_v -> [1, BV]
        o_tile = tl.dot(b_p_row, b_v, out_dtype=tl.float32)  # [1, BV]
        o_b = o_tile[0, :]  # [BV]

    # 无论 need_v 与否，都写回 o_buf（空块写 0），提升阶段间缓存命中
    # 用 block_ptr 存回 [1, BV] tile
    o_base = o_buf + i_hq * (NTB * V) + pid_tb * V
    o_bp = tl.make_block_ptr(
        base=o_base,
        shape=(1, V),
        strides=(V, 1),
        offsets=(0, pid_v * BV),
        block_shape=(1, BV),
        order=(1, 0),
    )
    tl.store(
        o_bp,
        tl.reshape(o_b, (1, BV)),
        boundary_check=(0, 1),
    )

    if pid_v == 0:
        tl.store(m_buf + i_hq * NTB + pid_tb, m_b)
        tl.store(l_buf + i_hq * NTB + pid_tb, l_b)


@triton.jit
def attn_fwd_q1_b1_stage2(
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

        # 读取该块的分子向量 o_b: [1, BV] tile via block_ptr
        o_base = o_buf + pid_hq * (NTB * V) + tb * V
        o_bp = tl.make_block_ptr(
            base=o_base,
            shape=(1, V),
            strides=(V, 1),
            offsets=(0, pid_v * BV),
            block_shape=(1, BV),
            order=(1, 0),
        )
        o_b_tile = tl.load(
            o_bp,
            boundary_check=(0, 1),
            padding_option="zero",
        ).to(tl.float32)  # [1, BV]
        o_b = o_b_tile[0, :]  # [BV]

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

    # 写回输出（使用 block_ptr 保存 [1, BV]）
    o_base = o + pid_hq * V
    o_bp = tl.make_block_ptr(
        base=o_base,
        shape=(1, V),
        strides=(V, 1),
        offsets=(0, pid_v * BV),
        block_shape=(1, BV),
        order=(1, 0),
    )
    tl.store(
        o_bp,
        tl.reshape(out_tile, (1, BV)).to(o_base.dtype.element_ty),
        boundary_check=(0, 1),
    )


def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,  # [HQ, K], fp16/bf16/fp32
    k: torch.Tensor,  # [HKV, T, K], same dtype as q
    v: torch.Tensor,  # [T, HKV, V], same dtype as q
    scale: float = None,
    BS: int = 128,    # 时间分块大小（可调，影响 NTB）
    BK: int = 64,
    BV: int = 64,
    qk_thresholds: torch.Tensor = None,  # 可选：预先计算的阈值 [HQ]
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
        # 请确保提供 calc_qk_threshold，或在外部传入 qk_thresholds
        # 这里假定存在该函数
        from your_module import calc_qk_threshold  # 用户需自行实现/导入
        qk_thresholds = calc_qk_threshold(q, k, scale).contiguous()
    else:
        assert qk_thresholds.shape == (HQ,), "qk_thresholds 形状应为 [HQ]"
        assert qk_thresholds.device == q.device, "qk_thresholds 需与 q 在同一设备"

    # Stage 1: 并行计算各时间分块的局部统计与分子
    grid1 = (triton.cdiv(V, BV), HQ, NTB)
    attn_fwd_q1_b1_stage1[grid1](
        q, k, v,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        qk_thresholds,  # 传入预计算/内算阈值
        HKV=HKV, HQ=HQ, K=K, V=V, G=G,
        BS=BS, BK=BK, BV=BV,
        # num_warps=4,
        # num_stages=3,
    )

    # Stage 2: 跨时间分块做稳定合并 + 归一化
    grid2 = (triton.cdiv(V, BV), HQ)
    attn_fwd_q1_b1_stage2[grid2](
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

        print(f"{T=}")

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