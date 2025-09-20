# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# 单查询（B=1, qlen=1）的 GQA 注意力 Triton 实现（两阶段）
# - Stage 1：沿 KV 头和时间块（tb）并行；在每个时间大块（BS）里继续在序列维做更小子块（BSI）for-loop，
#             在线累积 base-2 域的 m/l/o 统计量，写入中间缓冲区。
# - Stage 2：沿 tb 维度做稳定的在线合并，得到最终输出以及以 2 为底的 lse（对数和）。
#
# 张量布局（contiguous）：
# - q: [HQ, K]
# - k: [HKV, T, K]
# - v: [T, HKV, V]
# 其中 HQ = G * HKV（G 为每个 KV 头对应的 Q 头数量，GQA 组大小）。
# ------------------------------------------------------------------------------

import math
import os
from tqdm import tqdm

import torch
import triton
import triton.language as tl

# ------------------------------- Stage 1 --------------------------------------
# 仅调优 BSI、num_warps、num_stages；BM_DOT 固定为 16
# STAGE1_AUTOTUNE_CONFIGS = [
#     triton.Config({'BSI': 64},  num_warps=4, num_stages=3),
#     triton.Config({'BSI': 128},  num_warps=4, num_stages=3),
#     triton.Config({'BSI': 128},  num_warps=8, num_stages=3),
#     triton.Config({'BSI': 256}, num_warps=8, num_stages=3),
#     triton.Config({'BSI': 256}, num_warps=8, num_stages=4),
# ]

# # key 选择：与性能强相关但不改变 grid 的运行时参数；BS 参与 key（虽然不调优）
# @triton.autotune(configs=STAGE1_AUTOTUNE_CONFIGS, key=['T', 'K', 'V', 'G', 'BS'])
@triton.jit
def attn_fwd_stage1(
    q,       # [HQ, K]
    k,       # [HKV, T, K]
    v,       # [T, HKV, V]
    m_buf,   # [HQ, NTB]
    l_buf,   # [HQ, NTB]
    o_buf,   # [HQ, NTB, V] (fp32)
    scale: tl.constexpr,   # float
    T,       # int
    NTB,     # int
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,       # 每个 KV 头对应的 Q 头个数（组大小）
    BS: tl.constexpr,      # 时间大块大小（Python 侧固定）
    BSI: tl.constexpr = 64,# 自动调优维度：时间子块大小（<= BS）
    BM_DOT: tl.constexpr = 16,  # 固定为 16
):
    # 保证组大小不超过 16；同时固定 BM_DOT == 16
    tl.static_assert(BM_DOT == 16, "BM_DOT is fixed to 16.")
    tl.static_assert(G <= BM_DOT, "G (group size) must be <= 16.")

    # 网格: (pid_hkv, pid_tb)
    pid_hkv = tl.program_id(0)
    pid_tb = tl.program_id(1)

    base_hq = pid_hkv * G

    # 时间大块起点
    s0 = pid_tb * BS

    # 行补到 BM_DOT
    rows = tl.arange(0, BM_DOT)    # [0..15]
    row_mask = rows < G            # 只前 G 行有效（真实 Q 行）

    # 预取 Q tile: [BM_DOT, K]（一次加载，后续复用）
    offs_k = tl.arange(0, K)
    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=(row_mask[:, None]), other=0.0).to(tl.float16)  # [BM_DOT, K]

    # 常量
    RCP_LN2 = 1.4426950408889634  # 1 / ln(2)
    NEG_INF = float('-inf')

    # 在线累计器（整个 BS 大块的统计量），base-2 域
    m_rows = tl.full([BM_DOT], NEG_INF, tl.float32)  # [BM_DOT]
    l_rows = tl.zeros([BM_DOT], tl.float32)          # [BM_DOT]
    o_acc  = tl.zeros([BM_DOT, V], tl.float32)       # [BM_DOT, V]

    v_offs = tl.arange(0, V)

    # 子块循环：t0 = 0..BS-1，步长 BSI
    for t0 in range(0, BS, BSI):
        rel = t0 + tl.arange(0, BSI)                 # [BSI]
        rel_mask = rel < BS
        offs_ti = s0 + rel                           # 绝对时间 [BSI]
        ti_mask = rel_mask & (offs_ti < T)           # [BSI]

        # ---- Q·K^T： [BM_DOT, K] @ [K, BSI] ----
        k_ptrs_i = k + pid_hkv * T * K + (offs_ti[None, :] * K) + offs_k[:, None]
        # mask_k = ti_mask[None, :].repeat(K, 1)
        mask_k = (tl.full([K], True, tl.int1))[:, None] & ti_mask[None, :]
        # mask_k = mask_k.to(tl.int1)
        k_tile_i = tl.load(k_ptrs_i, mask=mask_k, other=0.0).to(tl.float16)  # [K, BSI]

        b_s_i = tl.dot(q_tile, k_tile_i, out_dtype=tl.float32)               # [BM_DOT, BSI]
        b_s_i = b_s_i * (scale * RCP_LN2)

        # 无效时间与补行置为 -inf
        b_s_i_act = tl.where(ti_mask[None, :], b_s_i, NEG_INF)               # [BM_DOT, BSI]
        b_s_i_act = tl.where(row_mask[:, None], b_s_i_act, NEG_INF)

        # 子块最大值与概率和
        m_i = tl.max(b_s_i_act, axis=1)                                      # [BM_DOT]
        b_p_i = tl.where(ti_mask[None, :], tl.exp2(b_s_i_act - m_i[:, None]), 0.0)  # [BM_DOT, BSI]
        l_i = tl.sum(b_p_i, axis=1)                                          # [BM_DOT]

        # ---- 概率加权 V：o_i = b_p_i @ V_i ----
        v_ptrs_i = v + (offs_ti[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
        b_v_i = tl.load(v_ptrs_i, mask=(ti_mask[:, None]), other=0.0).to(tl.float16)  # [BSI, V]
        o_i = tl.dot(b_p_i.to(tl.float16), b_v_i, out_dtype=tl.float32)               # [BM_DOT, V]

        # ---- 在线合并（base-2 域）----
        new_m = tl.maximum(m_rows, m_i)              # [BM_DOT]
        r_prev = tl.exp2(m_rows - new_m)             # [BM_DOT]
        r_blk  = tl.exp2(m_i - new_m)                # [BM_DOT]

        l_rows = l_rows * r_prev + l_i * r_blk       # [BM_DOT]
        o_acc  = o_acc  * r_prev[:, None] + o_i * r_blk[:, None]  # [BM_DOT, V]
        m_rows = new_m                               # [BM_DOT]

    # ---- 写回该 BS 大块的统计量（只写前 G 行）----
    m_ptrs = m_buf + (base_hq + rows) * NTB + pid_tb
    tl.store(m_ptrs, m_rows, mask=row_mask)

    l_ptrs = l_buf + (base_hq + rows) * NTB + pid_tb
    tl.store(l_ptrs, l_rows, mask=row_mask)

    o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTB * V) + pid_tb * V + v_offs[None, :]
    tl.store(o_ptrs, o_acc, mask=(row_mask[:, None]))


# ------------------------------- Stage 2 --------------------------------------
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
):
    # 网格: (pid_dummy=0, pid_hq)
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


# ------------------------------- Python 包装 -----------------------------------
def attn_fwd_q1_b1_splitT(
    q: torch.Tensor,  # [HQ, K], fp16/bf16/fp32
    k: torch.Tensor,  # [HKV, T, K], same dtype as q
    v: torch.Tensor,  # [T, HKV, V], same dtype as q
    scale: float = None,
    BS: int = 1024,    # 时间大块大小（影响 grid 与中间缓冲，不在 autotune 中变化）
    BSI: int = 256,    # 仅作为日志提示；实际由 autotune 选择
):
    """
    返回: (o, lse)
    o: [HQ, V], dtype = q.dtype
    lse: [HQ], dtype = fp32，底数为 2
    说明：
    - Stage 1 使用 autotune 搜索 (BSI, num_warps, num_stages)；
    - BS 由 Python 侧固定；BM_DOT 固定 16。
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 2 and k.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k.shape
    Tv, HKV2, V = v.shape
    assert Kk == K and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0, "GQA 需要 HQ 是 HKV 的整数倍"
    G = HQ // HKV
    assert G <= 16, "BM_DOT 固定为 16，要求 G<=16。"

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    NTB = triton.cdiv(T, BS)

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)
    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    # Stage 1（autotuned）
    grid1 = (HKV, NTB)
    attn_fwd_stage1[grid1](
        q, k, v,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G,
        BS=BS, BSI=BSI,
        # num_warps=8, num_stages=3
    )

    # Stage 2
    grid2 = (1, HQ)
    attn_fwd_stage2[grid2](
        m_buf, l_buf, o_buf,
        o, lse, NTB,
        HQ=HQ, V=V,
    )
    return o, lse


# ------------------------------- 布局与基准 -----------------------------------
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

    q_triton = q_rope_1[0, :, 0, :].contiguous()            # [HQ, D]
    k_triton = k_rope[0, :, :, :].contiguous()              # [HKV, T, D]
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()  # [T, HKV, Dv]
    return q_triton, k_triton, v_triton


def flash_compute(q_rope_1, k_rope, v):
    # 可选参考实现：FlashAttention（需要安装 flash-attn）
    from flash_attn import flash_attn_func
    out = flash_attn_func(
        q_rope_1.transpose(1, 2),
        k_rope.transpose(1, 2),
        v.transpose(1, 2),
        causal=False,
    )
    out = out.squeeze(0).squeeze(0)  # [H, Dv]
    return out


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
    BS = 256
    BSI = 256
    print(f"{BS=} {BSI=}")

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
        assert (Hq // Hkv) <= 16, "BM_DOT 固定为 16，要求 G<=16。"

        print(f"{T=} {Hq=} {Hkv=} {D=} {Dv=}")

        # 准备给 Triton 内核的布局（支持 GQA）
        q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)

        # 运行 Triton 实现
        scale = 1.0 / math.sqrt(D)

        o_triton, lse_triton = attn_fwd_q1_b1_splitT(
            q_triton, k_triton, v_triton,
            scale=scale, 
            BS=BS,
            BSI=BSI,
        )

        # 与 FlashAttention 输出做值对比（可选）
        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

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

        break