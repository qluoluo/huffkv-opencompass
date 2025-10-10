import os
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache")
# os.environ['TRITON_DUMP_ASSEMBLY'] = "1"

import math
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


# 说明（总体语义与张量布局）
# - 本 kernel 处理注意力前向（stage1，子块/裁剪版），以块（TB）+ 子块（SB）的方式扫描序列维度 T。
# - 网格维度：pid_hkv ∈ [0, HKV)（对应 K/V 头维），pid_tb ∈ [0, NTB)（对应大块 TB）
# - 每个 pid_hkv 一次性处理 G 行 Q（因此 HQ 必须满足 HQ = HKV * G），每个程序块取 BM_DOT 行（BM_DOT ≥ G），
#   多余的行由 row_mask 屏蔽。
# - 子块大小 SBS，TB 大小 BS，子块个数 NSB = ceil(BS / SBS)，NTBS = NTB * NSB（由外部传入，用于输出/掩码缓冲的步幅）
# - softmax 使用以 2 为底（exp2）的形式：softmax(x) = exp2(x * RCP_LN2)（数值等价）
#
# 各输入/输出和缓冲区的 shape 与内存布局（连续方向为最后一维）：
# - q:       [HQ, K]，行优先，连续维是 K；HQ = HKV * G。
# - k:       [HKV, T, K]，最后一维 K 连续（在 USE_FP8_K=False 时使用）。
# - k_bytes: [HKV, T, K]，dtype=float8_e5m2（仅加载 K 的“高 8 位”表征），最后一维 K 连续（在 USE_FP8_K=True 时使用）。
# - v:       [T, HKV, V]，最后一维 V 连续。
# - m_buf:   [HQ, NTBS]，行优先，连续维是 NTBS（每个 Q 行 × 所有(tb,sb)的线性索引）。
# - l_buf:   [HQ, NTBS]，行优先，连续维是 NTBS（对应 softmax 的分母，按子块内规范化）。
# - o_buf:   [HQ, NTBS, V]，最后一维 V 连续，其次是 NTBS（步幅 NTBS*V）。
# - thres_buf: [HQ]，每行 Q 的阈值（用于裁剪判断）。
# - mask_buf:  [HKV, NTBS]，标记(t b, s b)是否有效（保留=1，裁剪=0；裁剪时本 kernel 不写入，默认由外部清零）。
#
# 其他标量/常量：
# - scale: 缩放系数（通常是 1 / sqrt(K)）
# - T: 序列长度（time dimension）
# - NTB: TB 的个数（沿 T 方向的大块数）
# - NTBS: NTB * NSB（所有(tb,sb)对子块的总数；用作输出缓冲步幅）
# - HKV: K/V 头的数量（编译期常量）
# - HQ:  Q 头的“行”数量（编译期常量，HQ = HKV * G）
# - K:   特征维（d_k）
# - V:   特征维（d_v）
# - G:   每个 KV 头所对应的 Q 行数（一个程序块处理的有效行数）
# - BS:  TB 的长度（沿 T 方向的大块大小）
# - SBS: 子块长度（沿 T 方向的小块大小）
# - BM_DOT: 每个程序块内部用于 dot 的行 tile 尺寸（BM_DOT ≥ G，超出部分由 row_mask 屏蔽）
# - USE_FP8_K: 是否从 k_bytes 读取（float8_e5m2），否则从 k（float16）读取

@triton.jit
def attn_decode_stage1_pruned(
    q, k, k_bytes, v,                # 新增：k_bytes (fp8_e5m2*)
    m_buf, l_buf, o_buf,
    thres_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    # BM_DOT: tl.constexpr = 32,
    USE_FP8_K: tl.constexpr = True,  # 是否仅加载 k 的高 8 位（fp8_e5m2）
):
    pid_hkv = tl.program_id(0)  # 标量，int32，取值范围 [0, HKV)
    pid_tb  = tl.program_id(1)  # 标量，int32，取值范围 [0, NTB)
    base_hq = pid_hkv * G       # 标量，int32；该 KV 头对应的 Q 行起始索引（在 [0, HQ)）

    # 本 TB 的起始全局 T 偏移（标量）
    s0 = pid_tb * BS

    # 行/列常量与 Q tile
    rows     = tl.arange(0, BM_DOT)                 # [BM_DOT]，int32，当前程序块内的行索引（局部）
    row_mask = rows < G                             # [BM_DOT]，bool，前 G 行有效，其余行（若 BM_DOT>G）被屏蔽

    offs_k   = tl.arange(0, K)                      # [K]，int32，K 维上的列偏移
    # q_ptrs 指向一个 [BM_DOT, K] 的 tile（行是 base_hq+rows，列是 offs_k）
    # 内存布局：q 是 [HQ, K] 行优先（K 连续）
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    # if not USE_FP8_K:
    #     q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)
    # else:
    #     q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float8e5)

    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)  # [BM_DOT, K]，fp16

    # 常量
    RCP_LN2 = 1.4426950408889634                 # 标量，float64，= 1 / ln(2)
    NEG_INF = float("-inf")                      # 标量，float64，用于屏蔽无效位置
    TRUE_K  = tl.full([K], True, tl.int1)        # [K]，bool，全 True，用于便捷构造掩码

    # NSB 为编译期常量：每个大块内的子块数（= ceil(BS / SBS)）
    NSB: tl.constexpr = (BS + SBS - 1) // SBS    # 标量，constexpr

    # 循环每个子块
    for sb in tl.static_range(NSB):              # sb: 编译期静态循环，范围 [0, NSB)
        # 子块内的 T 偏移与 mask
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)  # [SBS]，int32，当前子块覆盖的 T 索引
        t_mask_sb = offs_t_sb < T                       # [SBS]，bool，越界屏蔽（最后一块可能不满）

        # K 路：根据开关选择 FP8 高字节还是 FP16
        if USE_FP8_K:
            # k_bytes: [HKV, T, K], dtype = float8_e5m2（仅高 8 位表示）
            # kb_ptrs 对应一个 [K, SBS] 的 tile（列主视角，仅指针算式；真实 load 结果转置到 [K, SBS]）
            # 内存访问：先选中第 pid_hkv 个头，再按 T * K 展开，最终取 K 维为最快维
            kb_ptrs = k_bytes + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
            k_tile = tl.load(
                kb_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),  # [K, SBS]，bool
                other=0.0
            # )
            ).to(tl.float16)  # k_tile: [K, SBS]，fp16（由 fp8_e5m2 解码到 fp16）
        else:
            # 原 FP16 路
            # k_ptrs 指向 [K, SBS] 的 tile，布局同上
            k_ptrs = k + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
            k_tile = tl.load(
                k_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),  # [K, SBS]
                other=0.0
            ).to(tl.float16)  # [K, SBS]，fp16

        # 注意：对数底为 2 的 softmax 形式
        # q_tile: [BM_DOT, K]，k_tile: [K, SBS] -> b_s: [BM_DOT, SBS]（fp32 累加）
        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2  # [BM_DOT, SBS]
        # 对越界 T 位置填充 -inf
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)                      # [BM_DOT, SBS]

        # 子块内行最大值（按列 axis=1 归约）
        m_rows_blk = tl.max(b_s_act, axis=1)  # [BM_DOT]，每行在该子块上的最大值（用于数稳 softmax）

        # 阈值与裁剪
        # 加载阈值（无效行用 -inf 占位）
        th_rows = tl.load(thres_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)  # [BM_DOT]

        # 统计满足 m_rows_blk >= th_rows 的有效行数；若为 0 则整块裁剪
        n_keep    = tl.sum(((m_rows_blk >= th_rows) & row_mask).to(tl.int32), axis=0)  # 标量，int32
        prune_blk = n_keep == 0

        # 写入索引：将子块映射为全局块 idx
        tb_sb = pid_tb * NSB + sb  # 标量，int32，范围 [0, NTBS)
        v_offs = tl.arange(0, V)   # [V]，int32，V 维偏移

        if not prune_blk:
            # softmax 规范化（子块内）
            m_rows = m_rows_blk                                                     # [BM_DOT]
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)  # [BM_DOT, SBS]，fp32
            l_rows = tl.sum(b_p, axis=1)                                            # [BM_DOT]，softmax 分母（子块内）

            # V 路
            # v: [T, HKV, V]，V 连续
            v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
            b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)   # [SBS, V]，fp16
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)                 # [BM_DOT, V]

            # 写回（注意 stride 里的 NTBS）
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)           # 写入每行的 m（数稳项）
            tl.store(l_ptrs, l_rows, mask=row_mask)           # 写入每行的 l（softmax 分母）
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])  # 写入部分输出（按子块累积）

            # 标记该 (tb, sb) 为有效
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_decode_stage2_masked(
    m_buf, l_buf, o_buf,     # [HQ, NTBS], [HQ, NTBS], [HQ, NTBS, V]
    mask_buf,                # [HKV, NTBS], int8（每个小块）
    o,                       # [HQ, V], out dtype = q.dtype
    NTBS,                    # int: 总块数 = NTB * NSB
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

    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_hq * (NTBS * V) + tb * V + v_offs)

            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk  = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o   = b_o   * r_prev + o_b * r_blk
            b_m   = new_m
        # 无效小块直接跳过

    # 空结果保护
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    # 现在的 mask_buf: [HKV, NTBS], int8, 1=keep, 0=skip
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()     # device 上做 sum，避免 int8 溢出
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())           # 注意：.item() 会触发同步


def attn_fwd_q1_b1_splitT(
    q_rope_1: torch.Tensor,   # [B, Hq, 1, D]
    k_rope:   torch.Tensor,   # [B, Hkv, T, D]
    v:        torch.Tensor,   # [B, Hkv, T, Dv]
    *,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    use_fp8_k_high_byte: bool = True,
    return_skip_ratio: bool = False,
):
    """
    输入/输出与 flash_compute 保持一致的 API：
      attn_fwd_q1_b1_splitT(q_rope_1, k_rope, v) -> out  (默认仅返回 out)
    其余参数作为可选 keyword 控制，默认不影响与 flash_compute 的一致性。
    """
    assert q_rope_1.is_cuda and k_rope.is_cuda and v.is_cuda
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4

    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk  = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape

    # 形状/约束校验（与原 demo 一致）
    assert B == Bk == Bv == 1, "该 kernel 仅支持 batch=1"
    assert qlen == 1,          "该 kernel 仅支持 qlen=1"
    assert Hkv == Hvv,         "k/v 的 head 数必须一致"
    assert Dq == Dk,           "q/k head_dim 不一致"
    assert T == Tv,            "k/v 的 seq_len 不一致"
    assert Hq % Hkv == 0,      "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

    # 将 layout 转为 Triton 内核所需
    q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)  # q:[HQ,K], k:[HKV,T,K], v:[T,HKV,V]
    HQ, K = q_triton.shape
    HKV, Tk, Kk = k_triton.shape
    Tv2, HKV2, V = v_triton.shape
    assert Tk == T and Tv2 == T and HKV2 == HKV and Kk == K

    G = HQ // HKV
    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    # scale
    scale = 1.0 / math.sqrt(K)

    # 块参数
    NTB  = triton.cdiv(T, BS)
    NSB  = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # 预处理1：阈值（注意这里应使用 TB 大小 BS，而非 SBS）
    thres_buf = torch.empty((HQ,), device=q_triton.device, dtype=torch.float32)
    attn_find_threshold_two_blocks[(HKV, 1)](
        q_triton, k_triton, thres_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
    )

    # 预处理2：k 的高 8 位（如需）
    if use_fp8_k_high_byte:
        k_bytes = k_triton.contiguous().view(torch.float8_e5m2)[..., 1::2].contiguous()
    else:
        k_bytes = None  # 内核里仍会传参，但分支不会用到

    # 中间缓冲
    o      = torch.empty((HQ, V), device=q_triton.device, dtype=q_triton.dtype)
    m_buf  = torch.empty((HQ, NTBS), device=q_triton.device, dtype=torch.float32)
    l_buf  = torch.empty((HQ, NTBS), device=q_triton.device, dtype=torch.float32)
    o_buf  = torch.empty((HQ, NTBS, V), device=q_triton.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q_triton.device, dtype=torch.int8)

    # Stage 1
    attn_decode_stage1_pruned[(HKV, NTB)](
        q_triton, k_triton, k_bytes if k_bytes is not None else k_triton, v_triton,
        m_buf, l_buf, o_buf,
        thres_buf, mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        USE_FP8_K=use_fp8_k_high_byte,
    )

    # 可选统计
    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2
    attn_decode_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V,
    )

    # 返回：默认与 flash_compute 一致（只返回 out）
    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o


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
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_54'
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_57'

    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256
    SBS = 256
    delta = 8.0
    # use_fp8_k_high_byte = False
    use_fp8_k_high_byte = True

    print(f"{BS=}")

    # 计时参数
    iters = 100
    warmup = 100

    # iters = 1
    # warmup = 0

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        # if layer_idx == 0:
        #     continue
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype).contiguous()  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype).contiguous()       # [B, Hkv, T, Dv]

        # 只取最后一个查询位置 -> qlen=1
        q_rope_1 = q_rope[:, :, -1:, :]  # [B, Hq, 1, D]

        # 直接调用（输入/输出与 flash_compute 一致）
        o_triton, skip_ratio = attn_fwd_q1_b1_splitT(
            q_rope_1, k_rope, v,
            BS=BS, SBS=SBS,
            delta=delta,
            use_fp8_k_high_byte=use_fp8_k_high_byte,
            return_skip_ratio=True,   # 如需统计可打开；对齐 flash 行为时设为 False
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, Dv]

        # 数值对比
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton():
            return attn_fwd_q1_b1_splitT(
                q_rope_1, k_rope, v,
                BS=BS, SBS=SBS,
                use_fp8_k_high_byte=use_fp8_k_high_byte,
                return_skip_ratio=False,  # 计时时关闭
            )

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        # break·