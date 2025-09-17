import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def fp16_bytes_view(x: torch.Tensor):
    assert x.dtype == torch.float16
    st = x.untyped_storage()
    byte_offset = x.storage_offset() * x.element_size()
    sizes   = list(x.size()) + [2]
    strides = [s * x.element_size() for s in x.stride()] + [1]
    return torch.empty(0, dtype=torch.uint8, device=x.device).set_(st, byte_offset, sizes, strides)


def fp16_from_hi_u8(hi_u8: torch.Tensor) -> torch.Tensor:
    # 给定 hi_u8（fp16 的高 8bit 视图），构造一个 fp16 张量，其低 8bit 清零，高 8bit=hi_u8
    assert hi_u8.dtype == torch.uint8
    zeros = torch.zeros_like(hi_u8, dtype=torch.uint8)
    # 拼一个 [..., 2] 的字节张量：[..., 0]=lo=0, [..., 1]=hi
    u8_2 = torch.stack([zeros, hi_u8], dim=-1).contiguous()
    # 通过 bytes 视图反向映射回 fp16
    st = u8_2.untyped_storage()
    byte_offset = u8_2.storage_offset()  # already bytes
    sizes = list(hi_u8.size())
    # u8_2 的 strides 是以元素为单位（此处是 u8），对 fp16 而言需除以 2，但由于我们用了 contiguous()，可以直接按紧凑布局 set_
    return torch.empty(0, dtype=torch.float16, device=hi_u8.device).set_(st, byte_offset, sizes, [s // 2 for s in u8_2.stride()[:-1]])


def calc_qk_threshold_u8hi(q: torch.Tensor, k_hi_u8: torch.Tensor, scale: float):
    # q: [HQ, K] (fp16/bf16/fp32)
    # k_hi_u8: [HKV, T, K] (uint8, 仅高 8bit)
    # 使用仅高 8bit 重构后的 k_fp16 来计算阈值，保证与主路径一致
    HQ, K = q.shape
    HKV, T, _ = k_hi_u8.shape
    G = HQ // HKV
    # 重构 k 的 fp16（低 8bit 清零）
    k_fp16 = fp16_from_hi_u8(k_hi_u8)  # [HKV, T, K], fp16
    k0 = k_fp16[:, :4, :]              # 前4个 key
    k1 = k_fp16[:, -32:, :]            # 后32个 key
    k_cat = torch.cat([k0, k1], dim=1)  # [HKV, 36, K]
    k_cat_gqa = k_cat.repeat_interleave(G, dim=0)  # [HQ, 36, K]

    q_expand = q.unsqueeze(1)  # [HQ, 1, K]
    dot = (q_expand * k_cat_gqa).sum(dim=-1).float()  # [HQ, 36] -> fp32
    max_val = dot.max(dim=-1).values                   # [HQ]
    threshold = max_val * scale - 3
    return threshold.contiguous()  # [HQ], fp32


@triton.jit
def attn_fwd_q1_b1_stage1_u8hi(
    q,             # [HQ, K], fp16/bf16/fp32
    k_hi,          # [HKV, T, K], uint8 (仅高 8bit)
    v_hi,          # [T, HKV, V], uint8 (仅高 8bit)
    m_buf,         # [HQ, NTB], fp32
    l_buf,         # [HQ, NTB], fp32
    o_buf,         # [HQ, NTB, V], fp32
    scale,         # float
    T,             # int
    NTB,           # int = ceil(T / BS)
    qk_thresholds, # [HQ], fp32（已 scale & 偏置）
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

    # 加载 threshold（已 scaled）
    th_ptr = qk_thresholds + i_hq
    threshold = tl.load(th_ptr).to(tl.float32)

    # 计算块内的 q·k 累加，k 来自高 8bit（uint8），在核内重构为 fp16（低字节清零）
    b_s = tl.zeros([BS], tl.float32)

    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        k_mask = offs_k < K

        # load q tile
        q_ptrs = q + i_hq * K + offs_k
        q_chunk = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BK]

        # load k_hi tile (uint8) -> reconstruct to fp16 by (u8<<8) bitcast
        k_ptrs_u8 = k_hi + i_h * T * K + (offs_t[None, :] * K) + offs_k[:, None]
        k_u8 = tl.load(
            k_ptrs_u8,
            mask=(k_mask[:, None] & t_mask[None, :]),
            other=0
        )  # [BK, BS], uint8

        k_u16 = k_u8.to(tl.uint16) << 8
        k_chunk = tl.cast(k_u16, tl.float16).to(tl.float32)  # bitcast to fp16, then widen to fp32

        b_s += tl.sum(q_chunk[:, None] * k_chunk, axis=0)  # [BS]

    # 切到 base-2 计算域
    RCP_LN2 = 1.4426950408889634
    b_s = b_s * scale * RCP_LN2  # base-2 对数域

    # 阈值筛选
    skip = b_s < threshold
    active_t = (~skip) & t_mask

    NEG_INF = float('-inf')
    b_s_act = tl.where(active_t, b_s, NEG_INF)
    m_b = tl.max(b_s_act, axis=0)

    b_p = tl.where(active_t, tl.exp2(b_s - m_b), 0.0)
    l_b = tl.sum(b_p, axis=0)

    num_active = tl.sum(active_t.to(tl.int32), axis=0)
    need_v = num_active > 0

    o_b = tl.zeros([BV], tl.float32)
    if need_v:
        # load v_hi tile (uint8) -> reconstruct to fp16 by (u8<<8) bitcast
        v_ptrs_u8 = v_hi + (offs_t[:, None] * (HKV * V)) + (i_h * V) + v_offs[None, :]
        v_u8 = tl.load(
            v_ptrs_u8,
            mask=(active_t[:, None] & v_mask[None, :]),
            other=0
        )  # [BS, BV], uint8

        v_u16 = v_u8.to(tl.uint16) << 8
        b_v = tl.cast(v_u16, tl.float16).to(tl.float32)  # bitcast -> fp16 -> fp32

        o_b = tl.sum(b_p[:, None] * b_v, axis=0)  # [BV]

    # 写回
    o_ptrs = o_buf + i_hq * (NTB * V) + pid_tb * V + v_offs
    tl.store(o_ptrs, o_b, mask=v_mask)

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

    v_offs = pid_v * BV + tl.arange(0, BV)
    v_mask = v_offs < V

    b_m = tl.full((), float('-inf'), tl.float32)
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([BV], tl.float32)

    for tb in range(0, NTB):
        m_b = tl.load(m_buf + pid_hq * NTB + tb)
        l_b = tl.load(l_buf + pid_hq * NTB + tb)
        has = l_b > 0.0

        o_b = tl.load(
            o_buf + pid_hq * (NTB * V) + tb * V + v_offs,
            mask=(v_mask & has),
            other=0.0
        )

        m_b_eff = tl.where(has, m_b, tl.full((), float('-inf'), tl.float32))
        new_m = tl.maximum(b_m, m_b_eff)
        r_prev = tl.exp2(b_m - new_m)
        r_blk  = tl.where(has, tl.exp2(m_b - new_m), 0.0)

        b_acc = b_acc * r_prev + l_b * r_blk
        b_o   = b_o   * r_prev + o_b * r_blk
        b_m   = new_m

    out_tile = b_o / b_acc
    if pid_v == 0:
        lse_val = b_m + tl.log2(b_acc)
        tl.store(lse + pid_hq, lse_val)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=v_mask)


def attn_fwd_q1_b1_splitT_u8hi(
    q: torch.Tensor,      # [HQ, K], fp16/bf16/fp32
    k_hi_u8: torch.Tensor,# [HKV, T, K], uint8
    v_hi_u8: torch.Tensor,# [T, HKV, V], uint8
    scale: float = None,
    BS: int = 128,
    BK: int = 64,
    BV: int = 64,
    qk_thresholds: torch.Tensor = None,  # [HQ], 可选
):
    assert q.is_cuda and k_hi_u8.is_cuda and v_hi_u8.is_cuda
    assert q.ndim == 2 and k_hi_u8.ndim == 3 and v_hi_u8.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k_hi_u8.shape
    Tv, HKV2, V = v_hi_u8.shape
    assert Kk == K and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0, "GQA 需要 HQ 是 HKV 的整数倍"
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    NTB = triton.cdiv(T, BS)

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    if qk_thresholds is None:
        qk_thresholds = calc_qk_threshold_u8hi(q, k_hi_u8, scale).contiguous()
    else:
        assert qk_thresholds.shape == (HQ,)
        assert qk_thresholds.device == q.device

    grid1 = (triton.cdiv(V, BV), HQ, NTB)
    attn_fwd_q1_b1_stage1_u8hi[grid1](
        q, k_hi_u8, v_hi_u8,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        qk_thresholds,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G,
        BS=BS, BK=BK, BV=BV,
        # num_warps=4,
        # num_stages=3,
    )

    grid2 = (triton.cdiv(V, BV), HQ)
    attn_fwd_q1_b1_stage2[grid2](
        m_buf, l_buf, o_buf, o, lse, NTB,
        HQ=HQ, V=V, BV=BV,
        # num_warps=4,
        # num_stages=3,
    )
    return o, lse


def to_triton_layout_q(q_rope_1: torch.Tensor):
    # q_rope_1: [B, Hq, 1, D] -> q_triton: [HQ, D]
    assert q_rope_1.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    assert B == 1 and qlen == 1
    return q_rope_1[0, :, 0, :].contiguous()  # [HQ, D]


def to_triton_layout_u8hi(k_rope: torch.Tensor, v: torch.Tensor):
    # k_rope: [B, Hkv, T, D], v: [B, Hkv, T, Dv]
    # 返回 k_hi_triton: [HKV, T, D] uint8, v_hi_triton: [T, HKV, Dv] uint8
    assert k_rope.ndim == 4 and v.ndim == 4
    B, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bv and Hkv == Hvv and T == Tv

    k_bytes = fp16_bytes_view(k_rope)  # [B, Hkv, T, Dk, 2]
    k_hi = k_bytes[..., 1]             # [B, Hkv, T, Dk] uint8
    k_hi_triton = k_hi[0].contiguous() # [HKV, T, Dk]

    v_bytes = fp16_bytes_view(v)       # [B, Hkv, T, Dv, 2]
    v_hi = v_bytes[..., 1]             # [B, Hkv, T, Dv]
    v_hi_triton = v_hi[0].permute(1, 0, 2).contiguous()  # [T, HKV, Dv]

    return k_hi_triton, v_hi_triton


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
    qf = q_triton.float()
    kf = k_triton.float()
    HQ, K = qf.shape
    HKV, T, Kk = kf.shape
    assert Kk == K
    assert HQ % HKV == 0
    G = HQ // HKV
    if G != 1:
        kf_rep = kf.repeat_interleave(G, dim=0)  # [HQ, T, K]
    else:
        kf_rep = kf
    scores = torch.einsum('hk, htk -> ht', qf, kf_rep) * scale
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fp16_u8hi_matmul_kernel(
    A_ptr, B8_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptrs = B8_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    tl.multiple_of(rk, 16)
    tl.multiple_of(rm, 16)
    tl.multiple_of(rn, 16)

    for k0 in range(0, K, BLOCK_K):
        k_mask_row = (k0 + rk) < K
        a = tl.load(
            A_ptrs,
            mask=(rm[:, None] < M) & k_mask_row[None, :],
            other=0.0,
            eviction_policy='evict_last'
        )

        b8 = tl.load(
            B_ptrs,
            mask=k_mask_row[:, None] & (rn[None, :] < N),
            other=0,
            eviction_policy='evict_first'
        )
        b16 = b8.to(tl.uint16) << 8
        b = tl.cast(b16, tl.float16)

        acc += tl.dot(a, b, out_dtype=tl.float32)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fp16_u8hi_matmul(A_fp16: torch.Tensor, B_hi_u8: torch.Tensor) -> torch.Tensor:
    # A: [M, K] fp16, B_hi_u8: [K, N] uint8, return C: [M, N] fp32
    assert A_fp16.dtype == torch.float16 and B_hi_u8.dtype == torch.uint8
    assert A_fp16.is_cuda and B_hi_u8.is_cuda and A_fp16.device == B_hi_u8.device
    assert A_fp16.shape[1] == B_hi_u8.shape[0]
    M, K = A_fp16.shape
    K2, N = B_hi_u8.shape
    assert K2 == K
    C = torch.empty((M, N), device=A_fp16.device, dtype=torch.float32)

    stride_am, stride_ak = A_fp16.stride()
    stride_bk, stride_bn = B_hi_u8.stride()
    stride_cm, stride_cn = C.stride()

    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)
    fp16_u8hi_matmul_kernel[grid](
        A_fp16, B_hi_u8, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C


if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS, BK, BV = 256, 64, 64

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
        q_triton = to_triton_layout_q(q_rope_1)               # [HQ, D], fp16
        k_hi_triton, v_hi_triton = to_triton_layout_u8hi(k_rope, v)  # k:[HKV,T,D] u8, v:[T,HKV,Dv] u8

        # 为对比/参考准备全精度 triton 布局（仅用于验证，不参与 u8hi 计算）
        k_triton_full = k_rope[0].contiguous()                # [HKV, T, D], fp16
        v_triton_full = v[0].permute(1, 0, 2).contiguous()    # [T, HKV, Dv], fp16

        scale = 1.0 / math.sqrt(D)

        # 关键：使用 k 的高 8bit 计算阈值（不计入测时）
        qk_thresholds = calc_qk_threshold_u8hi(q_triton, k_hi_triton, scale)

        # 运行 Triton u8hi 实现（仅 K/V 高 8bit）
        o_triton, lse_triton = attn_fwd_q1_b1_splitT_u8hi(
            q_triton, k_hi_triton, v_hi_triton,
            scale=scale, BS=BS, BK=BK, BV=BV,
            qk_thresholds=qk_thresholds,
        )

        # 与 Flash 全精度对比（参考）
        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, Dv]

        # 数值对比
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        # LSE 参考（高精度，以 base-2）
        lse_ref2 = lse_reference_base2_gqa(q_triton, k_triton_full, scale)
        lse_max_abs = (lse_triton.float() - lse_ref2).abs().max().item()
        lse_rel = (lse_triton.float() - lse_ref2).abs().max() / (lse_ref2.abs().max().clamp_min(1e-6))
        lse_rel = lse_rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")
        print(f"LSE (base-2) diff vs FP32 ref: max_abs={lse_max_abs:.3e}, rel={lse_rel:.3e}")

        # 性能计时（不包含阈值计算）
        def run_triton_u8hi():
            o, _ = attn_fwd_q1_b1_splitT_u8hi(
                q_triton, k_hi_triton, v_hi_triton,
                scale=scale, BS=BS, BK=BK, BV=BV,
                qk_thresholds=qk_thresholds,
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton_u8hi, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton(u8hi)={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")