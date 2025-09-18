import math
import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ===================== 4-bit 预压缩（两两打包到 1 byte） =====================

def compress_k_hi4_fp16(k_triton: torch.Tensor) -> torch.Tensor:
    """
    将 float16 的 k 提取每元素的高 4 位 (bit[15:12])，两两打包成 1 个字节。
    输入: k_triton [HKV, T, K], dtype=float16（必须）
    输出: k_hi4    [HKV, T, ceil(K/2)], dtype=uint8
    说明：CUDA 上对 uint16 位移不友好，这里转 int32 再按位处理。
    """
    assert k_triton.dtype == torch.float16
    assert k_triton.is_cuda
    k_u16 = k_triton.view(torch.uint16)  # 仅重解释
    # 取高 4 位
    nibbles = ((k_u16.to(torch.int32) >> 12) & 0xF).to(torch.uint8)  # [HKV, T, K]

    HKV, T, K = nibbles.shape
    K4 = (K + 1) // 2

    lo = nibbles[..., 0::2]  # 偶数 index -> 放到低半字节
    hi = nibbles[..., 1::2]  # 奇数 index -> 放到高半字节
    if hi.shape[-1] != lo.shape[-1]:
        # K 为奇数，最后一对缺高半字节，补 0
        hi = torch.cat([hi, torch.zeros_like(lo[..., :1])], dim=-1)

    packed = (lo | (hi << 4)).contiguous()  # [HKV, T, K4]
    return packed


def build_lut16_hi4_fp16(device, dtype=torch.float16):
    """
    构造 16 长度的 LUT：索引为高 4 位 c (bit[15:12])，输出为一个代表值（fp16/32）。
    近似策略（桶中心）：
      sign = c>>3
      exp5 = ((c & 7) << 2) | 2  （指数低 2 位取中间值 2）
      mant = 1 + 0.5             （尾数取中值 0.5）
      val  = sgn * 2^(exp5-15) * mant
    对于 exp5==0 当作 0；exp5==31 给最大有限值（避免 Inf）。
    """
    lut = torch.empty(16, dtype=dtype, device=device)
    max_f16 = float(torch.finfo(torch.float16).max)
    for c in range(16):
        sgn = -1.0 if (c & 0x8) else 1.0
        e_hi3 = (c & 0x7)
        exp5 = (e_hi3 << 2) | 0x2  # 指数低 2 位取 2
        if exp5 == 0:
            val = 0.0
        elif exp5 == 31:
            val = sgn * max_f16
        else:
            mant = 1.0 + 0.5
            val = sgn * (2.0 ** (exp5 - 15)) * mant
        lut[c] = torch.tensor(val, dtype=dtype, device=device)
    return lut


# ===================== Stage 1：4-bit + LUT 的 Triton 内核 =====================

@triton.jit
def attn_fwd_q1_b1_stage1_hi4_lut(
    q,            # [HQ, K]
    k4,           # [HKV, T, K4], uint8（两半字节打包在 1 字节）
    v,            # [T, HKV, V]
    lut16,        # [16], fp16/fp32
    m_buf,        # [HQ, NTB]
    l_buf,        # [HQ, NTB]
    o_buf,        # [HQ, NTB, V], fp32
    scale,        # float
    T,            # int
    NTB,          # int = ceil(T / BS)
    HKV: tl.constexpr,
    HQ: tl.constexpr,
    K: tl.constexpr,
    K4: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,     # GQA group size
    BS: tl.constexpr,    # time tile
    BK: tl.constexpr,    # K dim tile
    BV: tl.constexpr,    # V dim tile
):
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

    RCP_LN2 = 1.4426950408889634

    b_s = tl.zeros([BS], tl.float32)
    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        k_mask = offs_k < K

        # q tile
        q_ptrs = q + i_hq * K + offs_k
        q_chunk = tl.load(q_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BK]

        # 计算字节地址（每个字节存两个 4-bit），以及奇偶位选择
        byte_offs = (offs_k // 2)  # [BK]
        k4_ptrs = k4 + i_h * (T * K4) + (offs_t[None, :] * K4) + byte_offs[:, None]  # [BK, BS]
        bytes_u8 = tl.load(k4_ptrs, mask=(k_mask[:, None] & t_mask[None, :]), other=0).to(tl.int32)

        # 半字节选择：奇数 K -> 取高半字节，偶数 K -> 取低半字节
        odd = (offs_k % 2).to(tl.int32)[:, None]  # [BK,1]，值为 0 或 1
        code = (bytes_u8 >> (odd * 4)) & 0xF      # [BK,BS]，取出 4-bit 码

        # LUT 直接映射为值（fp16/fp32），再转 fp32 做 FMA
        k_vals = tl.load(lut16 + code)            # [BK,BS]
        k_chunk = k_vals.to(tl.float32)

        b_s += tl.sum(q_chunk[:, None] * k_chunk, axis=0)

    # softmax（以 2 为底）
    b_s = b_s * scale * RCP_LN2
    b_s_masked = tl.where(t_mask, b_s, float('-inf'))
    m_b = tl.max(b_s_masked, axis=0)

    b_p = tl.exp2(b_s - m_b)
    b_p = tl.where(t_mask, b_p, 0.0)
    l_b = tl.sum(b_p, axis=0)

    # 聚合 V
    v_ptrs = v + (offs_t[:, None] * (HKV * V)) + (i_h * V) + v_offs[None, :]
    b_v = tl.load(v_ptrs, mask=(t_mask[:, None] & v_mask[None, :]), other=0.0).to(tl.float32)
    o_b = tl.sum(b_p[:, None] * b_v, axis=0)

    # 写回
    o_ptrs = o_buf + i_hq * (NTB * V) + pid_tb * V + v_offs
    tl.store(o_ptrs, o_b, mask=v_mask)

    if pid_v == 0:
        tl.store(m_buf + i_hq * NTB + pid_tb, m_b)
        tl.store(l_buf + i_hq * NTB + pid_tb, l_b)


# ===================== Stage 2（与你原实现相同） =====================

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

    # 在线合并 across tb（base-2 域）
    b_m = tl.full((), float('-inf'), tl.float32)  # 当前全局最大值
    b_acc = tl.zeros((), tl.float32)              # 当前全局分母
    b_o = tl.zeros([BV], tl.float32)              # 当前全局分子向量

    # 沿 tb 做稳定合并
    for tb in range(0, NTB):
        m_b = tl.load(m_buf + pid_hq * NTB + tb)
        l_b = tl.load(l_buf + pid_hq * NTB + tb)
        o_b = tl.load(o_buf + pid_hq * (NTB * V) + tb * V + v_offs, mask=v_mask, other=0.0)

        new_m = tl.maximum(b_m, m_b)
        r_prev = tl.exp2(b_m - new_m)
        r_blk = tl.exp2(m_b - new_m)

        b_acc = b_acc * r_prev + l_b * r_blk
        b_o = b_o * r_prev + o_b * r_blk
        b_m = new_m

    # 归一化与 lse
    out_tile = b_o / b_acc
    if pid_v == 0:
        lse_val = b_m + tl.log2(b_acc)
        tl.store(lse + pid_hq, lse_val)

    # 写回输出
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=v_mask)


# ===================== 封装：splitT（使用 4-bit LUT） =====================

def attn_fwd_q1_b1_splitT_hi4_lut(
    q: torch.Tensor,         # [HQ, K], fp16/bf16/fp32
    k_hi4: torch.Tensor,     # [HKV, T, ceil(K/2)], uint8（压缩后）
    v: torch.Tensor,         # [T, HKV, V], same dtype as q
    lut16: torch.Tensor,     # [16], fp16/fp32, on device
    scale: float = None,
    BS: int = 128, BK: int = 64, BV: int = 64,
):
    assert q.is_cuda and k_hi4.is_cuda and v.is_cuda and lut16.is_cuda
    assert q.ndim == 2 and k_hi4.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, K4 = k_hi4.shape
    Tv, HKV2, V = v.shape
    assert HKV == HKV2 and Tv == T
    assert HQ % HKV == 0, "GQA 需要 HQ 是 HKV 的整数倍"
    G = HQ // HKV
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    NTB = triton.cdiv(T, BS)
    assert BK % 2 == 0, "BK 建议为偶数，避免半字节选择开销"

    # 输出与中间缓冲
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    lse = torch.empty((HQ,), device=q.device, dtype=torch.float32)
    m_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTB), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTB, V), device=q.device, dtype=torch.float32)

    grid1 = (triton.cdiv(V, BV), HQ, NTB)
    attn_fwd_q1_b1_stage1_hi4_lut[grid1](
        q, k_hi4, v, lut16,
        m_buf, l_buf, o_buf,
        scale, T, NTB,
        HKV=HKV, HQ=HQ, K=K, K4=K4, V=V, G=G,
        BS=BS, BK=BK, BV=BV,
    )

    grid2 = (triton.cdiv(V, BV), HQ)
    attn_fwd_q1_b1_stage2[grid2](
        m_buf, l_buf, o_buf, o, lse, NTB,
        HQ=HQ, V=V, BV=BV,
    )
    return o, lse


# ===================== 其余辅助（与原先一致） =====================

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
    out = flash_attn_func(
        q_rope_1.transpose(1, 2),
        k_rope.transpose(1, 2),
        v.transpose(1, 2),
        causal=False,
    )
    out = out.squeeze(0).squeeze(0)
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
    lse_e = torch.logsumexp(scores, dim=-1)
    lse_base2 = lse_e * RCP_LN2
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


# ===================== Demo / 主流程（把 HI8 改为 HI4+LUT） =====================

if __name__ == "__main__":
    from utils import load_qkvh

    torch.set_float32_matmul_precision("high")

    # exp_root = '.../longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16  # 该近似要求 k 为 fp16（为了从 bit 中提取）
    BS, BK, BV = 256, 64, 64  # Triton tile

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

        # 布局变换（GQA 支持）
        q_triton, k_triton, v_triton = to_triton_layout(q_rope_1, k_rope, v)
        assert k_triton.dtype == torch.float16, "k 必须是 float16 以便提取高 4 位"

        # 离线预压缩：k -> k_hi4（打包两半字节到 1 byte）
        k_hi4 = compress_k_hi4_fp16(k_triton)  # uint8, [HKV, T, ceil(K/2)]

        # 构造 16 长度 LUT（放 device 上）
        lut16 = build_lut16_hi4_fp16(device=q_triton.device, dtype=torch.float16)

        # 缩放
        scale = 1.0 / math.sqrt(D)

        # 计算（4-bit + LUT）
        o_hi4, lse_hi4 = attn_fwd_q1_b1_splitT_hi4_lut(
            q_triton, k_hi4, v_triton, lut16,
            scale=scale, BS=BS, BK=BK, BV=BV,
        )

        # 用 Flash 作为参考
        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # 数值对比（与 Flash 输出）
        max_abs = (o_hi4.float() - o_flash.float()).abs().max().item()
        rel = (o_hi4.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        # LSE 参考（高精度，用于 sanity check）
        lse_ref2 = lse_reference_base2_gqa(q_triton, k_triton, scale)  # [HQ], base-2
        lse_max_abs = (lse_hi4.float() - lse_ref2).abs().max().item()
        lse_rel = (lse_hi4.float() - lse_ref2).abs().max() / (lse_ref2.abs().max().clamp_min(1e-6))
        lse_rel = lse_rel.item()

        print(f"[HI4-LUT] Value diff vs Flash(GQA): max_abs={max_abs:.3e}, rel={rel:.3e}")
        print(f"[HI4-LUT] LSE (base-2) diff vs FP32 ref: max_abs={lse_max_abs:.3e}, rel={lse_rel:.3e}")

        # 性能对比（可选：对比 hi4 与 Flash）
        def run_hi4():
            o, _ = attn_fwd_q1_b1_splitT_hi4_lut(q_triton, k_hi4, v_triton, lut16, scale=scale, BS=BS, BK=BK, BV=BV)
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_hi4 = bench_op(run_hi4, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: HI4-LUT-Triton={ms_hi4:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_hi4/ms_flash:.2f}x")