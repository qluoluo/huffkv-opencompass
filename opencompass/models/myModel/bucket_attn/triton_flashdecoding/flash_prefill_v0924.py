import os
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache")
# os.environ['TRITON_DUMP_ASSEMBLY'] = "1"

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl



import math
import torch
import triton
import triton.language as tl


@triton.jit
def attn_prefill_stage1(
    q, k, k_bytes, v,                 # q: [HQ, TQ, K]; k/k_bytes: [HKV, T, K]; v: [T, HKV, V]
    m_buf, l_buf, o_buf,              # m/l: [TQ, HQ, NTBS]; o_buf: [TQ, HQ, NTBS, V]
    mask_buf,                         # [TQ, HKV, NTBS], int8
    scale, T, TQ, NTB, NTBS,          # ints
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr, SQT: tl.constexpr,
    USE_FP8_K: tl.constexpr = True,   # 是否仅加载 k 的高 8 位（fp8_e5m2）
    CAUSAL: tl.constexpr = True,      # 是否启用因果掩码 (t 与 tq 的关系)
):
    pid_hkv = tl.program_id(0)   # [0, HKV)
    pid_tb  = tl.program_id(1)   # [0, NTB)
    pid_ntq = tl.program_id(2)   # [0, NTQ)
    base_hq = pid_hkv * G

    # 本 TB 的起始全局 T 偏移 & 本 NTQ 的起始 Q 偏移
    s0    = pid_tb * BS
    q_s0  = pid_ntq * SQT

    # 维度常量
    g_rows = tl.arange(0, G)     # [G]
    offs_k = tl.arange(0, K)     # [K]
    v_offs = tl.arange(0, V)     # [V]

    # 常量
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 循环每个子块（沿 T）
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)  # [SBS]
        t_mask_sb = offs_t_sb < T                      # [SBS] 边界屏蔽

        # K 路：一次加载、被本 NTQ 内所有 tq 复用
        if USE_FP8_K:
            kb_ptrs = k_bytes + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
            k_tile  = tl.load(
                kb_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                other=0.0
            ).to(tl.float16)  # [K, SBS]
        else:
            k_ptrs = k + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
            k_tile = tl.load(
                k_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                other=0.0
            ).to(tl.float16)  # [K, SBS]

        # V 路：一次加载、复用
        v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
        b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)  # [SBS, V]

        # 本 NTQ 覆盖的 Q 子段
        offs_tq_blk = q_s0 + tl.arange(0, SQT)        # [SQT]
        q_mask_blk  = offs_tq_blk < TQ                # [SQT]

        # 遍历子段内的每个 tq（静态循环）：每次处理全部 G 行
        for sq in tl.static_range(SQT):
            tq        = q_s0 + sq
            sq_valid  = tq < TQ

            # Q 路：q_tile: [G, K]
            q_ptrs = q + (base_hq + g_rows)[:, None] * (TQ * K) + tq * K + offs_k[None, :]
            q_tile = tl.load(q_ptrs, other=0.0).to(tl.float16)

            # 因果掩码：对齐最后 token
            # t_max = tq + (T - TQ)，由 t_mask_sb(offs_t_sb < T) 保证不会越过 T-1
            if CAUSAL:
                t_max = tq + (T - TQ)
                t_mask_sq = t_mask_sb & (offs_t_sb <= t_max)
            else:
                t_mask_sq = t_mask_sb

            # 若当前 tq 无效，则整块无效
            t_mask_sq = t_mask_sq & tl.full([SBS], sq_valid, tl.int1)

            # 计算注意力分数（log2-softmax）
            b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2   # [G, SBS]
            b_s_act = tl.where(t_mask_sq[None, :], b_s, NEG_INF)                       # [G, SBS]

            # 子块内行最大值与 softmax 分母
            m_rows = tl.max(b_s_act, axis=1)                                           # [G]
            b_p    = tl.where(t_mask_sq[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0) # [G, SBS]
            l_rows = tl.sum(b_p, axis=1)                                               # [G]

            # V 路输出（子块内）
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)             # [G, V]

            # 是否保留该小块（如果该 tq 下子块有任何有效位置）
            valid_cnt = tl.sum(t_mask_sq.to(tl.int32), axis=0)  # 标量
            keep_blk  = valid_cnt > 0

            # 写回（带 tq）；仅在 keep 时写入并标记
            tb_sb  = pid_tb * NSB + sb
            g_mask = tl.full([G], keep_blk & sq_valid, tl.int1)

            m_ptrs = m_buf + tq * (HQ * NTBS) + (base_hq + g_rows) * NTBS + tb_sb
            l_ptrs = l_buf + tq * (HQ * NTBS) + (base_hq + g_rows) * NTBS + tb_sb
            o_ptrs = o_buf + tq * (HQ * NTBS * V) + (base_hq + g_rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=g_mask)
            tl.store(l_ptrs, l_rows, mask=g_mask)
            tl.store(o_ptrs, o_tile, mask=g_mask[:, None])

            # 标记该 (tq, tb, sb) 为有效
            keep_ptr = mask_buf + tq * (HKV * NTBS) + pid_hkv * NTBS + tb_sb
            tl.store(keep_ptr, tl.full((), 1, tl.int8), mask=(keep_blk & sq_valid))


@triton.jit
def attn_prefill_stage2(
    m_buf, l_buf, o_buf,        # [TQ, HQ, NTBS], [TQ, HQ, NTBS], [TQ, HQ, NTBS, V]
    mask_buf,                   # [TQ, HKV, NTBS], int8
    o,                          # [TQ, HQ, V], out dtype = q.dtype
    NTBS,                       # int
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr, TQ: tl.constexpr,
):
    pid_hkv = tl.program_id(0)   # [0, HKV)
    g       = tl.program_id(1)   # [0, G)
    pid_tq  = tl.program_id(2)   # [0, TQ)
    pid_hq  = pid_hkv * G + g

    v_offs  = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)

    b_m   = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o   = tl.zeros([V], tl.float32)

    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_tq * (HKV * NTBS) + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_tq * (HQ * NTBS) + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_tq * (HQ * NTBS) + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_tq * (HQ * NTBS * V) + pid_hq * (NTBS * V) + tb * V + v_offs)

            new_m  = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk  = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o   = b_o   * r_prev + o_b * r_blk
            b_m   = new_m

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_tq * (HQ * V) + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def attn_prefill_qN_b1_splitT(
    q: torch.Tensor,      # [HQ, TQ, K]
    k: torch.Tensor,      # [HKV, T, K], float16
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    SQT: int = 4,                      # 网格内 Q 的序列长度（tile）
    return_skip_ratio: bool = False,
    use_fp8_k_high_byte: bool = True,
    k_bytes: torch.Tensor | None = None,
    causal: bool = True,
):
    """
    Prefill 前向（Q 多步，B=1），按 T 维分块，不使用阈值裁剪。
    - 因果掩码在 T != TQ 时对齐最后一个 token：t_max = tq + (T - TQ)
    - 输入:
      - q: [HQ, TQ, K]
      - k: [HKV, T, K]
      - v: [T, HKV, V]
    - 输出:
      - o: [TQ, HQ, V]
    """
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
    HQ, TQ, Kq = q.shape
    HKV, T, Kk = k.shape
    Tv, HKV2, V = v.shape
    assert Kq == Kk and Tv == T and HKV2 == HKV
    assert HQ % HKV == 0
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(Kq)

    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS
    assert SQT >= 1

    NTB  = triton.cdiv(T, BS)
    NSB  = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB
    NTQ  = triton.cdiv(TQ, SQT)

    # 输出与中间缓冲
    o     = torch.empty((TQ, HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((TQ, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((TQ, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((TQ, HQ, NTBS, V), device=q.device, dtype=torch.float32)

    mask_buf = torch.zeros((TQ, HKV, NTBS), device=q.device, dtype=torch.int8)

    # 准备 k 的字节视图（高 8 位）
    if k_bytes is None:
        # 取 fp16 的高字节视图作为 fp8_e5m2 高位
        k_bytes = k.contiguous().view(torch.float8_e5m2)[..., 1::2].contiguous()

    # Stage 1：按 (HKV, NTB, NTQ) 并行；每个块处理 G * SQT 个 Q
    attn_prefill_stage1[(HKV, NTB, NTQ)](
        q, k, k_bytes, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, TQ, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=Kq, V=V, G=G, BS=BS, SBS=SBS, SQT=SQT,
        USE_FP8_K=use_fp8_k_high_byte,
        CAUSAL=causal,
        # 可按需设置：num_stages=3, num_warps=8,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept  = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2：归并各小块得到最终输出
    attn_prefill_stage2[(HKV, G, TQ)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V, TQ=TQ,
    )

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

        k_bytes = k_triton.contiguous().view(torch.float8_e5m2)[..., 1::2].contiguous() 
        # k_bytes = k_triton.contiguous().view(torch.float8_e5m2)[..., 1::2]

        thres_buf = precompute_attn_thresholds(
            q_triton, k_triton,
            scale=scale, BS=SBS, delta=delta,
        )
        torch.cuda.synchronize()

        o_triton, skip_ratio = attn_fwd_q1_b1_splitT(
            q_triton, k_triton, v_triton,
            scale=scale, BS=BS, SBS=SBS,
            delta=delta,
            thres_buf=thres_buf,
            return_skip_ratio=True,   # 仅在这里拿统计；计时时不要开
            use_fp8_k_high_byte=use_fp8_k_high_byte,
            k_bytes=k_bytes,
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton():
            o = attn_fwd_q1_b1_splitT(
                q_triton, k_triton, v_triton,
                scale=scale, BS=BS, SBS=SBS,
                thres_buf=thres_buf,
                return_skip_ratio=False,   # 计时时不开
                use_fp8_k_high_byte=use_fp8_k_high_byte,
                k_bytes=k_bytes,
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