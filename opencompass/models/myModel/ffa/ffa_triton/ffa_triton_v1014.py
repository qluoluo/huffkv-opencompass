import os
# os.environ["CUDA_VISIABLE_DEVICES"] = "4"
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
    # 形状常量
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr,
    # 新增：stride
    stride_q_hq: tl.constexpr, stride_q_k: tl.constexpr,
    stride_k_hkv: tl.constexpr, stride_k_t: tl.constexpr, stride_k_k: tl.constexpr,
    stride_th_hq: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)

    # q: [HQ, K]
    q_ptrs = q + (base_hq + rows)[:, None] * stride_q_hq + offs_k[None, :] * stride_q_k
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # tb = 0
    tb0     = 0
    offs_t0 = tb0 * BS + tl.arange(0, BS)
    t_mask0 = offs_t0 < T
    # k: [HKV, T, K]
    k_ptrs0 = k + pid_hkv * stride_k_hkv + offs_k[:, None] * stride_k_k + offs_t0[None, :] * stride_k_t
    k_tile0 = tl.load(k_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0    = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0    = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0      = tl.max(b_s0, axis=1)

    # tb = NTB-1（NTB==1 时等于 0）
    tb1     = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    k_ptrs1 = k + pid_hkv * stride_k_hkv + offs_k[:, None] * stride_k_k + offs_t1[None, :] * stride_k_t
    k_tile1 = tl.load(k_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1    = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1    = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1      = tl.max(b_s1, axis=1)

    th = tl.maximum(m0, m1) - delta
    tl.store(thres_buf + (base_hq + rows) * stride_th_hq, th, mask=row_mask)


def precompute_attn_thresholds(
    q_rope_1: torch.Tensor,    # [B, Hq, 1, D]
    k_rope: torch.Tensor,      # [B, Hkv, T, D]
    scale: float | None,
    BS: int,
    delta: float = 1000.0,
):
    # 直接用原始形状，去除 to_triton_layout
    assert q_rope_1.is_cuda and k_rope.is_cuda
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    assert B == Bk == 1, "该 kernel 仅支持 batch=1"
    assert qlen == 1, "该 kernel 仅支持 qlen=1"
    assert Dq == Dk, "q/k head_dim 不一致"

    HQ  = Hq
    HKV = Hkv
    K   = Dq
    G   = HQ // HKV
    assert HQ % HKV == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

    NTB = triton.cdiv(T, BS)
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    thres_buf = torch.empty((HQ,), device=q_rope_1.device, dtype=torch.float32)

    # 从原始张量收集 stride（不做 permute/contiguous）
    # q: [B, Hq, 1, D] 只用 Hq 和 D 两个维度的 stride
    stride_q_hq = q_rope_1.stride(1)
    stride_q_k  = q_rope_1.stride(3)
    # k: [B, Hkv, T, D] 用 Hkv/T/D 三个维度的 stride
    stride_k_hkv = k_rope.stride(1)
    stride_k_t   = k_rope.stride(2)
    stride_k_k   = k_rope.stride(3)
    # thres_buf: [HQ]
    stride_th_hq = thres_buf.stride(0)

    grid_th = (HKV, 1)
    attn_find_threshold_two_blocks[grid_th](
        # 直接把原始指针传进去（B=1, qlen=1 的偏移由切片/stride 处理）
        q_rope_1, k_rope, thres_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS, BM_DOT=16,
        stride_q_hq=stride_q_hq, stride_q_k=stride_q_k,
        stride_k_hkv=stride_k_hkv, stride_k_t=stride_k_t, stride_k_k=stride_k_k,
        stride_th_hq=stride_th_hq,
    )
    return thres_buf


@triton.jit
def attn_fwd_stage1_pruned(
    q, k, k_bytes, v,
    m_buf, l_buf, o_buf,
    thres_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr,
    USE_FP8_K: tl.constexpr,
    # 新增：所有输入/输出的 stride
    stride_q_hq: tl.constexpr, stride_q_k: tl.constexpr,
    stride_k_hkv: tl.constexpr, stride_k_t: tl.constexpr, stride_k_k: tl.constexpr,
    stride_kb_hkv: tl.constexpr, stride_kb_t: tl.constexpr, stride_kb_k: tl.constexpr,
    stride_v_t: tl.constexpr, stride_v_hkv: tl.constexpr, stride_v_v: tl.constexpr,
    stride_m_hq: tl.constexpr, stride_m_ntbs: tl.constexpr,
    stride_l_hq: tl.constexpr, stride_l_ntbs: tl.constexpr,
    stride_ob_hq: tl.constexpr, stride_ob_ntbs: tl.constexpr, stride_ob_v: tl.constexpr,
    stride_th_hq: tl.constexpr,
    stride_mask_hkv: tl.constexpr, stride_mask_ntbs: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    s0       = pid_tb * BS
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)

    # q: [HQ, K]
    q_ptrs = q + (base_hq + rows)[:, None] * stride_q_hq + offs_k[None, :] * stride_q_k
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # k/k_bytes: [HKV, T, K]
        if USE_FP8_K:
            kb_ptrs = k_bytes + pid_hkv * stride_kb_hkv + offs_k[:, None] * stride_kb_k + offs_t_sb[None, :] * stride_kb_t
            k_tile  = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
        else:
            k_ptrs = k + pid_hkv * stride_k_hkv + offs_k[:, None] * stride_k_k + offs_t_sb[None, :] * stride_k_t
            k_tile = tl.load(k_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
        m_rows_blk = tl.max(b_s_act, axis=1)

        th_rows = tl.load(thres_buf + (base_hq + rows) * stride_th_hq, mask=row_mask, other=NEG_INF)
        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb  = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                # v: [T, HKV, V]
                v_ptrs = v + offs_t_sb[:, None] * stride_v_t + pid_hkv * stride_v_hkv + v_offs[None, :] * stride_v_v
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            # 写回：m/l/o_buf
            m_ptrs = m_buf + (base_hq + rows) * stride_m_hq + tb_sb * stride_m_ntbs
            l_ptrs = l_buf + (base_hq + rows) * stride_l_hq + tb_sb * stride_l_ntbs
            o_ptrs = o_buf + (base_hq + rows)[:, None] * stride_ob_hq + tb_sb * stride_ob_ntbs + v_offs[None, :] * stride_ob_v

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

            # 标记该 (tb, sb) 为有效
            tl.store(mask_buf + pid_hkv * stride_mask_hkv + tb_sb * stride_mask_ntbs, tl.full((), 1, tl.int8))
        # else: 跳过


@triton.jit
def attn_fwd_stage2_masked(
    m_buf, l_buf, o_buf,
    mask_buf,
    o,
    NTBS,
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
    # 新增 stride
    stride_m_hq: tl.constexpr, stride_m_ntbs: tl.constexpr,
    stride_l_hq: tl.constexpr, stride_l_ntbs: tl.constexpr,
    stride_ob_hq: tl.constexpr, stride_ob_ntbs: tl.constexpr, stride_ob_v: tl.constexpr,
    stride_mask_hkv: tl.constexpr, stride_mask_ntbs: tl.constexpr,
    stride_o_hq: tl.constexpr, stride_o_v: tl.constexpr,
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
        keep = tl.load(mask_buf + pid_hkv * stride_mask_hkv + tb * stride_mask_ntbs).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * stride_m_hq + tb * stride_m_ntbs)
            l_b = tl.load(l_buf + pid_hq * stride_l_hq + tb * stride_l_ntbs)
            o_b = tl.load(o_buf + pid_hq * stride_ob_hq + tb * stride_ob_ntbs + v_offs * stride_ob_v)

            new_m  = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk  = tl.exp2(m_b - new_m)

            b_acc = b_acc * r_prev + l_b * r_blk
            b_o   = b_o   * r_prev + o_b * r_blk
            b_m   = new_m

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_hq * stride_o_hq + v_offs * stride_o_v
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    # 现在的 mask_buf: [HKV, NTBS], int8, 1=keep, 0=skip
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()     # device 上做 sum，避免 int8 溢出
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())           # 注意：.item() 会触发同步


def attn_fwd_q1_b1_splitT(
    q_rope_1: torch.Tensor,    # [B, Hq, 1, D], float16
    k_rope: torch.Tensor,      # [B, Hkv, T, D], float16
    v: torch.Tensor,           # [B, Hkv, T, Dv], float16
    scale: float | None = None,
    BS: int = 128,
    SBS: int | None = None,
    BM_DOT: int = 16,
    delta: float = 5.0,
    thres_buf: torch.Tensor | None = None,
    return_skip_ratio: bool = False,
    use_fp8_k_high_byte: bool = True,
    k_bytes: torch.Tensor | None = None,
):
    # 直接用原始形状，去除 to_triton_layout
    assert q_rope_1.is_cuda and k_rope.is_cuda and v.is_cuda
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk  = k_rope.shape
    Bv, Hv, Tv, Dv  = v.shape
    assert B == Bk == Bv == 1, "该 kernel 仅支持 batch=1"
    assert qlen == 1, "该 kernel 仅支持 qlen=1"
    assert Hkv == Hv, "k/v heads 必须一致"
    assert Dq == Dk, "q/k head_dim 不一致"
    assert T == Tv
    assert Hq % Hkv == 0, "GQA 要求 Hq 是 Hkv 的整数倍（或 MQA Hkv=1）"

    HQ  = Hq
    HKV = Hkv
    K   = Dq
    V   = Dv
    G   = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    NTB  = triton.cdiv(T, BS)
    NSB  = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o     = torch.empty((HQ, V), device=q_rope_1.device, dtype=q_rope_1.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q_rope_1.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q_rope_1.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q_rope_1.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q_rope_1.device, dtype=torch.int8)

    # 阈值
    if thres_buf is None:
        thres_buf = precompute_attn_thresholds(q_rope_1, k_rope, scale=scale, BS=SBS, delta=delta)
    else:
        assert thres_buf.shape == (HQ,) and thres_buf.dtype == torch.float32 and thres_buf.device == q_rope_1.device

    # 准备 k 的高 8 位视图：无需 contiguous，直接用 stride（基于原始形状）
    if k_bytes is None and use_fp8_k_high_byte:
        # 注意：view 到 float8 后，最后一维等价于 2x 的字节数；[..., 1::2] 取每个 fp16 的高字节
        k_bytes = k_rope.view(torch.float8_e5m2)[..., 1::2]

    # 收集 stride（全部基于原始形状）
    stride_q_hq = q_rope_1.stride(1)
    stride_q_k  = q_rope_1.stride(3)

    stride_k_hkv = k_rope.stride(1)
    stride_k_t   = k_rope.stride(2)
    stride_k_k   = k_rope.stride(3)

    stride_v_t   = v.stride(2)   # 注意：v 原始形状是 [B, HKV, T, V]，这里传 T 轴的 stride
    stride_v_hkv = v.stride(1)   # HKV 轴的 stride
    stride_v_v   = v.stride(3)   # V 轴的 stride

    stride_m_hq, stride_m_ntbs           = m_buf.stride()
    stride_l_hq, stride_l_ntbs           = l_buf.stride()
    stride_ob_hq, stride_ob_ntbs, stride_ob_v = o_buf.stride()
    stride_th_hq                         = thres_buf.stride(0)
    stride_mask_hkv, stride_mask_ntbs    = mask_buf.stride()

    if use_fp8_k_high_byte:
        assert k_bytes is not None, "use_fp8_k_high_byte=True 需要传入或构造 k_bytes"
        # k_bytes 的形状是 [B, HKV, T, K]，取 B=1 的视图；stride 按 HKV/T/K 三轴传入
        stride_kb_hkv = k_bytes.stride(1)
        stride_kb_t   = k_bytes.stride(2)
        stride_kb_k   = k_bytes.stride(3)
    else:
        stride_kb_hkv = stride_kb_t = stride_kb_k = 0

    # Stage 1
    attn_fwd_stage1_pruned[(HKV, NTB)](
        # 直接传原始张量指针（B=1 的偏移由 stride 处理；qlen=1 不参与索引）
        q_rope_1, k_rope, k_bytes, v,
        m_buf, l_buf, o_buf,
        thres_buf, mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS, BM_DOT=BM_DOT,
        USE_FP8_K=use_fp8_k_high_byte,
        stride_q_hq=stride_q_hq, stride_q_k=stride_q_k,
        stride_k_hkv=stride_k_hkv, stride_k_t=stride_k_t, stride_k_k=stride_k_k,
        stride_kb_hkv=stride_kb_hkv, stride_kb_t=stride_kb_t, stride_kb_k=stride_kb_k,
        stride_v_t=stride_v_t, stride_v_hkv=stride_v_hkv, stride_v_v=stride_v_v,
        stride_m_hq=stride_m_hq, stride_m_ntbs=stride_m_ntbs,
        stride_l_hq=stride_l_hq, stride_l_ntbs=stride_l_ntbs,
        stride_ob_hq=stride_ob_hq, stride_ob_ntbs=stride_ob_ntbs, stride_ob_v=stride_ob_v,
        stride_th_hq=stride_th_hq,
        stride_mask_hkv=stride_mask_hkv, stride_mask_ntbs=stride_mask_ntbs,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept  = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2
    attn_fwd_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V,
        stride_m_hq=stride_m_hq, stride_m_ntbs=stride_m_ntbs,
        stride_l_hq=stride_l_hq, stride_l_ntbs=stride_l_ntbs,
        stride_ob_hq=stride_ob_hq, stride_ob_ntbs=stride_ob_ntbs, stride_ob_v=stride_ob_v,
        stride_mask_hkv=stride_mask_hkv, stride_mask_ntbs=stride_mask_ntbs,
        stride_o_hq=o.stride(0), stride_o_v=o.stride(1),
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o


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
    from load_utils import load_qkvh

    # torch.set_float32_matmul_precision("high")

    # exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuxiaoran-240108120089/projects_zgliu/projects/huffKV/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'
    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'

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
    use_fp8_k_high_byte = False
    # use_fp8_k_high_byte = True

    print(f"{BS=}")

    # 计时参数
    iters = 100
    warmup = 100

    # iters = 1
    # warmup = 0

    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        if layer_idx == 0:
            continue
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype)       # [B, Hkv, T, Dv]
        
        # import ipdb; ipdb.set_trace()

        # 只取最后一个查询位置 -> qlen=1
        q_rope_1 = q_rope[:, :, -1:, :]  # [B, Hq, 1, D]

        # 直接基于原始形状计算 k 的高位字节视图（可选）
        k_bytes = k_rope.view(torch.float8_e5m2)[..., 1::2]
        # k_bytes = k_rope.view(torch.float8_e5m2)[..., 1::2].contiguous()
        # k_bytes = None
        
        q_rope_1 = q_rope_1.contiguous()
        k_rope = k_rope.contiguous()
        v = v.transpose(1,2).contiguous().transpose(1,2)

        # 阈值（原始形状）
        thres_buf = precompute_attn_thresholds(
            q_rope_1, k_rope,
            scale=None, BS=SBS, delta=delta,
        )
        # thres_buf = None
        torch.cuda.synchronize()

        # 前向（原始形状）
        o_triton, skip_ratio = attn_fwd_q1_b1_splitT(
            q_rope_1, k_rope, v,
            scale=None, BS=BS, SBS=SBS,
            delta=delta,
            thres_buf=thres_buf,
            return_skip_ratio=True,
            use_fp8_k_high_byte=use_fp8_k_high_byte,
            k_bytes=k_bytes,
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")

        o_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, Dv]

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton():
            o = attn_fwd_q1_b1_splitT(
                q_rope_1, k_rope, v,
                scale=None, BS=BS, SBS=SBS,
                thres_buf=thres_buf,
                return_skip_ratio=False,
                use_fp8_k_high_byte=use_fp8_k_high_byte,
                k_bytes=k_bytes,
            )
            return o

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        break