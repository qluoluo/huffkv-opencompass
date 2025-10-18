# 本文件将 k 的三种形式全部存储并在 Python 层传递：
# - 高 8 位：k_hi8（dtype=torch.float8_e5m2），可直接参与计算（kernel 内转换为 float16）
# - 低 8 位：k_lo8（dtype=torch.uint8），仅存储目前不参与计算
# - 完整 16 位：k_fp16（dtype=torch.float16），可直接用于计算
#
# 新增两个开关：
# - th_use_fp8: 控制阈值计算使用 8bit（k_hi8）还是 16bit（k_fp16）
# - ker_use_fp8: 控制主 kernel（Stage 1）使用 8bit（k_hi8）还是 16bit（k_fp16）
#
# 重要说明：
# - 假设小端存储（little-endian）：fp16 的低字节在偶数索引，高字节在奇数索引。
# - k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2]，k_lo8 = k_fp16.view(torch.uint8)[..., 0::2]。
# - Triton 内核对 k 的 dtype 不做假设，统一 tl.load 后 .to(tl.float16) 再参与计算（支持 fp8_e5m2 或 fp16）。

import os
# os.environ["CUDA_VISIABLE_DEVICES"] = "4"

# os.environ["TRITON_DUMP_ASSEMBLY"] = "1"
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache_fp8")

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


@triton.jit
def attn_compute_threshold_two_blocks(
    q, k_mem, threshold_buf, scale, T, NTB, delta,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, G: tl.constexpr, BS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    # k_mem: 可以是 fp8_e5m2 或 fp16，统一在 kernel 内转为 fp16 再计算
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
    kb_ptrs0 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(kb_ptrs0, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # tb = NTB-1（当 NTB==1 时等于 0，与 tb0 相同）
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_mem + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(kb_ptrs1, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    m2 = tl.maximum(m0, m1)
    th = m2 - delta
    tl.store(threshold_buf + (base_hq + rows), th, mask=row_mask)


def compute_attn_thresholds(
    q: torch.Tensor,          # [HQ, K]
    k_mem: torch.Tensor,      # [HKV, T, K], float8_e5m2 或 float16（内部会统一到 fp16 计算）
    scale: float,
    BS: int,
    delta: float = 1000.0,
):
    
    BS = 16
    
    assert q.is_cuda and k_mem.is_cuda
    HQ, K = q.shape
    HKV, T, Kk = k_mem.shape
    assert Kk == K and (HQ % HKV == 0)
    G = HQ // HKV
    NTB = triton.cdiv(T, BS)

    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)

    grid_th = (HKV, 1)
    attn_compute_threshold_two_blocks[grid_th](
        q, k_mem, threshold_buf,
        scale, T, NTB, delta,
        HKV=HKV, HQ=HQ, K=K, G=G, BS=BS,
        
        num_warps=2, num_stages=1
    )
    return threshold_buf


@triton.jit
def attn_forward_stage1_pruned(
    q, k_mem, v,            # k_mem 可为 fp8_e5m2 或 fp16，统一在 kernel 内转到 fp16 参与计算
    m_buf, l_buf, o_buf,
    threshold_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    # 本 TB 的起始全局 T 偏移
    s0 = pid_tb * BS

    # 行/列常量与 Q tile
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G

    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # 常量
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # NSB 为编译期常量：每个大块内的子块数（= ceil(BS / SBS)）
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 循环每个子块
    for sb in tl.static_range(NSB):
        # 子块内的 T 偏移与 mask
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # 仅载入 k 的对应 dtype（fp8 或 fp16），并转成 fp16 参与计算
        kb_ptrs = k_mem + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(
            kb_ptrs,
            mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
            other=0.0
        ).to(tl.float16)

        # 注意：对数底为 2 的 softmax 形式
        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2  # [BM_DOT, SBS]
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        # 子块内行最大值
        m_rows_blk = tl.max(b_s_act, axis=1)

        # 阈值与裁剪
        th_rows = tl.load(threshold_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)
        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        # 写入索引：将子块映射为全局块 idx
        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # softmax 规范化（子块内）
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            # V 路
            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            # 写回（注意 stride 里的 NTBS）
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

            # 标记该 (tb, sb) 为有效
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))
        # else: 跳过


@triton.jit
def attn_forward_stage2_masked(
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


def attn_forward_q1_b1_splitT(
    q: torch.Tensor,         # [HQ, K]
    k_hi8: torch.Tensor,     # [HKV, T, K], float8_e5m2（高 8 位）
    k_lo8: torch.Tensor,     # [HKV, T, K], uint8（低 8 位）
    k_fp16: torch.Tensor,    # [HKV, T, K], float16（完整 k）
    v: torch.Tensor,         # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    threshold_buf: torch.Tensor | None = None,
    return_skip_ratio: bool = False,
    th_use_fp8: bool = True,   # True: 阈值计算用 k_hi8；False: 用 k_fp16
    ker_use_fp8: bool = True,  # True: 主 kernel 用 k_hi8；False: 用 k_fp16
):
    # 同时存储并传入 k_hi8 / k_lo8 / k_fp16；实际计算按开关选择 k_hi8 或 k_fp16。
    assert q.is_cuda and k_hi8.is_cuda and k_lo8.is_cuda and k_fp16.is_cuda and v.is_cuda
    assert q.ndim == 2 and k_hi8.ndim == 3 and k_lo8.ndim == 3 and k_fp16.ndim == 3 and v.ndim == 3
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    HKV2, T2, Kk2 = k_lo8.shape
    HKV3, T3, Kk3 = k_fp16.shape
    Tv, HKVv, V = v.shape
    assert Kk == K and Kk2 == K and Kk3 == K
    assert T2 == T and T3 == T and Tv == T
    assert HKV2 == HKV and HKV3 == HKV and HKVv == HKV
    # 类型约束
    # assert k_hi8.dtype == torch.float8_e5m2
    # assert k_lo8.dtype == torch.uint8
    # assert k_fp16.dtype == torch.float16
    assert HQ % HKV == 0
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o   = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)

    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # 阈值：按开关选择 k_hi8 或 k_fp16
    if threshold_buf is None:
        k_for_th = k_hi8 if th_use_fp8 else k_fp16
        threshold_buf = compute_attn_thresholds(
            q, k_for_th,
            scale=scale, BS=BS, delta=delta,
        )
    else:
        assert threshold_buf.shape == (HQ,) and threshold_buf.dtype == torch.float32 and threshold_buf.device == q.device

    # Stage 1：按开关选择 k_hi8 或 k_fp16
    k_for_kernel = k_hi8 if ker_use_fp8 else k_fp16
    attn_forward_stage1_pruned[(HKV, NTB)](
        q, k_for_kernel, v,
        m_buf, l_buf, o_buf,
        threshold_buf, mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        # num_stages=3, num_warps=8,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    attn_forward_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o, NTBS,
        HKV=HKV, G=G, HQ=HQ, V=V,
    )

    if return_skip_ratio:
        return o, skip_ratio
    else:
        return o


def convert_to_triton_layout(q_rope_1, k_rope, v):
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


def pack_k_hi_lo(k_fp16: torch.Tensor):
    # 输入 k_fp16: [HKV, T, K], float16
    # 输出：
    # - k_hi8: [HKV, T, K], float8_e5m2（取 fp16 的高字节，并按 e5m2 解释）
    # - k_lo8: [HKV, T, K], uint8（取 fp16 的低字节，原始 bits）
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


def flash_attn_compute(q_rope_1, k_rope, v):
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


def benchmark(fn, iters=50, warmup=10):
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

    torch.set_float32_matmul_precision("high")

    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result'

    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_57'
    
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_8k'
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_51_32k'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_58_128k'
 
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256
    SBS = 256
    delta = 8.0

    KERNEL_USE_FP8 = False   # True: 主 kernel 用 8bit（k_hi8）；False: 用 16bit（k_fp16）
    PRECOMPUTE_THRESHOLD = True
    THRESH_USE_FP8 = False   # True: 阈值计算用 8bit（k_hi8）；False: 用 16bit（k_fp16）
    

    print(f"{BS=} {SBS=} {delta=} {KERNEL_USE_FP8=} {THRESH_USE_FP8=}")

    # 计时参数
    iters = 1000
    warmup = 1000

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        if layer_idx == 0:
            continue
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)  # [B, Hq, T, D]
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)  # [B, Hkv, T, D]
        v      = layer_qkvh_data["v"].to('cuda', dtype=dtype)       # [B, Hkv, T, Dv]

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
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        scale = 1.0 / math.sqrt(D)

        # 将 k 拆分为高/低字节存储，同时保留完整的 k_fp16
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)
        k_full = k_triton_fp16  # 完整 k
        
        torch.cuda.synchronize()
            
        k_for_th = k_hi8 if THRESH_USE_FP8 else k_full
        def run_th():
            return compute_attn_thresholds(
                q_triton, k_for_th,
                scale=scale, BS=BS, delta=delta,
            )

        ms_th = benchmark(run_th, iters=iters, warmup=warmup)
        threshold_buf = run_th()
        print(f"Precompute threshold: {ms_th:.3f} ms")
        
        if not PRECOMPUTE_THRESHOLD:
            threshold_buf = None
        
        torch.cuda.synchronize()

        o_triton, skip_ratio = attn_forward_q1_b1_splitT(
            q_triton,
            k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_full,
            v=v_triton,
            scale=scale, BS=BS, SBS=SBS,
            delta=delta,
            threshold_buf=threshold_buf,
            return_skip_ratio=True,
            th_use_fp8=THRESH_USE_FP8,
            ker_use_fp8=KERNEL_USE_FP8,
        )
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")

        o_flash = flash_attn_compute(q_rope_1, k_rope, v)  # [Hq, V]

        # 数值对比（与 Flash 输出）
        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        rel = (o_triton.float() - o_flash.float()).abs().max() / (o_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # 性能对比
        def run_triton():
            o = attn_forward_q1_b1_splitT(
                q_triton,
                k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_full,
                v=v_triton,
                scale=scale, BS=BS, SBS=SBS,
                threshold_buf=threshold_buf,
                return_skip_ratio=False,
                th_use_fp8=THRESH_USE_FP8,
                ker_use_fp8=KERNEL_USE_FP8,
            )
            return o

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        ms_triton = benchmark(run_triton, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        if layer_idx >= 3:
            break