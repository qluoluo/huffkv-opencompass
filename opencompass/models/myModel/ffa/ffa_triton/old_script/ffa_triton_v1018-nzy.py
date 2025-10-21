import os
# os.environ["CUDA_VISIABLE_DEVICES"] = "4"

# os.environ["TRITON_DUMP_ASSEMBLY"] = "1"
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache_fp8")

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


# ==============================================================================
# 原始实现（为了对比）
# ==============================================================================

@triton.jit
def attn_compute_threshold_two_blocks(
    q, k_hi8, threshold_buf, scale, T, NTB, delta,
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
    kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(kb_ptrs0, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # tb = NTB-1（当 NTB==1 时等于 0，与 tb0 相同）
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(kb_ptrs1, mask=(tl.full([K], True, tl.int1)[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    m2 = tl.maximum(m0, m1)
    th = m2 - delta
    tl.store(threshold_buf + (base_hq + rows), th, mask=row_mask)


def compute_attn_thresholds(
    q: torch.Tensor, k_hi8: torch.Tensor, scale: float, BS: int, delta: float = 1000.0,
):
    BS=16
    
    assert q.is_cuda and k_hi8.is_cuda
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    assert Kk == K and (HQ % HKV == 0)
    G = HQ // HKV
    NTB = triton.cdiv(T, BS)
    threshold_buf = torch.empty((HQ,), device=q.device, dtype=torch.float32)
    grid_th = (HKV, 1)
    attn_compute_threshold_two_blocks[grid_th](
        q, k_hi8, threshold_buf, scale, T, NTB, delta, HKV=HKV, HQ=HQ, K=K, G=G, BS=BS
    )
    return threshold_buf


@triton.jit
def attn_forward_stage1_pruned(
    q, k_hi8, v, m_buf, l_buf, o_buf, threshold_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G
    s0 = pid_tb * BS
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)
    NSB: tl.constexpr = (BS + SBS - 1) // SBS
    th_rows = tl.load(threshold_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T
        kb_ptrs = k_hi8 + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)
        b_s = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)
        m_rows_blk = tl.max(b_s_act, axis=1)
        below = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid
        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)
        if not prune_blk:
            m_rows = m_rows_blk
            b_p = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)
            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o, NTBS,
    HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr,
):
    pid_hkv = tl.program_id(0)
    g = tl.program_id(1)
    pid_hq = pid_hkv * G + g
    v_offs = tl.arange(0, V)
    neg_inf = tl.full((), float('-inf'), tl.float32)
    b_m = neg_inf
    b_acc = tl.zeros((), tl.float32)
    b_o = tl.zeros([V], tl.float32)
    for tb in range(0, NTBS):
        keep = tl.load(mask_buf + pid_hkv * NTBS + tb).to(tl.int1)
        if keep:
            m_b = tl.load(m_buf + pid_hq * NTBS + tb)
            l_b = tl.load(l_buf + pid_hq * NTBS + tb)
            o_b = tl.load(o_buf + pid_hq * (NTBS * V) + tb * V + v_offs)
            new_m = tl.maximum(b_m, m_b)
            r_prev = tl.exp2(b_m - new_m)
            r_blk = tl.exp2(m_b - new_m)
            b_acc = b_acc * r_prev + l_b * r_blk
            b_o = b_o * r_prev + o_b * r_blk
            b_m = new_m
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)
    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def attn_forward_q1_b1_splitT(
    q: torch.Tensor, k_hi8: torch.Tensor, k_lo8: torch.Tensor, v: torch.Tensor,
    scale: float = None, BS: int = 128, SBS: int | None = None, delta: float = 5.0,
    threshold_buf: torch.Tensor | None = None, return_skip_ratio: bool = False,
):
    HQ, K = q.shape
    HKV, T, _ = k_hi8.shape
    _, _, V = v.shape
    G = HQ // HKV
    if scale is None: scale = 1.0 / math.sqrt(K)
    if SBS is None: SBS = BS
    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    if threshold_buf is None:
        threshold_buf = compute_attn_thresholds(q, k_hi8, scale=scale, BS=BS, delta=delta)

    attn_forward_stage1_pruned[(HKV, NTB)](
        q, k_hi8, v, m_buf, l_buf, o_buf, threshold_buf, mask_buf,
        scale, T, NTB, NTBS, HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS
    )
    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    attn_forward_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf, mask_buf, o, NTBS, HKV=HKV, G=G, HQ=HQ, V=V
    )
    if return_skip_ratio: return o, skip_ratio
    else: return o


# ==============================================================================
# 新的融合实现
# ==============================================================================

@triton.jit
def attn_forward_stage1_fused_threshold(
    q, k_hi8, v,               # 仅传入 k_hi8（fp8_e5m2*）
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta, # threshold_buf 被 delta 替代
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

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

    # --------------------------------------------------------------------------
    # 1. 在内核开始时计算阈值
    # --------------------------------------------------------------------------
    # tb = 0
    tb0 = 0
    offs_t0 = tb0 * BS + tl.arange(0, BS)
    t_mask0 = offs_t0 < T
    kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)

    # tb = NTB-1
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)

    m2 = tl.maximum(m0, m1)
    th_rows = m2 - delta  # 阈值现在是寄存器中的一个 tl.tensor

    # --------------------------------------------------------------------------
    # 2. 处理当前程序负责的块 (pid_tb)，并使用上面计算的阈值
    # --------------------------------------------------------------------------
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 循环每个子块
    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

        # 使用在寄存器中的 th_rows 进行裁剪
        below   = (m_rows_blk < th_rows) & row_mask
        n_below = tl.sum(below.to(tl.int32), axis=0)
        n_valid = tl.sum(row_mask.to(tl.int32), axis=0)
        prune_blk = n_below == n_valid

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)
            l_rows = tl.sum(b_p, axis=1)

            need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
            o_tile = tl.zeros([BM_DOT, V], tl.float32)
            if need_v:
                v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
                o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)

            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]
            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


def attn_forward_fused(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [HKV, T, K], float8_e5m2
    k_lo8: torch.Tensor,  # [HKV, T, K], uint8
    v: torch.Tensor,      # [T, HKV, V]
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
):
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    Tv, HKVv, V = v.shape
    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    m_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q.device, dtype=torch.int8)

    # Stage 1: 直接调用融合了阈值计算的内核
    attn_forward_stage1_fused_threshold[(HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta, # 传入 delta
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2: 保持不变
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


# ==============================================================================
# 通用工具函数和主函数
# ==============================================================================

def convert_to_triton_layout(q_rope_1, k_rope, v):
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == 1 and qlen == 1
    q_triton = q_rope_1[0, :, 0, :].contiguous()
    k_triton = k_rope[0, :, :, :].contiguous()
    v_triton = v[0, :, :, :].permute(1, 0, 2).contiguous()
    return q_triton, k_triton, v_triton


def pack_k_hi_lo(k_fp16: torch.Tensor):
    k_fp16 = k_fp16.contiguous()
    k_hi8 = k_fp16.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k_fp16.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8


def flash_attn_compute(q_rope_1, k_rope, v):
    from flash_attn import flash_attn_func
    out = flash_attn_func(q_rope_1.transpose(1, 2), k_rope.transpose(1, 2), v.transpose(1, 2), causal=False)
    return out.squeeze(0).squeeze(0)


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

    # 请替换为你的路径
    # exp_root_dir = '/inspire/hdd/project/heziweiproject/heziwei-25044/projects_zyning/huffkv-opencompass/opencompass/models/myModel/bucket_attn/attn_analysis/result'
    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result'
    
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_58_128k'
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256
    SBS = 256
    delta = 5.0
    iters = 1000
    warmup = 1000

    print(f"BS={BS}, SBS={SBS}, delta={delta}")

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        if layer_idx == 0: continue
        print(f"\n========== Layer {layer_idx} ==========")
        q_rope = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)
        k_rope = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)
        v = layer_qkvh_data["v"].to('cuda', dtype=dtype)
        q_rope_1 = q_rope[:, :, -1:, :]

        _, Hq, _, D = q_rope_1.shape
        _, Hkv, T, _ = k_rope.shape
        _, _, _, Dv = v.shape
        print(f"T={T} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv}")

        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        scale = 1.0 / math.sqrt(D)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        # --- 验证数值正确性 ---
        o_fused, skip_ratio = attn_forward_fused(
            q_triton, k_hi8, k_lo8, v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True
        )
        print(f"Skipped block ratio (fused): {skip_ratio:.3%}")

        o_flash = flash_attn_compute(q_rope_1, k_rope, v)

        max_abs = (o_fused.float() - o_flash.float()).abs().max().item()
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}")
        assert max_abs < 10, "Fused implementation has a significant correctness issue."

        # --- 性能对比 ---
        def run_original():
            return attn_forward_q1_b1_splitT(
                q_triton, k_hi8, k_lo8, v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta,
                threshold_buf=None, return_skip_ratio=False
            )

        def run_fused():
            return attn_forward_fused(
                q_triton, k_hi8, k_lo8, v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False
            )

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        # 预热并运行
        ms_original = benchmark(run_original, iters=iters, warmup=warmup)
        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

        print(f"Speed (Original): {ms_original:.3f} ms")
        print(f"Speed (Fused)   : {ms_fused:.3f} ms, Speedup vs Original: {ms_original/ms_fused:.2f}x")
        print(f"Speed (Flash)   : {ms_flash:.3f} ms, Ratio vs Flash: {ms_fused/ms_flash:.2f}x")

        break # 只测试一个 layer