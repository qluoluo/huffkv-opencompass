import os
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache")
# os.environ['TRITON_DUMP_ASSEMBLY'] = "1"

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


def precompute_attn_thresholds(
    q: torch.Tensor,    # [HQ, K] 或 [1, Hq, 1, K]
    k: torch.Tensor,    # [HKV, T, K] 或 [1, HKV, T, K]
    scale: float,
    BS: int,
    NTB: int,
    delta: float,
    *,
    head_n: int = 5,          # 头部 token 数
    tail_n: int = 0,          # 尾部 token 数
    dedup: bool = True,       # 是否对 head/tail 索引做去重
    BM_DOT: int = 16,         # 保留参数以兼容原签名（未使用）
):
    """
    纯 torch 版本：用头部 head_n 与尾部 tail_n 的 token 计算每个 Q 行的最大得分，再减 delta 得到阈值。
    返回 thres_buf: [HQ], float32, 在 q.device 上。
    """
    # 规范到 2D/3D 视图
    if q.ndim == 4:
        B, Hq, qlen, Dq = q.shape
        assert B == 1 and qlen == 1, "q 应为 [HQ,K] 或 [1,Hq,1,K]"
        q = q[0, :, 0, :]
    elif q.ndim != 2:
        raise AssertionError("q must be [HQ,K] or [1,Hq,1,K]")

    if k.ndim == 4:
        Bk, Hkv, T, Dk = k.shape
        assert Bk == 1, "k 应为 [HKV,T,K] 或 [1,HKV,T,K]"
        k = k[0, :, :, :]
    elif k.ndim != 3:
        raise AssertionError("k must be [HKV,T,K] or [1,HKV,T,K]")

    device = q.device
    HQ, K = q.shape
    HKV, T, K2 = k.shape
    assert K == K2, "q/k head_dim mismatch"
    assert HQ % HKV == 0, "GQA requires HQ % HKV == 0"
    G = HQ // HKV

    # 选取要看的时间步索引
    head_n_eff = int(min(max(head_n, 0), T))
    tail_n_eff = int(min(max(tail_n, 0), T))
    idx_list = []
    if head_n_eff > 0:
        idx_list.append(torch.arange(0, head_n_eff, device=device, dtype=torch.long))
    if tail_n_eff > 0:
        idx_list.append(torch.arange(T - tail_n_eff, T, device=device, dtype=torch.long))
    if len(idx_list) == 0:
        # 不剪枝：全 -inf
        return torch.full((HQ,), float("-inf"), device=device, dtype=torch.float32)

    idx = torch.cat(idx_list)
    if dedup:
        idx = torch.unique(idx)  # 去重，避免重复计算

    # 把 q 以 GQA 分组视图成 [HKV, G, K]（用 as_strided 避免复制）
    SQH, SQK = q.stride()
    q_group = q.as_strided(size=(HKV, G, K),
                           stride=(G * SQH, SQH, SQK))  # 只重解释 stride，不拷贝

    # 取出所需的 k 切片 [HKV, S, K]
    k_sel = k.index_select(dim=1, index=idx)  # S = idx.numel()

    # 计算分数（float32 累加，log2 空间）
    RCP_LN2 = 1.4426950408889634
    # [HKV, G, S] = [HKV,G,K] @ [HKV,K,S]
    scores = torch.matmul(
        q_group.to(torch.float32),
        k_sel.to(torch.float32).transpose(1, 2)
    ) * (scale * RCP_LN2)

    # 每行最大 -> 阈值
    mmax = scores.max(dim=2).values  # [HKV, G]
    thres_buf = (mmax - delta).reshape(HQ).to(torch.float32).contiguous()
    return thres_buf


@triton.jit
def attn_decode_stage1_pruned(
    q, k, k_bytes, v,                # k_bytes: 外部提供的 float8_e5m2（连续），当 USE_FP8_K=True
    m_buf, l_buf, o_buf,
    thres_buf, mask_buf,
    scale, T, NTB, NTBS,
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    # strides (in elements; runtime ints)
    SQH, SQK, SKH, SKT, SKK, SVH, SVT, SVV, SKBH, SKBT, SKBK,
    BM_DOT: tl.constexpr = 16,
    USE_FP8_K: tl.constexpr = True,  # whether to load K from float8 high-bytes
):
    pid_hkv = tl.program_id(0)  # [0, HKV)
    pid_tb  = tl.program_id(1)  # [0, NTB)
    base_hq = pid_hkv * G

    # TB start in T
    s0 = pid_tb * BS

    # rows/cols and Q tile
    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)

    # q: [HQ, K] arbitrary stride
    q_ptrs   = q + (base_hq + rows)[:, None] * SQH + offs_k[None, :] * SQK
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)  # [BM_DOT, K]

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # NSB = ceil(BS / SBS)
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        # K path
        if USE_FP8_K:
            # k_bytes: arbitrary stride float8_e5m2 view [HKV, T, K]（外部已构造且连续）
            kb_ptrs = k_bytes + pid_hkv * SKBH + offs_t_sb[None, :] * SKBT + offs_k[:, None] * SKBK
            k_tile = tl.load(
                kb_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                other=0.0
            ).to(tl.float16)  # decode to fp16
        else:
            # k fp16: arbitrary stride [HKV, T, K]
            k_ptrs = k + pid_hkv * SKH + offs_k[:, None] * SKK + offs_t_sb[None, :] * SKT
            k_tile = tl.load(
                k_ptrs,
                mask=(TRUE_K[:, None] & t_mask_sb[None, :]),
                other=0.0
            ).to(tl.float16)

        # scores (log base 2 form)
        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2  # [BM_DOT, SBS]
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        # per-row block max
        m_rows_blk = tl.max(b_s_act, axis=1)  # [BM_DOT]

        # threshold pruning
        th_rows = tl.load(thres_buf + (base_hq + rows), mask=row_mask, other=NEG_INF)
        n_keep    = tl.sum(((m_rows_blk >= th_rows) & row_mask).to(tl.int32), axis=0)
        prune_blk = n_keep == 0

        tb_sb = pid_tb * NSB + sb
        v_offs = tl.arange(0, V)

        if not prune_blk:
            # softmax within sub-block
            m_rows = m_rows_blk
            b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)  # [BM_DOT, SBS]
            l_rows = tl.sum(b_p, axis=1)

            # V path: v is [Hkv, T, V] arbitrary stride
            v_ptrs = v + pid_hkv * SVH + offs_t_sb[:, None] * SVT + v_offs[None, :] * SVV
            b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)   # [SBS, V]
            o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)                 # [BM_DOT, V]

            # write back
            m_ptrs = m_buf + (base_hq + rows) * NTBS + tb_sb
            l_ptrs = l_buf + (base_hq + rows) * NTBS + tb_sb
            o_ptrs = o_buf + (base_hq + rows)[:, None] * (NTBS * V) + tb_sb * V + v_offs[None, :]

            tl.store(m_ptrs, m_rows, mask=row_mask)
            tl.store(l_ptrs, l_rows, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])

            tl.store(mask_buf + pid_hkv * NTBS + tb_sb, tl.full((), 1, tl.int8))


@triton.jit
def attn_decode_stage2_masked(
    m_buf, l_buf, o_buf,     # [HQ, NTBS], [HQ, NTBS], [HQ, NTBS, V]
    mask_buf,                # [HKV, NTBS], int8
    o,                       # [HQ, V], out dtype = q.dtype
    NTBS: tl.constexpr,      # 编译期常量，便于静态展开循环
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

    for tb in tl.static_range(NTBS):
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

    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty, tl.zeros([V], tl.float32), b_o / b_acc)

    o_ptrs = o + pid_hq * V + v_offs
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty))


def compute_skipped_block_ratio(mask_buf: torch.Tensor) -> float:
    assert mask_buf.dtype == torch.int8
    kept = mask_buf.to(torch.int32).sum()
    total = mask_buf.numel()
    skip_ratio = 1.0 - (kept.float() / float(total))
    return float(skip_ratio.item())



def to_triton_layout_views(q_rope_1, k_rope, v):
    # Input:
    # - q_rope_1: [B, Hq, 1, D]
    # - k_rope:   [B, Hkv, T, D]
    # - v:        [B, Hkv, T, Dv]
    # Return (zero-copy views; no permute/contiguous):
    # - q_view: [HQ, K]      -> q_rope_1[0, :, 0, :]
    # - k_view: [HKV, T, K]  -> k_rope[0]
    # - v_view: [HKV, T, V]  -> v[0]   (note: NOT [T, HKV, V])
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk  = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape

    assert B == Bk == Bv == 1, "batch must be 1"
    assert qlen == 1, "qlen must be 1"
    assert Hkv == Hvv, "k/v head mismatch"
    assert Dq == Dk, "q/k head_dim mismatch"
    assert T == Tv, "k/v seq_len mismatch"
    assert Hq % Hkv == 0, "GQA requires Hq % Hkv == 0"

    q_view = q_rope_1[0, :, 0, :]      # [Hq, D]
    k_view = k_rope[0, :, :, :]        # [Hkv, T, D]
    v_view = v[0, :, :, :]             # [Hkv, T, Dv]
    return q_view, k_view, v_view


def attn_fwd_q1_b1_splitT(
    q_rope_1: torch.Tensor,   # [B, Hq, 1, D]
    k_rope:   torch.Tensor,   # [B, Hkv, T, D]
    v:        torch.Tensor,   # [B, Hkv, T, Dv]
    *,
    thres_buf = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    use_fp8_k_high_byte: bool = True,
    k_bytes_external: torch.Tensor | None = None,  # 外部构造且连续的 fp8 高字节视图
    return_skip_ratio: bool = False,
):
    """
    与原版保持相同的函数签名与返回；当 use_fp8_k_high_byte=True 时，必须提供 k_bytes_external（连续）。
    """
    assert q_rope_1.is_cuda and k_rope.is_cuda and v.is_cuda
    assert q_rope_1.ndim == 4 and k_rope.ndim == 4 and v.ndim == 4

    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk  = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape

    assert B == Bk == Bv == 1, "kernel only supports batch=1"
    assert qlen == 1,          "kernel only supports qlen=1"
    assert Hkv == Hvv,         "k/v heads must match"
    assert Dq == Dk,           "q/k head_dim mismatch"
    assert T == Tv,            "k/v seq_len mismatch"
    assert Hq % Hkv == 0,      "GQA requires Hq % Hkv == 0"

    # Zero-copy views (no permute/contiguous)
    q_view, k_view, v_view = to_triton_layout_views(q_rope_1, k_rope, v)  # q:[HQ,K], k:[HKV,T,K], v:[HKV,T,V]
    HQ, K = q_view.shape
    HKV, Tk, Kk = k_view.shape
    HKV2, Tv2, V = v_view.shape
    assert Tk == T and Tv2 == T and HKV2 == HKV and Kk == K

    G = HQ // HKV
    if SBS is None:
        SBS = BS
    assert 1 <= SBS <= BS

    scale = 1.0 / math.sqrt(K)

    NTB  = triton.cdiv(T, BS)
    NSB  = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # Precompute thresholds (uses k fp16)
    if thres_buf is None:
        thres_buf = precompute_attn_thresholds(
            q_view, k_view, scale=scale, BS=BS, NTB=NTB, delta=delta
        )

    # 使用外部提供的 k_bytes（必须，且连续）
    if use_fp8_k_high_byte:
        assert k_bytes_external is not None, "use_fp8_k_high_byte=True 时必须传入 k_bytes_external"
        assert k_bytes_external.dtype == torch.float8_e5m2, "k_bytes_external 必须是 float8_e5m2"
        assert k_bytes_external.is_cuda, "k_bytes_external 必须在 CUDA 上"
        assert k_bytes_external.is_contiguous(), "k_bytes_external 必须是连续内存（.contiguous()）"
        assert tuple(k_bytes_external.shape) == (HKV, T, K), f"k_bytes_external 形状应为 [HKV,T,K]，得到 {tuple(k_bytes_external.shape)}"
        k_bytes = k_bytes_external
        SKBH, SKBT, SKBK = k_bytes.stride()
    else:
        # 不使用 fp8 路径
        k_bytes = k_view  # 不会在 kernel 中被访问
        SKBH = SKBT = SKBK = 0

    # Buffers
    o      = torch.empty((HQ, V), device=q_view.device, dtype=q_view.dtype)
    m_buf  = torch.empty((HQ, NTBS), device=q_view.device, dtype=torch.float32)
    l_buf  = torch.empty((HQ, NTBS), device=q_view.device, dtype=torch.float32)
    o_buf  = torch.empty((HQ, NTBS, V), device=q_view.device, dtype=torch.float32)
    mask_buf = torch.zeros((HKV, NTBS), device=q_view.device, dtype=torch.int8)

    # Strides (elements)
    SQH, SQK      = q_view.stride()
    SKH, SKT, SKK = k_view.stride()
    SVH, SVT, SVV = v_view.stride()

    # Stage 1
    attn_decode_stage1_pruned[(HKV, NTB)](
        q_view, k_view, k_bytes, v_view,
        m_buf, l_buf, o_buf,
        thres_buf, mask_buf,
        scale, T, NTB, NTBS,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        SQH=SQH, SQK=SQK, SKH=SKH, SKT=SKT, SKK=SKK, SVH=SVH, SVT=SVT, SVV=SVV,
        SKBH=SKBH, SKBT=SKBT, SKBK=SKBK,
        USE_FP8_K=use_fp8_k_high_byte,
    )

    # Optional stats
    skip_ratio = None
    if return_skip_ratio:
        kept = mask_buf.to(torch.int32).sum()
        total = mask_buf.numel()
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())

    # Stage 2
    attn_decode_stage2_masked[(HKV, G)](
        m_buf, l_buf, o_buf,
        mask_buf,
        o,
        NTBS=NTBS,   # 关键字传参，匹配 tl.constexpr 形参
        HKV=HKV, G=G, HQ=HQ, V=V,
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
    from load_utils import load_attn_input

    data_root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/ffa_Attention/dump_weights/result'
    
    data_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_57'

    data_root = os.path.join(data_root_dir, data_root_subdir)
    layer_data_root = os.path.join(data_root, 'layer_data')

    dtype = torch.float16
    device = 'cuda:0'
    BS = 256
    SBS = 256
    delta = 8
    use_fp8_k_high_byte = True  # 仅在外部构造 kbytes；计时不包含构造过程

    print(f"{BS=}")

    # 计时参数
    iters = 100
    warmup = 100

    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_attn_input(layer_data_root, device=device))):
        print(f"\n========== Layer {layer_idx} ==========")
        
        if layer_idx == 0:
            continue
        
        q_proj_state, k_

        # 与 kernel 对齐的零拷贝视图
        q_view, k_view, v_view = to_triton_layout_views(q_rope_1, k_rope, v)  # q:[HQ,K], k:[HKV,T,K], v:[HKV,T,V]
        T = k_view.shape[1]
        NTB = triton.cdiv(T, BS)

        print(f"{T=}")

        thres_buf = precompute_attn_thresholds(
            q_view,            # 注意：这里传 2D 视图
            k_view,            # 以及 3D 视图
            scale=1.0 / math.sqrt(k_view.shape[-1]),
            BS=BS,
            NTB=NTB,           # 与后续 Stage 1 使用保持一致
            delta=delta,
        )

        # thres_buf = None

        # 只取最后一个查询位置 -> qlen=1
        

        # ===== 在计时之外准备连续的 kbytes（fp8 高字节视图）=====
        # 说明：这里复用你原来的“高字节视图”语义：
        #   先把 fp16 的 K 以 float8_e5m2 视图方式展开为 2*K，
        #   再取高字节通道（[..., 1::2]），并 .contiguous() 以保证连续。
        # 注意：这一步只做一次，不计入后续算子计时。
        k_view_for_bytes = k_rope[0]  # [HKV, T, K]，fp16
        # 按照原始代码的做法：把 fp16 的底层字节当作 float8_e5m2 视图，然后取高字节
        k_bytes_full = k_view_for_bytes.view(torch.float8_e5m2)  # 视图：最后一维等效变为 2*K
        k_bytes_ext = k_bytes_full[..., 1::2].contiguous()       # 取高字节并做成连续张量

        # ===== 正确性/统计（不计时）=====
        out_triton, skip_ratio = attn_fwd_q1_b1_splitT(
            q_rope_1, k_rope, v,
            thres_buf=thres_buf,
            BS=BS, SBS=SBS,
            delta=delta,
            use_fp8_k_high_byte=use_fp8_k_high_byte,
            k_bytes_external=k_bytes_ext,   # 传入外部构造的连续 kbytes
            return_skip_ratio=True,
        )

        out_flash = flash_compute(q_rope_1, k_rope, v)  # [Hq, Dv]

        # 数值对比
        max_abs = (out_triton.float() - out_flash.float()).abs().max().item()
        mean_abs = (out_triton.float() - out_flash.float()).abs().mean().item()
        rel = (out_triton.float() - out_flash.float()).abs().max() / (out_flash.float().abs().max().clamp_min(1e-6))
        rel = rel.item()

        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTB)")
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, rel={rel:.3e}")

        # ===== 性能计时：仅算子本体，不包含 kbytes 构造 =====
        def run_triton():
            return attn_fwd_q1_b1_splitT(
                q_rope_1, k_rope, v,
                thres_buf=thres_buf,
                BS=BS, SBS=SBS,
                delta=delta,
                use_fp8_k_high_byte=use_fp8_k_high_byte,
                k_bytes_external=k_bytes_ext,  # 复用外部构造的连续 kbytes
                return_skip_ratio=False,
            )

        def run_flash():
            return flash_compute(q_rope_1, k_rope, v)

        ms_triton = bench_op(run_triton, iters=iters, warmup=warmup)
        ms_flash = bench_op(run_flash, iters=iters, warmup=warmup)
        print(f"Speed: Triton={ms_triton:.3f} ms, Flash={ms_flash:.3f} ms, ratio={ms_triton/ms_flash:.2f}x")

        # 可根据需要 break