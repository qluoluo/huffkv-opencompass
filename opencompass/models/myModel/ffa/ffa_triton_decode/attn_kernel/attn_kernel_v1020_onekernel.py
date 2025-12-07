# kernels/attn_kernel.py
import math
import torch
import triton
import triton.language as tl

__all__ = ["attn_forward_single_kernel", "attn_forward"]

@triton.jit
def attn_forward_single_kernel(
    q, k_hi8, v, o,
    scale, T, NTB, NTBS, delta,
    kept_buf,                          # 用于统计保留子块数，长度 HKV，int32
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
):
    # 每个 program 处理一个 HKV 下的全部 G 行（<= BM_DOT）与全部时间块
    pid_hkv = tl.program_id(0)
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)

    # 取 q, [BM_DOT, K]
    q_ptrs = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    # 阈值：用 tb0=0 和 tb1=NTB-1 两个 TB 估计
    th_rows = tl.full([BM_DOT], NEG_INF, tl.float32)

    # tb0
    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
    k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
    b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
    b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
    m0 = tl.max(b_s0, axis=1)  # [BM_DOT]

    # tb_last
    tb1 = NTB - 1
    offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
    t_mask1 = offs_t1 < T
    kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
    k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
    b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
    b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
    m1 = tl.max(b_s1, axis=1)  # [BM_DOT]

    m2 = tl.maximum(m0, m1)
    th_rows = m2 - delta  # [BM_DOT]

    # 跨 TB×SB 的稳定累加器（逐行）
    b_m   = tl.full([BM_DOT], NEG_INF, tl.float32)  # 每行当前最大
    b_acc = tl.zeros([BM_DOT], tl.float32)          # 每行 sum(exp(·))
    b_o   = tl.zeros([BM_DOT, V], tl.float32)       # 每行输出加权和

    kept = tl.zeros((), tl.int32)                   # 统计保留的子块个数（原 mask 的 1 数量）
    v_offs = tl.arange(0, V)
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    # 遍历所有 TB
    for tb in range(0, NTB):
        s0 = tb * BS
        # 细分为 SB 子块并判剪枝
        for sb in tl.static_range(NSB):
            offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
            t_mask_sb = offs_t_sb < T

            # K[t, k] tile
            kb_ptrs = k_hi8 + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
            k_tile  = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

            # 分数 + 激活
            b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2  # [BM_DOT, SBS]
            b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

            # 该子块的每行最大
            m_rows_blk = tl.max(b_s_act, axis=1)  # [BM_DOT]

            # 剪枝判断：若该 SB 子块的所有有效行都在阈值下，则整块跳过
            below    = (m_rows_blk < th_rows) & row_mask
            n_below  = tl.sum(below.to(tl.int32), axis=0)
            n_valid  = tl.sum(row_mask.to(tl.int32), axis=0)
            prune_blk = n_below == n_valid

            if not prune_blk:
                # 块内 softmax 归一项和加权和
                m_rows = m_rows_blk
                b_p    = tl.where(t_mask_sb[None, :], tl.exp2(b_s - m_rows[:, None]), 0.0)  # [BM_DOT, SBS]
                l_rows = tl.sum(b_p, axis=1)  # [BM_DOT]

                # 需要 V 时再读 V
                need_v = tl.sum(t_mask_sb.to(tl.int32), axis=0) > 0
                o_tile = tl.zeros([BM_DOT, V], tl.float32)
                if need_v:
                    v_ptrs = v + (offs_t_sb[:, None] * (HKV * V)) + (pid_hkv * V) + v_offs[None, :]
                    b_v    = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)  # [SBS, V]
                    o_tile = tl.dot(b_p.to(tl.float16), b_v, out_dtype=tl.float32)               # [BM_DOT, V]

                # 跨块稳定归并（逐行）
                new_m  = tl.maximum(b_m, m_rows)
                r_prev = tl.exp2(b_m - new_m)
                r_blk  = tl.exp2(m_rows - new_m)
                b_acc  = b_acc * r_prev + l_rows * r_blk
                b_o    = b_o * r_prev[:, None] + o_tile * r_blk[:, None]
                b_m    = new_m

                kept += 1

    # 归一化并写回输出
    is_empty = b_acc == 0.0
    out_tile = tl.where(is_empty[:, None], tl.zeros([BM_DOT, V], tl.float32), b_o / b_acc[:, None])
    o_ptrs = o + (base_hq + rows)[:, None] * V + v_offs[None, :]
    tl.store(o_ptrs, out_tile.to(o_ptrs.dtype.element_ty), mask=row_mask[:, None])

    # 写回本 HKV 的保留子块数
    tl.store(kept_buf + pid_hkv, kept)

def attn_forward(
    q: torch.Tensor,
    k_hi8: torch.Tensor,
    k_lo8: torch.Tensor,   # 保持签名一致（此核未使用）
    k_fp16: torch.Tensor,
    v: torch.Tensor,
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
):
    HQ, K = q.shape
    HKV, T, Kk = k_hi8.shape
    Tv, HKVv, V = v.shape
    assert K == Kk and HKV == HKVv and Tv == T

    G = HQ // HKV
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)
    kept_buf = torch.zeros((HKV,), device=q.device, dtype=torch.int32)

    attn_forward_single_kernel[(HKV,)](
        q, k_hi8, v, o,
        scale, T, NTB, NTBS, delta,
        kept_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    if return_skip_ratio:
        kept = kept_buf.to(torch.int64).sum()
        total = HKV * NTBS
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())
        return o, skip_ratio
    else:
        return o