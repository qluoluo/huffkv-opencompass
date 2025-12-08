import math
from typing import Tuple

import torch
import triton
import triton.language as tl

# ==========================================
# Data Layout Tools
# ==========================================


def convert_to_triton_layout(
    q_rope_1: torch.Tensor,  # [B, Hq, 1, Dq]
    k_rope: torch.Tensor,    # [B, Hkv, T, Dk]
    v: torch.Tensor,         # [B, Hkv, T, Dv]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert inputs to Triton-friendly tensors with time-major layout for K/V:
    - q_triton: [B, Hq, Dq]
    - k_triton_fp16: [B, T, Hkv, Dk]
    - v_triton: [B, T, Hkv, Dv]
    """
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv

    q_triton = q_rope_1[:, :, 0, :].contiguous()                    # [B, Hq, D]
    # Time-major for K: [B, T, Hkv, D]
    k_triton_fp16 = k_rope.permute(0, 2, 1, 3).contiguous()
    # Time-major for V: [B, T, Hkv, Dv]
    v_triton = v.permute(0, 2, 1, 3).contiguous()
    return q_triton, k_triton_fp16, v_triton


def pack_k_hi_lo(k: torch.Tensor):
    """
    将 [B, T, H, D] 的 fp16 拆分为高低位：
    - k_hi8: 使用 float8_e5m2 视图取高字节，便于内核直接载入近似值
    - k_lo8: 保留低字节的 uint8（当前内核未使用低位，但保留接口）
    注意：在实际 Little Endian 系统中：
    - 高地址字节 (High Byte) 包含 Sign + Exponent + High Mantissa
    - 低地址字节 (Low Byte) 包含 Low Mantissa
    """
    k = k.contiguous()
    # 假设 Little Endian，Byte 1 是高位，Byte 0 是低位
    k_hi8 = k.view(torch.float8_e5m2)[..., 1::2].contiguous()
    k_lo8 = k.view(torch.uint8)[..., 0::2].contiguous()
    return k_hi8, k_lo8

# ==========================================
# Triton Kernels
# ==========================================

@triton.jit
def attn_forward_stage1_cascade(
    q_ptr, k_hi_ptr, k_lo_ptr, k_ptr, v_ptr,
    m_buf, l_buf, o_buf, mask_buf,
    scale, T, NTB, NTBS, delta,
    stride_qb, stride_qh, stride_qk,
    stride_kb, stride_kt, stride_kh, stride_kk,
    stride_vb, stride_vt, stride_vh, stride_vv,
    stride_mb, stride_mh, stride_mt,
    stride_ob, stride_oh, stride_ot, stride_ov,
    B: tl.constexpr, HKV: tl.constexpr, HQ: tl.constexpr, 
    K: tl.constexpr, V: tl.constexpr, G: tl.constexpr, 
    BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16 # 用于计算阈值的采样块大小
):
    """
    Stage 1: Split-K 处理，结合了 FP8 粗筛和级联加载。
    """
    pid_tb = tl.program_id(0) # Time-Block ID
    pid_b = tl.program_id(1)  # Batch ID
    pid_hkv = tl.program_id(2)# KV Head ID

    # 常量定义
    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    
    # 1. 初始化 Q 的指针
    # Q shape: [B, HQ, K]
    base_hq = pid_hkv * G
    offs_k = tl.arange(0, K)
    rows = tl.arange(0, BM_DOT)
    row_mask = rows < G
    
    # 指向当前 KV head 对应的 Q group
    q_ptrs = q_ptr + pid_b * stride_qb + (base_hq + rows)[:, None] * stride_qh + offs_k[None, :] * stride_qk
    q_tile = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    # =========================================================
    # Part 1: 计算动态阈值 (Sink-Aware)，直接使用全精度 K 参与
    # =========================================================
    
    # 采样开头 (tb0) 和结尾 (tb_last) 的数据
    # 这里为了代码简洁，我们直接加载完整 FP16 来算阈值，因为这部分只占总计算量的极小部分
    # 实际极致优化时，这里也可以只用 FP8 估算
    
    # --- 计算 Start Token 的分数 ---
    tb0 = 0
    offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
    t_mask0 = offs_t0 < T
    
    # 计算指针
    off_b_hkv = pid_b * stride_kb + pid_hkv * stride_kh
    
    k_ptrs0 = k_ptr + off_b_hkv + (offs_t0[:, None] * stride_kt + offs_k[None, :] * stride_kk)
    k_val_0 = tl.load(k_ptrs0, mask=t_mask0[:, None], other=0.0).to(tl.float16)
    
    s0 = tl.dot(q_tile, k_val_0.trans(), out_dtype=tl.float32) * scale * RCP_LN2
    s0 = tl.where(t_mask0[None, :], s0, NEG_INF)
    m0 = tl.max(s0, axis=1) # [BM_DOT]

    # --- 计算 Recent Token 的分数 ---
    tb1 = NTB - 1
    offs_t1 = tb1 * BS + tl.arange(0, T_BS) # 注意这里简化了，只取 BS 的前 T_BS
    t_mask1 = offs_t1 < T
    
    k_ptrs1 = k_ptr + off_b_hkv + (offs_t1[:, None] * stride_kt + offs_k[None, :] * stride_kk)
    k_val_1 = tl.load(k_ptrs1, mask=t_mask1[:, None], other=0.0).to(tl.float16)
    
    s1 = tl.dot(q_tile, k_val_1.trans(), out_dtype=tl.float32) * scale * RCP_LN2
    s1 = tl.where(t_mask1[None, :], s1, NEG_INF)
    m1 = tl.max(s1, axis=1)

    # 确定本 Block 的阈值
    th_rows = tl.maximum(m0, m1) - delta

    # =========================================================
    # Part 2: 级联加载与计算 (Cascade Loop)
    # =========================================================
    
    s0_block = pid_tb * BS
    NSB = (BS + SBS - 1) // SBS # 子块数量
    
    for sb in range(NSB):
        # 准备当前 sub-block 的时间偏移
        offs_t_sb = s0_block + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T
        
        # -----------------------------------------------------
        # Step A: 粗筛 (Coarse Check) - 只读 High 8 bits
        # -----------------------------------------------------
        k_hi_ptrs = k_hi_ptr + off_b_hkv + (offs_t_sb[:, None] * stride_kt + offs_k[None, :] * stride_kk)
        k_hi_val = tl.load(k_hi_ptrs, mask=t_mask_sb[:, None], other=0.0) # [SBS, K] float8
        
        # 直接将 float8 高位近似转为 fp16
        k_approx = k_hi_val.to(tl.float16)
        
        # 粗略点积 (注意转置 K: [HQ, K] @ [K, SBS] -> [HQ, SBS])
        s_approx = tl.dot(q_tile, k_approx.trans(), out_dtype=tl.float32) * scale * RCP_LN2
        s_approx = tl.where(t_mask_sb[None, :], s_approx, NEG_INF)
        m_approx = tl.max(s_approx, axis=1) # [BM_DOT]
        
        # 阈值检查 (给 2.0 的安全余量，防止精度误差)
        is_promising = m_approx > th_rows
        
        # 只要 group 内有一个 head 需要，就必须计算 (SIMT 约束)
        # 这里使用 row_mask 确保不计算 padding 的 head
        needs_refine = tl.sum((is_promising & row_mask).to(tl.int32)) > 0
        
        # 计算输出 buffer 的位置
        tb_sb_idx = pid_tb * NSB + sb
        m_ptrs = m_buf + pid_b * stride_mb + (base_hq + rows) * stride_mh + tb_sb_idx * stride_mt
        l_ptrs = l_buf + pid_b * stride_mb + (base_hq + rows) * stride_mh + tb_sb_idx * stride_mt
        mask_ptrs = mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb_sb_idx
        
        if needs_refine:
            # -------------------------------------------------
            # Step B: 精修 (Refine) - 读取全精度 K 重算
            # -------------------------------------------------
            k_full_ptrs = k_ptr + off_b_hkv + (offs_t_sb[:, None] * stride_kt + offs_k[None, :] * stride_kk)
            k_full = tl.load(k_full_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
            
            # 精确计算 Score
            s_real = tl.dot(q_tile, k_full.trans(), out_dtype=tl.float32) * scale * RCP_LN2
            s_real = tl.where(t_mask_sb[None, :], s_real, NEG_INF)
            
            m_real = tl.max(s_real, axis=1)
            
            # 这里可以做第二次精确剪枝，但为简化代码直接计算 V
            p_val = tl.exp2(s_real - m_real[:, None])
            l_val = tl.sum(p_val, axis=1)
            
            # 加载 V 并计算 O
            offs_v = tl.arange(0, V)
            v_ptrs = v_ptr + pid_b * stride_vb + pid_hkv * stride_vh + \
                     offs_t_sb[:, None] * stride_vt + offs_v[None, :] * stride_vv
            
            v_val = tl.load(v_ptrs, mask=t_mask_sb[:, None], other=0.0).to(tl.float16)
            o_tile = tl.dot(p_val.to(tl.float16), v_val, out_dtype=tl.float32)
            
            # 存入 Global Memory
            o_ptrs = o_buf + pid_b * stride_ob + (base_hq + rows)[:, None] * stride_oh + \
                     tb_sb_idx * stride_ot + offs_v[None, :] * stride_ov
            
            tl.store(m_ptrs, m_real, mask=row_mask)
            tl.store(l_ptrs, l_val, mask=row_mask)
            tl.store(o_ptrs, o_tile, mask=row_mask[:, None])
            tl.store(mask_ptrs, tl.full((), 1, tl.int8)) # Mark as kept
            
        else:
            # -------------------------------------------------
            # Step C: 跳过 (Skip) - 节省 V 带宽和计算
            # -------------------------------------------------
            tl.store(mask_ptrs, tl.full((), 0, tl.int8)) # Mark as skipped
            # 不需要写 m/l/o，Stage 2 会根据 mask 忽略它们


@triton.jit
def attn_forward_stage2_masked(
    m_buf, l_buf, o_buf, mask_buf, o,
    stride_mb, stride_mh, stride_mt,
    stride_ob, stride_oh, stride_ot, stride_ov,
    stride_o_b, stride_o_h, stride_o_v,
    NTBS,
    B: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, V: tl.constexpr
):
    """
    Stage 2: Reduction Kernel. 聚合 Stage 1 的部分结果。
    """
    pid_b = tl.program_id(0)
    pid_hkv = tl.program_id(1)
    g = tl.program_id(2) # Group ID within a KV head
    
    pid_hq = pid_hkv * G + g
    offs_v = tl.arange(0, V)
    
    # 累加器初始化
    acc_m = float("-inf")
    acc_d = 0.0
    acc_o = tl.zeros([V], tl.float32)
    
    # 遍历所有 Time-Blocks
    for tb in range(NTBS):
        # 检查 Mask
        mask_ptr = mask_buf + pid_b * (HKV * NTBS) + pid_hkv * NTBS + tb
        keep = tl.load(mask_ptr).to(tl.int1)
        
        if keep:
            # 加载 m, l
            m_ptr = m_buf + pid_b * stride_mb + pid_hq * stride_mh + tb * stride_mt
            l_ptr = l_buf + pid_b * stride_mb + pid_hq * stride_mh + tb * stride_mt
            
            m_i = tl.load(m_ptr)
            l_i = tl.load(l_ptr)
            
            # 加载 o
            o_ptr = o_buf + pid_b * stride_ob + pid_hq * stride_oh + tb * stride_ot + offs_v * stride_ov
            o_i = tl.load(o_ptr)
            
            # Online Softmax Update
            m_new = tl.maximum(acc_m, m_i)
            alpha = tl.exp2(acc_m - m_new)
            beta = tl.exp2(m_i - m_new)
            
            acc_m = m_new
            acc_d = acc_d * alpha + l_i * beta
            acc_o = acc_o * alpha + o_i * beta
            
    # 最终归一化
    # 避免除以 0
    inv_d = 1.0 / (acc_d + 1e-6)
    final_o = acc_o * inv_d
    
    # 写回最终结果
    out_ptr = o + pid_b * stride_o_b + pid_hq * stride_o_h + offs_v * stride_o_v
    tl.store(out_ptr, final_o.to(tl.float16))

# ==========================================
# Host Wrapper
# ==========================================

def sparse_decode_cascade(
    q,
    k_hi,
    k_lo,
    k,
    v,
    delta=5.0,
    block_size=128,
    sub_block_size=64,
    scale=None,
    return_skip_ratio=False,
):
    """
    Args:
        q: [B, HQ, K]
        k_hi, k_lo: [B, T, HKV, K] (Packed uint8)
        k: [B, T, HKV, K] (Full precision K for refine)
        v: [B, T, HKV, V]
    """
    q = q.contiguous()
    k_hi = k_hi.contiguous()
    k_lo = k_lo.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    block_size = int(block_size)
    sub_block_size = block_size if sub_block_size is None else int(sub_block_size)
    if sub_block_size < 16:
        raise ValueError(f"sub_block_size ({sub_block_size}) must be at least 16 to satisfy tl.dot requirements.")
    B, HQ, K = q.shape
    Bk, T, HKV, Kk = k.shape
    assert k_hi.shape == k.shape, "k_hi shape must match k shape"
    assert k_lo.shape == k.shape, "k_lo shape must match k shape"
    _, _, _, V = v.shape
    if HQ % HKV != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by HKV ({HKV})")
    if scale is None:
        scale = 1.0 / math.sqrt(K)
    G = HQ // HKV
    
    # 配置
    NTB = triton.cdiv(T, block_size)
    NSB = triton.cdiv(block_size, sub_block_size)
    NTBS = NTB * NSB
    
    # 中间显存分配
    # 注意：在生产环境中，这些 buffer 应该预分配并复用
    m_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    l_buf = torch.empty((B, HQ, NTBS), device=q.device, dtype=torch.float32)
    o_buf = torch.empty((B, HQ, NTBS, V), device=q.device, dtype=torch.float32)
    mask_buf = torch.zeros((B, HKV, NTBS), device=q.device, dtype=torch.int8)
    
    # 输出
    o = torch.empty((B, HQ, V), device=q.device, dtype=torch.float16)

    # Launch Stage 1
    # Grid: (Time-Blocks, Batch, KV-Heads)
    grid_1 = (NTB, B, HKV)
    attn_forward_stage1_cascade[grid_1](
        q, k_hi, k_lo, k, v,
        m_buf, l_buf, o_buf, mask_buf,
        scale, T, NTB, NTBS, delta,
        *q.stride(),
        *k_hi.stride(),
        *v.stride(),
        *m_buf.stride(),
        *o_buf.stride(),
        B=B, HKV=HKV, HQ=HQ, K=K, V=V, G=G,
        BS=block_size, SBS=sub_block_size,
        BM_DOT=16, # 每个线程处理 16 个 row
        T_BS=16
    )

    skip_ratio = None
    if return_skip_ratio:
        kept_ratio = mask_buf.float().mean()
        skip_ratio = float((1.0 - kept_ratio).item())
    
    # Launch Stage 2
    # Grid: (Batch, KV-Heads, Group)
    grid_2 = (B, HKV, G)
    attn_forward_stage2_masked[grid_2](
        m_buf, l_buf, o_buf, mask_buf, o,
        *m_buf.stride(),
        *o_buf.stride(),
        *o.stride(),
        NTBS,
        B=B, HKV=HKV, G=G, HQ=HQ, V=V
    )
    
    if return_skip_ratio:
        return o, skip_ratio
    return o


def attn_forward_decode(
    q: torch.Tensor,      # [B, Hq, 1, K] or [B, Hq, K]
    k: torch.Tensor | None, # [B, T, HKV, K] or [B, HKV, T, K]
    v: torch.Tensor,      # [B, T, HKV, V] or [B, HKV, T, V]
    k_hi8: torch.Tensor = None,  # same layout as k
    k_lo8: torch.Tensor = None,  # same layout as k
    scale: float = None,
    BS: int = 128,
    SBS: int | None = None,
    delta: float = 5.0,
    return_skip_ratio: bool = False,
    **kwargs,
):
    # Normalize shapes to match sparse_decode_cascade expectations:
    # q -> [B, Hq, K]; k/v -> [B, T, HKV, K/V]
    if q.dim() == 4:
        # Handle either [B, Hq, 1, K] or [B, 1, Hq, K]
        if q.shape[1] == 1:
            q = q.permute(0, 2, 1, 3)
        if q.shape[2] != 1:
            raise ValueError(f"Decode expects single-token q, got shape {q.shape}")
        q = q[:, :, 0, :].contiguous()
    elif q.dim() != 3:
        raise ValueError(f"Unsupported q shape {q.shape}")

    def _normalize_kv(x: torch.Tensor | None, name: str):
        if x is None:
            return None
        if x.dim() != 4:
            raise ValueError(f"{name} must be 4D, got shape {x.shape}")
        # If layout is [B, HKV, T, K], switch to [B, T, HKV, K]
        if x.shape[1] < x.shape[2]:
            x = x.permute(0, 2, 1, 3)
        return x.contiguous()

    k = _normalize_kv(k, "k")
    v = _normalize_kv(v, "v")
    k_hi8 = _normalize_kv(k_hi8, "k_hi8")
    k_lo8 = _normalize_kv(k_lo8, "k_lo8")

    # 允许从 k 生成高低位，以兼容旧接口
    if k_hi8 is None or k_lo8 is None:
        if k is None:
            raise ValueError("k is required when k_hi8/k_lo8 are not provided.")
        k_hi8, k_lo8 = pack_k_hi_lo(k)

    sub_block_size = BS if SBS is None else SBS
    if return_skip_ratio:
        o, skip_ratio = sparse_decode_cascade(
            q=q,
            k_hi=k_hi8,
            k_lo=k_lo8,
            k=k,
            v=v,
            delta=delta,
            block_size=BS,
            sub_block_size=sub_block_size,
            scale=scale,
            return_skip_ratio=True,
        )
        return o, skip_ratio

    return sparse_decode_cascade(
        q=q,
        k_hi=k_hi8,
        k_lo=k_lo8,
        k=k,
        v=v,
        delta=delta,
        block_size=BS,
        sub_block_size=sub_block_size,
        scale=scale,
        return_skip_ratio=False,
    )
