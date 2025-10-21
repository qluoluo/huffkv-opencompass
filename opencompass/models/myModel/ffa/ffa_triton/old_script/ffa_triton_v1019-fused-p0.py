import os
# os.environ["CUDA_VISIABLE_DEVICES"] = "4"

# os.environ["TRITON_DUMP_ASSEMBLY"] = "1"
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache_fp8")

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl


# 注意力计算第一阶段 - 融合阈值剪枝版本
# 实现策略：
# - 使用分布式锁机制确保每个head-KV组(pid_hkv)中只有一个线程计算阈值
# - 第一个获得锁的线程(owner)负责计算剪枝阈值并写入共享内存
# - 其他线程检查阈值是否就绪：就绪则复用，未就绪则设阈值为负无穷(禁用剪枝)
# - 避免非owner线程的自旋等待和重复计算，提高执行效率

# 关键特性：
# 1. 阈值计算：owner线程采样首尾两个token块计算最大值，减去delta得到剪枝阈值
# 2. 锁机制：使用原子CAS操作确保只有一个线程成为owner
# 3. 就绪标志：owner计算完成后设置ready标志，其他线程直接读取或跳过剪枝
# 4. 容错处理：未就绪时非owner线程禁用剪枝，确保计算正确性

@triton.jit
def attn_forward_stage1_fused_threshold(
    q, k_hi8, v,
    m_buf, l_buf, o_buf,
    mask_buf,
    scale, T, NTB, NTBS, delta,
    th_buf, th_ready, th_lock,          # th_lock 保留但不使用
    HKV: tl.constexpr, HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    G: tl.constexpr, BS: tl.constexpr, SBS: tl.constexpr,
    BM_DOT: tl.constexpr = 16,
    T_BS: tl.constexpr = 16,
    MAX_SPIN: tl.constexpr = 256,       # 保留但不使用
):
    pid_hkv = tl.program_id(0)
    pid_tb  = tl.program_id(1)
    base_hq = pid_hkv * G

    rows     = tl.arange(0, BM_DOT)
    row_mask = rows < G
    offs_k   = tl.arange(0, K)
    q_ptrs   = q + (base_hq + rows)[:, None] * K + offs_k[None, :]
    q_tile   = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0).to(tl.float16)

    RCP_LN2 = 1.4426950408889634
    NEG_INF = float("-inf")
    TRUE_K  = tl.full([K], True, tl.int1)

    ready_ptr = th_ready + pid_hkv
    th_ptrs   = th_buf   + (base_hq + rows)

    th_rows = tl.full([BM_DOT], NEG_INF, tl.float32)

    # 仅 pid_tb==0 负责计算阈值；其它 tb 在计算前检查 ready
    is_owner = pid_tb == 0
    rdy0 = tl.load(ready_ptr).to(tl.int32) == 1

    if rdy0:
        # 已算好 -> 直接读阈值
        th_rows = tl.load(th_ptrs, mask=row_mask, other=NEG_INF)
    else:
        if is_owner:
            # 只有 pid_tb==0 计算阈值并写回
            tb0 = 0
            offs_t0 = tb0 * T_BS + tl.arange(0, T_BS)
            t_mask0 = offs_t0 < T
            kb_ptrs0 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t0[None, :] * K
            k_tile0 = tl.load(kb_ptrs0, mask=(TRUE_K[:, None] & t_mask0[None, :]), other=0.0).to(tl.float16)
            b_s0 = tl.dot(q_tile, k_tile0, out_dtype=tl.float32) * scale * RCP_LN2
            b_s0 = tl.where(t_mask0[None, :], b_s0, NEG_INF)
            m0 = tl.max(b_s0, axis=1)

            tb1 = NTB - 1
            offs_t1 = tb1 * T_BS + tl.arange(0, T_BS)
            t_mask1 = offs_t1 < T
            kb_ptrs1 = k_hi8 + pid_hkv * T * K + offs_k[:, None] + offs_t1[None, :] * K
            k_tile1 = tl.load(kb_ptrs1, mask=(TRUE_K[:, None] & t_mask1[None, :]), other=0.0).to(tl.float16)
            b_s1 = tl.dot(q_tile, k_tile1, out_dtype=tl.float32) * scale * RCP_LN2
            b_s1 = tl.where(t_mask1[None, :], b_s1, NEG_INF)
            m1 = tl.max(b_s1, axis=1)

            m2 = tl.maximum(m0, m1)
            th_rows = m2 - delta

            # 先写阈值，再把 ready 置 1（用 CAS 0->1，确保内存顺序）
            tl.store(th_ptrs, th_rows, mask=row_mask)
            tl.atomic_cas(ready_ptr, tl.full((), 0, tl.int32), tl.full((), 1, tl.int32))
        else:
            # 非 owner 且尚未 ready：直接“全部保留”（阈值置为 -inf）
            th_rows = tl.full([BM_DOT], NEG_INF, tl.float32)

    # 后续计算保持不变：使用 th_rows（若为 -inf 将不会剪枝，全部保留）
    s0 = pid_tb * BS
    NSB: tl.constexpr = (BS + SBS - 1) // SBS

    for sb in tl.static_range(NSB):
        offs_t_sb = s0 + sb * SBS + tl.arange(0, SBS)
        t_mask_sb = offs_t_sb < T

        kb_ptrs = k_hi8 + pid_hkv * T * K + (offs_t_sb[None, :] * K) + offs_k[:, None]
        k_tile = tl.load(kb_ptrs, mask=(TRUE_K[:, None] & t_mask_sb[None, :]), other=0.0).to(tl.float16)

        b_s     = tl.dot(q_tile, k_tile, out_dtype=tl.float32) * scale * RCP_LN2
        b_s_act = tl.where(t_mask_sb[None, :], b_s, NEG_INF)

        m_rows_blk = tl.max(b_s_act, axis=1)

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
    
    # 新增：阈值缓冲、就绪标志、原子锁
    threshold_buf  = torch.empty((HQ,),  device=q.device, dtype=torch.float32)
    threshold_ready = torch.zeros((HKV,), device=q.device, dtype=torch.int32)
    threshold_lock  = torch.zeros((HKV,), device=q.device, dtype=torch.int32)

    attn_forward_stage1_fused_threshold[(HKV, NTB)](
        q, k_hi8, v,
        m_buf, l_buf, o_buf,
        mask_buf,
        scale, T, NTB, NTBS, delta,
        threshold_buf, threshold_ready, threshold_lock,   # 传入
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
        MAX_SPIN=256,
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
    # exp_root_subdir = 'meta-llama_Llama-3.2-3B/longbench_gov_report_48_57'
    
    exp_root_dir = '/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result'
    
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_68_8k'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_68_32k'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_68_64k'
    # exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_68_128k'
    exp_root_subdir = 'Llama-3_2-3B/longbench_gov_report_48_68_256k'
    
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, 'layer_data')

    dtype = torch.float16
    BS = 256
    SBS = 256
    delta = 5.0
    iters = 1000
    warmup = 1000

    # 结果保存目录：当前文件夹/plot/当前文件名(无后缀)/
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    file_stem = os.path.splitext(os.path.basename(this_file))[0]
    plot_root_dir = os.path.join(this_dir, "plot", file_stem)
    os.makedirs(plot_root_dir, exist_ok=True)

    # 使用非交互式后端绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    def to_k_str(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"

    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        if layer_idx == 0:
            continue

        # 整层张量
        q_rope_full = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)   # [B=1, Hq, T, D]
        k_rope_full = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)   # [B=1, Hkv, T, D]
        v_full = layer_qkvh_data["v"].to('cuda', dtype=dtype)             # [B=1, Hkv, T, Dv]

        _, Hq, T_full, D = q_rope_full.shape
        _, Hkv, _, Dk = k_rope_full.shape
        _, _, _, Dv = v_full.shape

        # 从 1k 到当前 T 的长度（包含 T_full）
        step = 1024
        lengths = list(range(step, T_full + 1, step))
        if len(lengths) == 0 or lengths[-1] != T_full:
            lengths.append(T_full)

        fused_ms_list = []
        flash_ms_list = []
        x_lengths = []

        for L in tqdm(lengths, desc=f"Layer{layer_idx}"):
            # 每个长度 L：
            # 1) q 取第 L 个位置（最后一个 token）
            q_rope_1 = q_rope_full[:, :, L-1:L, :]         # [1, Hq, 1, D]
            # 2) k/v 取前 L 个
            k_rope = k_rope_full[:, :, :L, :]              # [1, Hkv, L, D]
            v = v_full[:, :, :L, :]                        # [1, Hkv, L, Dv]

            # 转换布局（每次循环新做）
            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
            scale = 1.0 / math.sqrt(D)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            # 基准函数
            def run_fused():
                return attn_forward_fused(
                    q_triton, k_hi8, k_lo8, v_triton,
                    scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False
                )

            def run_flash():
                return flash_attn_compute(q_rope_1, k_rope, v)

            ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
            ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

            fused_ms_list.append(ms_fused)
            flash_ms_list.append(ms_flash)
            x_lengths.append(L)

        # 绘图并保存
        plt.figure(figsize=(8, 5))
        plt.plot(x_lengths, fused_ms_list, label="Triton fused", marker="o")
        plt.plot(x_lengths, flash_ms_list, label="FlashAttn", marker="s")
        plt.xlabel("Sequence length (T)")
        plt.ylabel("Latency per run (ms)")
        
        plt.ylim(0, 0.4)

        # 标题中的 Tmax 用多少 k
        Tmax_k_str = to_k_str(T_full)
        plt.title(f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta})")

        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        layer_plot_dir = plot_root_dir
        os.makedirs(layer_plot_dir, exist_ok=True)

        # 文件名也改成多少 k
        plot_path = os.path.join(layer_plot_dir, f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        # 仅打印最后一个长度对应的参数和速度（T 也用 k 可选）
        last_ms_fused = fused_ms_list[-1]
        last_ms_flash = flash_ms_list[-1]
        print(f"Layer {layer_idx} | T={Tmax_k_str} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} | BS={BS} SBS={SBS} delta={delta} | Fused={last_ms_fused:.3f} ms Flash={last_ms_flash:.3f} ms")

        # 额外进行一次数值校验与跳过率，仅在最后长度上（不打印，只确保计算正确可选保存）
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
        scale = 1.0 / math.sqrt(D)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)
        _ = attn_forward_fused(q_triton, k_hi8, k_lo8, v_triton, scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False)
        _ = flash_attn_compute(q_rope_1, k_rope_full, v_full)

        break  # 如需只测试一个 layer，取消注释