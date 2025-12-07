import os
# os.environ["CUDA_VISIABLE_DEVICES"] = "4"

# os.environ["TRITON_DUMP_ASSEMBLY"] = "1"
# os.environ["TRITON_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), "triton_cache_fp8")

import math
from tqdm import tqdm

import torch
import triton
import triton.language as tl
import json

@triton.jit
def attn_forward_all_in_one(
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


def attn_forward_fused(
    q: torch.Tensor,      # [HQ, K]
    k_hi8: torch.Tensor,  # [HKV, T, K], float8_e5m2
    k_lo8: torch.Tensor,  # [HKV, T, K], uint8 (保留以保持签名一致，此核未使用)
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
    assert K == Kk and HKV == HKVv and Tv == T

    G = HQ // HKV

    if scale is None:
        scale = 1.0 / math.sqrt(K)
    if SBS is None:
        SBS = BS

    NTB = triton.cdiv(T, BS)
    NSB = triton.cdiv(BS, SBS)
    NTBS = NTB * NSB

    # 直接输出到 o，不再需要中间缓冲
    o = torch.empty((HQ, V), device=q.device, dtype=q.dtype)

    # 统计每个 HKV 的保留子块数（相当于原先 mask_buf 的 1 的计数）
    kept_buf = torch.zeros((HKV,), device=q.device, dtype=torch.int32)

    # 单 kernel 一次 launch 完成全部流程
    attn_forward_all_in_one[(HKV,)](
        q, k_hi8, v, o,
        scale, T, NTB, NTBS, delta,
        kept_buf,
        HKV=HKV, HQ=HQ, K=K, V=V, G=G, BS=BS, SBS=SBS,
    )

    if return_skip_ratio:
        kept = kept_buf.to(torch.int64).sum()  # 避免溢出
        total = HKV * NTBS
        skip_ratio = float((1.0 - (kept.float() / float(total))).item())
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

    # 是否绘制整条曲线；若为 False，仅对 layer1 的最大长度做一次新计算，不读写缓存，不作绘图
    PLOT_LINE = True

    # 结果保存目录：当前文件夹/plot/当前文件名(无后缀)/
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    file_stem = os.path.splitext(os.path.basename(this_file))[0]
    plot_root_dir = os.path.join(this_dir, "plot", file_stem)
    os.makedirs(plot_root_dir, exist_ok=True)
    
    # 原始数据缓存目录
    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    def _dtype_key(dt: torch.dtype) -> str:
        return {
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
            torch.float32: "fp32",
        }.get(dt, str(dt))

    def _cache_file_path(layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup):
        def _to_k(n: int) -> str:
            val = n / 1024.0
            return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
        fname = (
            f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
            f"_BS{BS}_SBS{SBS}_delta{delta:g}_{_dtype_key(dtype)}"
            f"_step{step}_it{iters}_wu{warmup}.json"
        )
        return os.path.join(raw_data_dir, fname)

    def save_raw_cache(path, meta: dict, lengths, fused_ms, flash_ms, skip_ratios):
        payload = {
            "meta": meta,
            "lengths": [int(x) for x in lengths],
            "fused_ms": [float(x) for x in fused_ms],
            "flash_ms": [float(x) for x in flash_ms],
            "skip_ratios": [None if x is None else float(x) for x in skip_ratios],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load_raw_cache(path):
        with open(path, "r") as f:
            data = json.load(f)
        return (
            data["lengths"],
            data["fused_ms"],
            data["flash_ms"],
            data.get("skip_ratios", [None] * len(data["lengths"])),
            data.get("meta", {}),
        )

    # 使用非交互式后端绘图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    def to_k_str(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    
    if PLOT_LINE:
        # 只测试 layer1（索引为 1）
        for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
            if layer_idx != 1:
                continue

            q_rope_full = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)   # [B=1, Hq, T, D]
            k_rope_full = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)   # [B=1, Hkv, T, D]
            v_full      = layer_qkvh_data["v"].to('cuda', dtype=dtype)        # [B=1, Hkv, T, Dv]

            _, Hq, T_full, D = q_rope_full.shape
            _, Hkv, _, Dk = k_rope_full.shape
            _, _, _, Dv = v_full.shape

            # 从 1k 到当前 T 的长度（包含 T_full）
            step = 1024
            lengths = list(range(step, T_full + 1, step))
            if len(lengths) == 0 or lengths[-1] != T_full:
                lengths.append(T_full)

            # 尝试复用缓存
            cache_path = _cache_file_path(
                layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup
            )

            fused_ms_list, flash_ms_list, x_lengths, skip_ratios = [], [], [], []
            cache_hit = os.path.exists(cache_path)
            if cache_hit:
                x_lengths, fused_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
            else:
                # 未命中缓存，正常跑测
                for L in tqdm(lengths, desc=f"Layer{layer_idx}"):
                    q_rope_1 = q_rope_full[:, :, L-1:L, :]
                    k_rope = k_rope_full[:, :, :L, :]
                    v      = v_full[:, :, :L, :]

                    q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
                    scale = 1.0 / math.sqrt(D)
                    k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

                    # 仅用于记录 skip ratio（不计入计时）
                    _o, sr = attn_forward_fused(
                        q_triton, k_hi8, k_lo8, v_triton,
                        scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True
                    )
                    skip_ratios.append(float(sr))

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

                # 保存原始数据到本地
                meta = dict(
                    layer_idx=layer_idx, T_full=int(T_full),
                    Hq=int(Hq), Hkv=int(Hkv), D=int(D), Dv=int(Dv),
                    BS=int(BS), SBS=int(SBS), delta=float(delta),
                    dtype=_dtype_key(dtype), step=int(step),
                    iters=int(iters), warmup=int(warmup),
                )
                save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)

            # 绘图（仅 layer1）
            plt.figure(figsize=(8, 5))
            plt.plot(x_lengths, fused_ms_list, label="Triton fused", marker="o")
            plt.plot(x_lengths, flash_ms_list, label="FlashAttn", marker="s")
            plt.xlabel("Sequence length (T)")
            plt.ylabel("Latency per run (ms)")
            plt.ylim(0, 0.4)

            Tmax_k_str = to_k_str(T_full)
            plt.title(f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta})")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend()
            layer_plot_dir = plot_root_dir
            os.makedirs(layer_plot_dir, exist_ok=True)
            plot_path = os.path.join(layer_plot_dir, f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}.png")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()

            # 打印最后一个长度对应的参数和速度
            last_ms_fused = fused_ms_list[-1]
            last_ms_flash = flash_ms_list[-1]
            print(f"Layer {layer_idx} | T={Tmax_k_str} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} | BS={BS} SBS={SBS} delta={delta} | Fused={last_ms_fused:.3f} ms Flash={last_ms_flash:.3f} ms")

            # 数值与跳过率校验（最后长度）
            q_rope_1 = q_rope_full[:, :, T_full-1:T_full, :]
            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
            scale = 1.0 / math.sqrt(D)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            o_triton, skip_ratio = attn_forward_fused(
                q_triton, k_hi8, k_lo8, v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta,
                return_skip_ratio=True
            )
            o_flash = flash_attn_compute(q_rope_1, k_rope_full, v_full)

            max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
            mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
            print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
            print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

            break  # 只处理 layer1
    else:
        # 不绘图、不读写缓存：仅对 layer1 的最大长度做一次新计算
        for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
            if layer_idx != 1:
                continue

            q_rope_full = layer_qkvh_data["q_rope"].to('cuda', dtype=dtype)   # [B=1, Hq, T, D]
            k_rope_full = layer_qkvh_data["k_rope"].to('cuda', dtype=dtype)   # [B=1, Hkv, T, D]
            v_full      = layer_qkvh_data["v"].to('cuda', dtype=dtype)        # [B=1, Hkv, T, Dv]

            _, Hq, T_full, D = q_rope_full.shape
            _, Hkv, _, Dk = k_rope_full.shape
            _, _, _, Dv = v_full.shape

            L = T_full
            q_rope_1 = q_rope_full[:, :, L-1:L, :]
            k_rope = k_rope_full[:, :, :L, :]
            v      = v_full[:, :, :L, :]

            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
            scale = 1.0 / math.sqrt(D)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            def run_fused():
                return attn_forward_fused(
                    q_triton, k_hi8, k_lo8, v_triton,
                    scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False
                )

            def run_flash():
                return flash_attn_compute(q_rope_1, k_rope, v)

            ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
            ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)

            Tmax_k_str = to_k_str(T_full)
            print(f"Layer {layer_idx} | T={Tmax_k_str} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} | BS={BS} SBS={SBS} delta={delta} | Fused={ms_fused:.3f} ms Flash={ms_flash:.3f} ms")

            # 数值与跳过率校验（最大长度）
            q_rope_1 = q_rope_full[:, :, T_full-1:T_full, :]
            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
            scale = 1.0 / math.sqrt(D)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            o_triton, skip_ratio = attn_forward_fused(
                q_triton, k_hi8, k_lo8, v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta,
                return_skip_ratio=True
            )
            o_flash = flash_attn_compute(q_rope_1, k_rope_full, v_full)

            max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
            mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
            print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
            print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

            break  # 只处理 layer1