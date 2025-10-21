# run_attn_bench.py
import os
import math
from tqdm import tqdm
import torch


from utils.layout import convert_to_triton_layout, pack_k_hi_lo
from utils.bench import benchmark
from utils.flash import flash_attn_compute
from utils.cache import (
    dtype_key, to_k_str, make_cache_file_path, save_raw_cache, load_raw_cache
)
from utils.plot import plot_speed_curve
from utils.load import load_qkvh

if __name__ == "__main__":
    from attn_kernel.attn_kernel_v1019_unfused import attn_forward
    # from attn_kernel.attn_kernel_v1019_fused import attn_forward
    # from attn_kernel.attn_kernel_v1020_onekernel import attn_forward

    # 获取attn kernel名称用于文件名
    attn_kernel_name = attn_forward.__module__.split('.')[-1]  # 获取'attn_kernel_v1020_onekernel'
    torch.set_float32_matmul_precision("high")

    # ========= 超参数（保留在主文件中）=========
    dtype  = torch.float16
    BS     = 256
    SBS    = 256
    delta  = 5.0
    iters  = 1000
    warmup = 1000
    PLOT_LINE = True

    # ========= 路径（保留在主文件中）=========
    exp_root_dir  = "/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    
    # exp_root_subdir = "Llama-3_2-3B/longbench_gov_report_48_68_32k"
    exp_root_subdir = "Llama-3_2-3B/longbench_gov_report_48_68_256k"
    
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    # 图与缓存目录 - 现在包含attn kernel名称
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    file_stem = os.path.splitext(os.path.basename(this_file))[0]
    # 在目录名中包含attn kernel名称
    plot_root_dir = os.path.join(this_dir, "plot", f"{file_stem}_{attn_kernel_name}")
    plot_root_dir = os.path.join(this_dir, "plot", f"{attn_kernel_name}")
    os.makedirs(plot_root_dir, exist_ok=True)
    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
        if layer_idx == 0:
            continue

        # 通用准备
        q_rope_full = layer_qkvh_data["q_rope"].to("cuda", dtype=dtype)   # [B=1, Hq, T, D]
        k_rope_full = layer_qkvh_data["k_rope"].to("cuda", dtype=dtype)   # [B=1, Hkv, T, D]
        v_full      = layer_qkvh_data["v"].to("cuda", dtype=dtype)        # [B=1, Hkv, T, Dv]

        _, Hq, T_full, D  = q_rope_full.shape
        _, Hkv, _, _      = k_rope_full.shape
        _, _, _, Dv       = v_full.shape
        scale = 1.0 / math.sqrt(D)

        # 统一的单长度基准函数
        def bench_one_length(L):
            q_rope_1 = q_rope_full[:, :, L-1:L, :]
            k_rope   = k_rope_full[:, :, :L, :]
            v        = v_full[:, :, :L, :]

            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            def run_fused():
                # return attn_forward(q_triton, k_hi8, k_lo8, v_triton,
                #                         scale=scale, BS=BS, SBS=SBS, delta=delta,
                #                         return_skip_ratio=False)
                return attn_forward(q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                                        scale=scale, BS=BS, SBS=SBS, delta=delta,
                                        return_skip_ratio=False)

            def run_flash():
                return flash_attn_compute(q_rope_1, k_rope, v)

            # 仅统计 skip ratio（不计时）
            # _, sr = attn_forward(q_triton, k_hi8, k_lo8, v_triton,
            #                         scale=scale, BS=BS, SBS=SBS, delta=delta,
            #                         return_skip_ratio=True)
            
            _, sr = attn_forward(q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                                        scale=scale, BS=BS, SBS=SBS, delta=delta,
                                        return_skip_ratio=True)

            ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
            ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
            return ms_fused, ms_flash, float(sr)

        # 统一的数值与跳过率校验（使用全长）
        def validate_full():
            q_rope_1 = q_rope_full[:, :, T_full-1:T_full, :]
            q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
            k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

            o_triton, skip_ratio = attn_forward(q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                                        scale=scale, BS=BS, SBS=SBS, delta=delta,
                                        return_skip_ratio=True)
            o_flash = flash_attn_compute(q_rope_1, k_rope_full, v_full)

            max_abs  = (o_triton.float() - o_flash.float()).abs().max().item()
            mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
            print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
            print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

        if PLOT_LINE:
            # 构造长度序列（保证包含 T_full）
            step = 1024
            lengths = list(range(step, T_full, step)) + [T_full]

            # 在缓存元数据中包含attn kernel名称
            cache_path = make_cache_file_path(
                raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup
            )

            if os.path.exists(cache_path):
                x_lengths, fused_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
            else:
                fused_ms_list, flash_ms_list, skip_ratios = [], [], []
                for L in tqdm(lengths, desc=f"Layer{layer_idx}"):
                    ms_fused, ms_flash, sr = bench_one_length(L)
                    fused_ms_list.append(ms_fused)
                    flash_ms_list.append(ms_flash)
                    skip_ratios.append(sr)
                x_lengths = lengths

                meta = dict(
                    layer_idx=layer_idx, T_full=int(T_full),
                    Hq=int(Hq), Hkv=int(Hkv), D=int(D), Dv=int(Dv),
                    BS=int(BS), SBS=int(SBS), delta=float(delta),
                    dtype=dtype_key(dtype), step=int(step),
                    iters=int(iters), warmup=int(warmup),
                    attn_kernel=attn_kernel_name,  # 添加attn kernel名称到元数据
                )
                save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)

            # 在绘图函数中传递attn kernel名称
            plot_path = plot_speed_curve(
                x_lengths, fused_ms_list, flash_ms_list,
                T_full, BS, SBS, delta, layer_idx, plot_root_dir, attn_kernel_name
            )

            print(f"Layer {layer_idx} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
                f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
                f"Fused={fused_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms")
            print(f"Saved plot to: {plot_path}")

            validate_full()
        else:
            # 只跑最大长度
            ms_fused, ms_flash, _ = bench_one_length(T_full)
            print(f"Layer {layer_idx} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
                f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
                f"Fused={ms_fused:.3f} ms Flash={ms_flash:.3f} ms")

            validate_full()

        break  # 只处理 layer1