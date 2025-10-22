# run_attn_bench.py
import os
import sys
import math
from tqdm import tqdm
import torch
import argparse
import importlib

from utils.bench import benchmark
from utils.flash import flash_attn_compute
from utils.cache import (
    dtype_key, to_k_str, make_cache_file_path, save_raw_cache, load_raw_cache
)
from utils.plot import plot_speed_curve
from utils.load import load_qkvh

def parse_args():
    parser = argparse.ArgumentParser(description="Run attention benchmark with configurable hyperparameters.")
    parser.add_argument("--kernel", type=str,
                        # 例如：attn_kernel.attn_kernel_v1022_fused_grid1d
                        default="attn_kernel.attn_kernel_v1022_fused_grid1d",
                        help="Python module path for attn_forward")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--BS", type=int, default=256, help="Block size (BS)")
    parser.add_argument("--SBS", type=int, default=256, help="Sub block size (SBS)")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta parameter for skipping")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations before timing")
    parser.add_argument("--no-plot-line", action="store_true", help="Disable length sweep plotting")
    parser.add_argument("--step", type=int, default=1024, help="Step size for length sweep")
    return parser.parse_args()

def map_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    elif dtype_str == "bf16":
        return torch.bfloat16
    elif dtype_str == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype string: {dtype_str}")

def find_existing_plot_path(plot_dir: str, layer_idx: int):
    """
    Best-effort scan for an existing plot file for this layer under plot_dir.
    We don't rely on exact naming from plot_speed_curve, so match common patterns.
    """
    if not os.path.isdir(plot_dir):
        return None
    layer_keys = [f"layer{layer_idx}", f"layer_{layer_idx}", f"l{layer_idx}_", f"_{layer_idx}_"]
    for fname in os.listdir(plot_dir):
        low = fname.lower()
        if not (low.endswith(".png") or low.endswith(".pdf") or low.endswith(".jpg") or low.endswith(".jpeg")):
            continue
        if any(k in low for k in layer_keys):
            return os.path.join(plot_dir, fname)
    return None

if __name__ == "__main__":
    args = parse_args()

    # 动态加载内核模块
    kernel_module = importlib.import_module(args.kernel)
    if not hasattr(kernel_module, "attn_forward"):
        raise AttributeError(f"Module {args.kernel} does not define 'attn_forward'")
    attn_forward = getattr(kernel_module, "attn_forward")
    
    # 同时从内核模块使用合并后的 layout 工具
    if not hasattr(kernel_module, "convert_to_triton_layout") or not hasattr(kernel_module, "pack_k_hi_lo"):
        raise AttributeError(f"Module {args.kernel} must define 'convert_to_triton_layout' and 'pack_k_hi_lo'")
    
    convert_to_triton_layout = getattr(kernel_module, "convert_to_triton_layout")
    pack_k_hi_lo = getattr(kernel_module, "pack_k_hi_lo")

    attn_kernel_name = kernel_module.__name__.split('.')[-1]
    torch.set_float32_matmul_precision("high")

    dtype  = map_dtype(args.dtype)
    BS     = int(args.BS)
    SBS    = int(args.SBS)
    delta  = float(args.delta)
    iters  = int(args.iters)
    warmup = int(args.warmup)
    PLOT_LINE = not args.no_plot_line
    step = int(args.step)

    exp_root_dir  = "/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    exp_root_subdir = "Llama-3_2-3B/longbench_gov_report_48_68_256k"
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    plot_root_dir = os.path.join(this_dir, "plot", f"{attn_kernel_name}", f"BS{BS}_SBS{SBS}_delta{delta}")
    os.makedirs(plot_root_dir, exist_ok=True)
    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    # 只使用 layer_1 的数据（写死）
    layer_idx = 1

    # 如果画图已经存在，则快速跳过（不加载数据、不计算、不画图）
    existing_plot = find_existing_plot_path(plot_root_dir, layer_idx)
    if existing_plot is not None:
        print(f"[Info] Found existing plot for Layer {layer_idx}: {existing_plot}")
        print("[Info] Skipping benchmark and plotting...")
        sys.exit(0)

    # 加载 layer_1 的数据
    layer_qkvh_data_iter = load_qkvh(layer_data_root, device='cpu', start_layer=layer_idx)
    try:
        layer_qkvh_data = next(layer_qkvh_data_iter)
    except StopIteration:
        raise RuntimeError(f"No data found for layer_{layer_idx} in {layer_data_root}")

    q_rope_full = layer_qkvh_data["q_rope"].to("cuda", dtype=dtype)   # [B=1, Hq, T, D]
    k_rope_full = layer_qkvh_data["k_rope"].to("cuda", dtype=dtype)   # [B=1, Hkv, T, D]
    v_full      = layer_qkvh_data["v"].to("cuda", dtype=dtype)        # [B=1, Hkv, T, Dv]

    _, Hq, T_full, D  = q_rope_full.shape
    _, Hkv, _, _      = k_rope_full.shape
    _, _, _, Dv       = v_full.shape
    scale = 1.0 / math.sqrt(D)

    def bench_one_length(L):
        q_rope_1 = q_rope_full[:, :, L-1:L, :]   # 取最后一个 query 步长
        k_rope   = k_rope_full[:, :, :L, :]
        v        = v_full[:, :, :L, :]

        # 使用合并到内核模块的布局工具，得到 K/V 为 [T, Hkv, D]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        def run_fused():
            return attn_forward(
                q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False
            )

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        _, sr = attn_forward(
            q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True
        )

        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fused, ms_flash, float(sr)

    def validate_full():
        q_rope_1 = q_rope_full[:, :, T_full-1:T_full, :]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        o_triton, skip_ratio = attn_forward(
            q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True
        )
        o_flash = flash_attn_compute(q_rope_1, k_rope_full, v_full)

        max_abs  = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

    # 长度 sweep
    step = int(args.step)
    lengths = list(range(step, T_full, step)) + [T_full]

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
            attn_kernel=attn_kernel_name,
        )
        save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)

    plot_path = plot_speed_curve(
        x_lengths, fused_ms_list, flash_ms_list,
        T_full, BS, SBS, delta, layer_idx, plot_root_dir, attn_kernel_name
    )

    print(f"Layer {layer_idx} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
          f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
          f"Fused={fused_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms")
    print(f"Saved plot to: {plot_path}")

    validate_full()
