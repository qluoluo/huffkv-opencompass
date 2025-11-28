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
                        # 例如：attn_kernel.attn_kernel_v1029_fused_nothres
                        default="attn_kernel.attn_kernel_v1109_fused_bsz",
                        help="Python module path for attn_forward")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--BS", type=int, default=256, help="Block size (BS)")
    parser.add_argument("--SBS", type=int, default=256, help="Sub block size (SBS)")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta parameter for skipping")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations before timing")
    parser.add_argument("--no-plot-line", action="store_true", help="Disable length sweep plotting")
    parser.add_argument("--step", type=int, default=1024, help="Step size for length sweep")
    parser.add_argument("--max-length", dest="max_length", type=int, default=None,
                        help="最大测试长度（若为 None，则使用数据的完整长度）")
    parser.add_argument("--no-thres-time", action="store_true",
                        help="排除阈值计算时间：外部预计算阈值并传入注意力内核")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
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

def find_existing_plot_path(plot_dir: str, layer_idx: int, bsz: int):
    """
    Best-effort scan for an existing plot file for this layer under plot_dir.
    We don't rely on exact naming from plot_speed_curve, so match common patterns.
    """
    if not os.path.isdir(plot_dir):
        return None
    layer_keys = [f"layer{layer_idx}", f"layer_{layer_idx}", f"l{layer_idx}_", f"_{layer_idx}_"]
    bsz_keys = [f"bsz{bsz}", f"batch{bsz}", f"_b{bsz}"]
    
    for fname in os.listdir(plot_dir):
        low = fname.lower()
        if not (low.endswith(".png") or low.endswith(".pdf") or low.endswith(".jpg") or low.endswith(".jpeg")):
            continue
        if any(k in low for k in layer_keys) and any(k in low for k in bsz_keys):
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

    # 可选：外部阈值计算函数（用于 --no-thres-time）
    compute_threshold_external = getattr(kernel_module, "compute_threshold_external", None)

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
    bsz = int(args.bsz)

    exp_root_dir  = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    exp_root_subdir = "Llama-3_2-3B/longbench_gov_report_48_68_256k"
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    lmax_name = str(args.max_length) if args.max_length is not None else ""
    plot_root_dir = os.path.join(this_dir, "plot", f"{attn_kernel_name}",
                                 f"BS{BS}_SBS{SBS}_delta{delta}_bsz{bsz}" + 
                                 (f"_{lmax_name}" if args.max_length is not None else "") + 
                                 (f"_nothres" if args.no_thres_time else "")
                                )
    os.makedirs(plot_root_dir, exist_ok=True)
    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)

    # 使用多个层的数据来构建 batch
    layer_indices = list(range(1, 1 + bsz))  # 从 layer_1 开始，取 bsz 个层

    # 如果画图已经存在，则快速跳过（不加载数据、不计算、不画图）
    existing_plot = find_existing_plot_path(plot_root_dir, layer_indices[0], bsz)
    if existing_plot is not None:
        print(f"[Info] Found existing plot for Layers {layer_indices} with bsz={bsz}: {existing_plot}")
        print("[Info] Skipping benchmark and plotting...")
        sys.exit(0)

    # 加载多个层的数据并拼接
    layer_qkvh_data_list = []
    layer_qkvh_data_iter = load_qkvh(layer_data_root, device='cpu', start_layer=layer_indices[0])
    
    for i in range(bsz):
        try:
            layer_data = next(layer_qkvh_data_iter)
            layer_qkvh_data_list.append(layer_data)
            print(f"[Info] Loaded data for layer_{layer_indices[i]}")
        except StopIteration:
            raise RuntimeError(f"Not enough layers to form batch size {bsz}. Only found {i} layers.")

    # 拼接不同层的数据作为 batch
    q_rope_full_list = []
    k_rope_full_list = []
    v_full_list = []
    
    for layer_data in layer_qkvh_data_list:
        q_rope_full_list.append(layer_data["q_rope"])
        k_rope_full_list.append(layer_data["k_rope"])
        v_full_list.append(layer_data["v"])
    
    # 拼接成 batch 维度 [bsz, H, T, D]
    q_rope_full = torch.cat(q_rope_full_list, dim=0).to("cuda", dtype=dtype)
    k_rope_full = torch.cat(k_rope_full_list, dim=0).to("cuda", dtype=dtype)
    v_full = torch.cat(v_full_list, dim=0).to("cuda", dtype=dtype)
    
    print(f"{q_rope_full.shape=}, {k_rope_full.shape=}, {v_full.shape=}")
    
    if args.max_length is not None and args.max_length > 0:
        q_rope_full = q_rope_full[..., :args.max_length, :]
        k_rope_full = k_rope_full[..., :args.max_length, :]
        v_full = v_full[..., :args.max_length, :]

    bsz_actual, Hq, T_full, D  = q_rope_full.shape
    _, Hkv, _, _      = k_rope_full.shape
    _, _, _, Dv       = v_full.shape
    scale = 1.0 / math.sqrt(D)

    print(f"[Info] Using batch size: {bsz_actual}, Hq: {Hq}, Hkv: {Hkv}, T_full: {T_full}, D: {D}, Dv: {Dv}")

    def bench_one_length(L):
        q_rope_1 = q_rope_full[:, :, L-1:L, :]   # 取每个 batch 的最后一个 query 步长 [bsz, Hq, 1, D]
        k_rope   = k_rope_full[:, :, :L, :]      # [bsz, Hkv, L, D]
        v        = v_full[:, :, :L, :]           # [bsz, Hkv, L, Dv]

        # 使用合并到内核模块的布局工具，得到 K/V 为 [bsz, T, Hkv, D]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        # 可选：外部预计算阈值（不计入 fused 计时）
        pre_th = None
        if args.no_thres_time:
            if compute_threshold_external is None:
                raise AttributeError(f"Module {args.kernel} does not define 'compute_threshold_external' required by --no-thres-time")
            # 注意：NTB 的计算逻辑与内核保持一致（使用 BS）
            NTB = (L + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_triton, k_fp16=k_triton_fp16, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        def run_fused():
            return attn_forward(
                q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False,
                precomputed_threshold=pre_th
            )

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        # 运行一次获取 skip ratio
        output, sr = attn_forward(
            q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True,
            precomputed_threshold=pre_th
        )

        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fused, ms_flash, float(sr)

    def validate_full():
        q_rope_1 = q_rope_full[:, :, T_full-1:T_full, :]  # [bsz, Hq, 1, D]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        pre_th = None
        if args.no_thres_time:
            NTB = (T_full + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_triton, k_fp16=k_triton_fp16, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        o_triton, skip_ratio = attn_forward(
            q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True,
            precomputed_threshold=pre_th
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
        raw_data_dir, f"layers_{layer_indices[0]}-{layer_indices[-1]}", T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=bsz
    )

    if os.path.exists(cache_path):
        x_lengths, fused_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
    else:
        fused_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc=f"Layers{layer_indices[0]}-{layer_indices[-1]}(bsz={bsz})"):
            ms_fused, ms_flash, sr = bench_one_length(L)
            fused_ms_list.append(ms_fused)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)
        x_lengths = lengths

        meta = dict(
            layer_indices=layer_indices, T_full=int(T_full),
            Hq=int(Hq), Hkv=int(Hkv), D=int(D), Dv=int(Dv),
            BS=int(BS), SBS=int(SBS), delta=float(delta),
            dtype=dtype_key(dtype), step=int(step),
            iters=int(iters), warmup=int(warmup),
            attn_kernel=attn_kernel_name,
            no_thres_time=bool(args.no_thres_time),
            bsz=int(bsz),
        )
        save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)

    plot_path = plot_speed_curve(
        x_lengths, fused_ms_list, flash_ms_list,
        T_full, BS, SBS, delta, f"layers_{layer_indices[0]}_bsz_{bsz}", plot_root_dir, attn_kernel_name
    )

    print(f"Layers {layer_indices} | bsz={bsz} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
          f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
          f"Fused={fused_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms")
    print(f"Saved plot to: {plot_path}")

    validate_full()