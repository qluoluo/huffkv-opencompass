# run_attn_bench.py
import argparse
import importlib
import math
import os
import re

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import (
    dtype_key,
    load_raw_cache,
    make_cache_file_path,
    save_raw_cache,
    to_k_str,
)
from utils.flash import flash_attn_compute
from utils.load import load_qkvh
from utils.plot import plot_speed_curve

EXP_ROOT_DIR = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
EXP_ROOT_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention benchmark with configurable hyperparameters.")
    parser.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1109_fused_bsz",
        help="Python module path for attn_forward (e.g., attn_kernel.attn_kernel_v1029_fused_nothres)",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--BS", type=int, default=256, help="Block size (BS)")
    parser.add_argument("--SBS", type=int, default=None, help="Sub block size (SBS). Defaults to BS when omitted.")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta parameter for skipping")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations before timing")
    parser.add_argument("--no-plot-line", action="store_true", help="Disable length sweep plotting")
    parser.add_argument("--step", type=int, default=1024, help="Step size for length sweep")
    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default=None,
        help="最大测试长度（若为 None，则使用数据的完整长度）",
    )
    parser.add_argument(
        "--no-thres-time",
        action="store_true",
        help="排除阈值计算时间：外部预计算阈值并传入注意力内核",
    )
    parser.add_argument("--bsz", type=int, default=1, help="Batch size (number of layers to combine)")
    return parser.parse_args()


def map_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def load_kernel_components(kernel_path: str):
    kernel_module = importlib.import_module(kernel_path)
    if not hasattr(kernel_module, "attn_forward"):
        raise AttributeError(f"Module {kernel_path} does not define 'attn_forward'")

    if not hasattr(kernel_module, "convert_to_triton_layout") or not hasattr(kernel_module, "pack_k_hi_lo"):
        raise AttributeError(f"Module {kernel_path} must define 'convert_to_triton_layout' and 'pack_k_hi_lo'")

    attn_forward = getattr(kernel_module, "attn_forward")
    convert_to_triton_layout = getattr(kernel_module, "convert_to_triton_layout")
    pack_k_hi_lo = getattr(kernel_module, "pack_k_hi_lo")
    compute_threshold_external = getattr(kernel_module, "compute_threshold_external", None)
    return kernel_module, attn_forward, convert_to_triton_layout, pack_k_hi_lo, compute_threshold_external


def get_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    name = props.name.strip()
    total_mem_gb = math.ceil(props.total_memory / (1024**3))
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name) or "gpu"
    tag = f"{safe_name}_{total_mem_gb}GB"
    return tag, name, total_mem_gb, device_idx


def build_plot_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, bsz, max_length, no_thres_time, base_dir):
    lmax_name = str(max_length) if max_length is not None else ""
    plot_root_dir = os.path.join(
        base_dir,
        "plot",
        f"{attn_kernel_name}",
        gpu_tag,
        f"BS{BS}_SBS{SBS}_delta{delta}_bsz{bsz}"
        + (f"_{lmax_name}" if max_length is not None else "")
        + (f"_nothres" if no_thres_time else ""),
    )
    os.makedirs(plot_root_dir, exist_ok=True)

    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    return plot_root_dir, raw_data_dir


def load_layer_batch(layer_data_root, layer_indices, dtype, max_length):
    layer_qkvh_data_list = []
    layer_qkvh_data_iter = load_qkvh(layer_data_root, device="cpu", start_layer=layer_indices[0])

    for i, layer_idx in enumerate(layer_indices):
        try:
            layer_data = next(layer_qkvh_data_iter)
        except StopIteration:
            raise RuntimeError(f"Not enough layers to form batch size {len(layer_indices)}. Only found {i} layers.")
        layer_qkvh_data_list.append(layer_data)
        print(f"[Info] Loaded data for layer_{layer_idx}")

    q_rope_full_list, k_rope_full_list, v_full_list = [], [], []
    for layer_data in layer_qkvh_data_list:
        q_rope_full_list.append(layer_data["q_rope"])
        k_rope_full_list.append(layer_data["k_rope"])
        v_full_list.append(layer_data["v"])

    q_rope_full = torch.cat(q_rope_full_list, dim=0).to("cuda", dtype=dtype)
    k_rope_full = torch.cat(k_rope_full_list, dim=0).to("cuda", dtype=dtype)
    v_full = torch.cat(v_full_list, dim=0).to("cuda", dtype=dtype)
    print(f"{q_rope_full.shape=}, {k_rope_full.shape=}, {v_full.shape=}")

    if max_length is not None and max_length > 0:
        q_rope_full = q_rope_full[..., :max_length, :]
        k_rope_full = k_rope_full[..., :max_length, :]
        v_full = v_full[..., :max_length, :]

    return q_rope_full, k_rope_full, v_full


def main():
    args = parse_args()
    (
        kernel_module,
        attn_forward,
        convert_to_triton_layout,
        pack_k_hi_lo,
        compute_threshold_external,
    ) = load_kernel_components(args.kernel)

    attn_kernel_name = kernel_module.__name__.split(".")[-1]
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    iters = int(args.iters)
    warmup = int(args.warmup)
    plot_line = not args.no_plot_line  # CLI flag retained for compatibility; currently unused
    step = int(args.step)
    bsz = int(args.bsz)

    exp_root = os.path.join(EXP_ROOT_DIR, EXP_ROOT_SUBDIR)
    layer_data_root = os.path.join(exp_root, "layer_data")

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    plot_root_dir, raw_data_dir = build_plot_dirs(
        attn_kernel_name, gpu_tag, BS, SBS, delta, bsz, args.max_length, args.no_thres_time, this_dir
    )

    layer_indices = list(range(1, 1 + bsz))

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype, args.max_length)

    bsz_actual, Hq, T_full, D = q_rope_full.shape
    _, Hkv, _, _ = k_rope_full.shape
    _, _, _, Dv = v_full.shape
    scale = 1.0 / math.sqrt(D)

    print(f"[Info] Using batch size: {bsz_actual}, Hq: {Hq}, Hkv: {Hkv}, T_full: {T_full}, D: {D}, Dv: {Dv}")

    def bench_one_length(L):
        q_rope_1 = q_rope_full[:, :, L - 1 : L, :]
        k_rope = k_rope_full[:, :, :L, :]
        v = v_full[:, :, :L, :]

        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        pre_th = None
        if args.no_thres_time:
            if compute_threshold_external is None:
                raise AttributeError(
                    f"Module {args.kernel} does not define 'compute_threshold_external' required by --no-thres-time"
                )
            NTB = (L + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_triton, k_fp16=k_triton_fp16, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        def run_fused():
            return attn_forward(
                q=q_triton,
                k_hi8=k_hi8,
                k_lo8=k_lo8,
                k_fp16=k_triton_fp16,
                v=v_triton,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                return_skip_ratio=False,
                precomputed_threshold=pre_th,
            )

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        output, sr = attn_forward(
            q=q_triton,
            k_hi8=k_hi8,
            k_lo8=k_lo8,
            k_fp16=k_triton_fp16,
            v=v_triton,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
            precomputed_threshold=pre_th,
        )

        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fused, ms_flash, float(sr)

    def validate_full():
        q_rope_1 = q_rope_full[:, :, T_full - 1 : T_full, :]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope_full, v_full)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        pre_th = None
        if args.no_thres_time:
            NTB = (T_full + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_triton, k_fp16=k_triton_fp16, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        o_triton, skip_ratio = attn_forward(
            q=q_triton,
            k_hi8=k_hi8,
            k_lo8=k_lo8,
            k_fp16=k_triton_fp16,
            v=v_triton,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
            precomputed_threshold=pre_th,
        )
        o_flash = flash_attn_compute(q_rope_1, k_rope_full, v_full)

        max_abs = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        print(f"Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
        print(f"Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

    lengths = list(range(step, T_full, step)) + [T_full]
    cache_path = make_cache_file_path(
        raw_data_dir,
        f"layers_{layer_indices[0]}-{layer_indices[-1]}",
        T_full,
        Hq,
        Hkv,
        D,
        Dv,
        BS,
        SBS,
        delta,
        dtype,
        step,
        iters,
        warmup,
        bsz=bsz,
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
            layer_indices=layer_indices,
            T_full=int(T_full),
            Hq=int(Hq),
            Hkv=int(Hkv),
            D=int(D),
            Dv=int(Dv),
            BS=int(BS),
            SBS=int(SBS),
            delta=float(delta),
            dtype=dtype_key(dtype),
            step=int(step),
            iters=int(iters),
            warmup=int(warmup),
            attn_kernel=attn_kernel_name,
            no_thres_time=bool(args.no_thres_time),
            bsz=int(bsz),
        )
        save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)

    plot_path = plot_speed_curve(
        x_lengths,
        fused_ms_list,
        flash_ms_list,
        T_full,
        BS,
        SBS,
        delta,
        f"layers_{layer_indices[0]}_bsz_{bsz}",
        plot_root_dir,
        attn_kernel_name,
    )

    print(
        f"Layers {layer_indices} | bsz={bsz} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
        f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
        f"Fused={fused_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms"
    )
    print(f"Saved plot to: {plot_path}")

    validate_full()


if __name__ == "__main__":
    main()
