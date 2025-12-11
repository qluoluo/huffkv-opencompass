# Benchmarking & plotting for 2-bit quantized attention kernel, mirroring run_attn_bench.py.
import argparse
import importlib
import math
import os
import re

import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import dtype_key, load_raw_cache, make_cache_file_path, save_raw_cache, to_k_str
from utils.flash import flash_attn_compute
from utils.load import load_qkvh
from utils.plot import plot_speed_curve

# Ensure attn_kernel package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in os.sys.path:
    os.sys.path.append(THIS_DIR)

# Default kernel for q2
from attn_kernel.attn_kernel_v1210_fused_bsz_q2 import attn_forward_decode_quantized

EXP_ROOT_DIR = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
EXP_ROOT_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"


def parse_args():
    p = argparse.ArgumentParser(description="Run quantized attn kernel test with recorded layer data.")
    p.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1210_fused_bsz_q2",
        help="Python module path for attn_forward_decode_quantized",
    )
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128)
    p.add_argument("--SBS", type=int, default=None)
    p.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help="Delta value for skipping; run the script once per delta to compare (e.g., 3, 5, 8, 10).",
    )
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="If set and >0, truncate to this length; if <0, use the full recorded length.",
    )
    p.add_argument("--step", type=int, default=1024, help="Step size for length sweep when plotting.")
    p.add_argument("--iters", type=int, default=200, help="Benchmark iters")
    p.add_argument("--warmup", type=int, default=50, help="Benchmark warmup")
    return p.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


def convert_layout(q_rope_1: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor):
    B, Hq, qlen, Dq = q_rope_1.shape
    Bk, Hkv, T, Dk = k_rope.shape
    Bv, Hvv, Tv, Dv = v.shape
    assert B == Bk == Bv and qlen == 1 and Tv == T and Hvv == Hkv
    q = q_rope_1[:, :, 0, :].contiguous()
    k = k_rope.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    return q, k, v


def quantize_k_2bit_no_token_dim(k: torch.Tensor):
    # Scale/zero are per (B, HKV, K); token dimension is removed and broadcasted later.
    k_min = k.amin(dim=1)
    k_max = k.amax(dim=1)
    scale = ((k_max - k_min).clamp_min(1e-6) / 3.0).contiguous()
    zero = k_min.contiguous()
    k_q = torch.round((k - zero[:, None, :, :]) / scale[:, None, :, :]).clamp(0, 3).to(torch.uint8)

    # Pack 4x2-bit values into a single byte to avoid storing each 2-bit value as uint8
    B, T, HKV, K = k_q.shape
    values_per_byte = 4  # 8 bits / 2 bits
    k_packed_len = (K + values_per_byte - 1) // values_per_byte
    pad = k_packed_len * values_per_byte - K
    if pad:
        pad_tensor = torch.zeros((B, T, HKV, pad), device=k_q.device, dtype=k_q.dtype)
        k_q = torch.cat([k_q, pad_tensor], dim=-1)
    k_q = k_q.view(B, T, HKV, k_packed_len, values_per_byte)
    k_q_packed = (
        k_q[..., 0]
        | (k_q[..., 1] << 2)
        | (k_q[..., 2] << 4)
        | (k_q[..., 3] << 6)
    ).contiguous()
    return k_q_packed, scale, zero


def load_kernel_components(kernel_path: str):
    kernel_module = importlib.import_module(kernel_path)
    if not hasattr(kernel_module, "attn_forward_decode_quantized"):
        raise AttributeError(f"Module {kernel_path} does not define 'attn_forward_decode_quantized'")
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode_quantized")
    return kernel_module, attn_forward_decode


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


def build_plot_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, layer_idx, max_length, base_dir):
    lmax_name = str(max_length) if max_length is not None else ""
    plot_root_dir = os.path.join(
        base_dir,
        "plot",
        f"{attn_kernel_name}",
        gpu_tag,
        f"layer{layer_idx}_BS{BS}_SBS{SBS}_delta{delta}"
        + (f"_{lmax_name}" if max_length is not None else ""),
    )
    os.makedirs(plot_root_dir, exist_ok=True)

    raw_data_dir = os.path.join(plot_root_dir, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    return plot_root_dir, raw_data_dir


def load_layer_batch(layer_data_root, layer_idx, dtype, max_length):
    data_iter = load_qkvh(layer_data_root, device="cpu", start_layer=layer_idx)
    try:
        layer_data = next(data_iter)
    except StopIteration:
        raise RuntimeError(f"No layer data found for layer_{layer_idx}")

    q_rope = layer_data["q_rope"]
    k_rope = layer_data["k_rope"]
    v = layer_data["v"]

    q_rope = q_rope.to("cuda", dtype=dtype)
    k_rope = k_rope.to("cuda", dtype=dtype)
    v = v.to("cuda", dtype=dtype)

    if max_length is not None and max_length > 0:
        q_rope = q_rope[..., :max_length, :]
        k_rope = k_rope[..., :max_length, :]
        v = v[..., :max_length, :]

    return q_rope, k_rope, v


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = int(args.SBS) if args.SBS is not None else BS
    delta = float(args.delta)
    step = int(args.step)
    iters = int(args.iters)
    warmup = int(args.warmup)
    max_length = None if args.max_length is not None and args.max_length < 0 else args.max_length

    kernel_module, attn_forward_decode = load_kernel_components(args.kernel)
    attn_kernel_name = kernel_module.__name__.split(".")[-1]

    exp_root = os.path.join(EXP_ROOT_DIR, EXP_ROOT_SUBDIR)
    layer_data_root = os.path.join(exp_root, "layer_data")

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, args.layer, dtype, max_length)

    B, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    print(f"[Info] Layer={args.layer}, B={B}, Hq={Hq}, Hkv={Hkv}, T_full={T_full}, K={K}, V={V}")

    lengths = list(range(step, T_full, step)) + [T_full]

    def bench_one_length(L, delta):
        q_rope_1 = q_rope_full[:, :, L - 1 : L, :].contiguous()
        k_rope = k_rope_full[:, :, :L, :].contiguous()
        v = v_full[:, :, :L, :].contiguous()

        q, k, v = convert_layout(q_rope_1, k_rope, v)
        q_1 = q.unsqueeze(1)  # [B, 1, Hq, K]
        k_q, k_scale, k_zero = quantize_k_2bit_no_token_dim(k)

        def run_q2():
            return attn_forward_decode(
                q=q_1,
                k_q=k_q,
                k_scale=k_scale,
                k_zero=k_zero,
                k=k,
                v=v,
                k_bits=2,
                scale=scale,
                BS=BS,
                SBS=SBS,
                delta=delta,
                return_skip_ratio=False,
            )

        def run_flash():
            return flash_attn_compute(q, k, v)

        # One forward to obtain skip ratio and validate shapes
        _, skip_ratio = attn_forward_decode(
            q=q_1,
            k_q=k_q,
            k_scale=k_scale,
            k_zero=k_zero,
            k=k,
            v=v,
            k_bits=2,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
        )

        ms_q2 = benchmark(run_q2, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_q2, ms_flash, float(skip_ratio)

    def validate_full(delta):
        q_rope_1 = q_rope_full[:, :, T_full - 1 : T_full, :].contiguous()
        q, k, v = convert_layout(q_rope_1, k_rope_full, v_full)
        q_1 = q.unsqueeze(1)
        k_q, k_scale, k_zero = quantize_k_2bit_no_token_dim(k)

        o_triton, skip_ratio = attn_forward_decode(
            q=q_1,
            k_q=k_q,
            k_scale=k_scale,
            k_zero=k_zero,
            k=k,
            v=v,
            k_bits=2,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
        )
        o_flash = flash_attn_compute(q, k, v)
        max_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().mean().item()
        print(
            f"[Validate] delta={delta} | skip_ratio={skip_ratio:.3%} | "
            f"max_abs={max_abs_vs_flash:.3e}, mean_abs={mean_abs_vs_flash:.3e}"
        )

    print(f"[Info] Running delta={delta}")
    plot_root_dir, raw_data_dir = build_plot_dirs(attn_kernel_name, gpu_tag, BS, SBS, delta, args.layer, max_length, THIS_DIR)
    cache_path = make_cache_file_path(
        raw_data_dir,
        f"layer_{args.layer}",
        T_full,
        Hq,
        Hkv,
        K,
        V,
        BS,
        SBS,
        delta,
        dtype,
        step,
        iters,
        warmup,
    )

    if os.path.exists(cache_path):
        x_lengths, q2_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        q2_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc=f"delta={delta:g}"):
            ms_q2, ms_flash, sr = bench_one_length(L, delta)
            q2_ms_list.append(ms_q2)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)
        x_lengths = lengths
        meta = dict(
            layer_idx=int(args.layer),
            T_full=int(T_full),
            Hq=int(Hq),
            Hkv=int(Hkv),
            D=int(K),
            Dv=int(V),
            BS=int(BS),
            SBS=int(SBS),
            delta=float(delta),
            dtype=dtype_key(dtype),
            step=int(step),
            iters=int(iters),
            warmup=int(warmup),
            attn_kernel=attn_kernel_name,
        )
        save_raw_cache(cache_path, meta, x_lengths, q2_ms_list, flash_ms_list, skip_ratios)
        print(f"[Info] Saved raw benchmark data to {cache_path}")

    plot_path = plot_speed_curve(
        x_lengths,
        q2_ms_list,
        flash_ms_list,
        T_full,
        BS,
        SBS,
        delta,
        f"layer_{args.layer}",
        plot_root_dir,
        attn_kernel_name,
        skip_ratios=skip_ratios,
    )
    print(
        f"[Result] Layer {args.layer} | T={to_k_str(T_full)} | BS={BS} SBS={SBS} delta={delta} | "
        f"Q2={q2_ms_list[-1]:.3f} ms, Flash={flash_ms_list[-1]:.3f} ms"
    )
    print(f"[Result] Saved plot to: {plot_path}")
    validate_full(delta)


if __name__ == "__main__":
    main()
