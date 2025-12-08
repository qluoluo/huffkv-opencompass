# Local single-length benchmark for configurable attention kernels
import argparse
import importlib
import math
import os

import torch

from utils.bench import benchmark
from utils.flash import flash_attn_compute
from utils.load import load_qkvh

EXP_ROOT_DIR = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
EXP_ROOT_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a local single-length benchmark for a Triton attention kernel.")
    parser.add_argument(
        "--kernel",
        type=str,
        default="attn_kernel.attn_kernel_v1109_fused_bsz",
        help="Python module path that exposes attn_forward_decode/convert_to_triton_layout/pack_k_hi_lo",
    )
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--target-length", type=int, default=2048, help="Sequence length to benchmark (clipped to data)")
    parser.add_argument("--BS", type=int, default=128, help="Block size for the kernel")
    parser.add_argument("--SBS", type=int, default=None, help="Sub block size; defaults to BS when omitted")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta threshold for skipping")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations")
    parser.add_argument("--bsz", type=int, default=1, help="Number of layers to combine (batch size)")
    parser.add_argument(
        "--exp-root-dir",
        type=str,
        default=EXP_ROOT_DIR,
        help="Root directory for experiment data (contains EXP_ROOT_SUBDIR)",
    )
    parser.add_argument(
        "--exp-root-subdir",
        type=str,
        default=EXP_ROOT_SUBDIR,
        help="Subdirectory under exp root where layer_data lives",
    )
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
    required = ("attn_forward_decode", "convert_to_triton_layout", "pack_k_hi_lo")
    missing = [name for name in required if not hasattr(kernel_module, name)]
    if missing:
        raise AttributeError(f"Module {kernel_path} is missing required symbols: {missing}")
    return (
        kernel_module,
        kernel_module.attn_forward_decode,
        kernel_module.convert_to_triton_layout,
        kernel_module.pack_k_hi_lo,
    )


def load_layer_batch(layer_data_root, layer_indices, dtype):
    layer_qkvh_data_list = []
    layer_qkvh_data_iter = load_qkvh(layer_data_root, device="cpu", start_layer=layer_indices[0])
    for i, layer_idx in enumerate(layer_indices):
        try:
            layer_data = next(layer_qkvh_data_iter)
        except StopIteration:
            raise RuntimeError(f"Not enough layers for batch size {len(layer_indices)}. Loaded {i} layers.")
        layer_qkvh_data_list.append(layer_data)
        print(f"[Info] Loaded data for layer_{layer_idx}")

    q_rope_full = torch.cat([d["q_rope"] for d in layer_qkvh_data_list], dim=0).to("cuda", dtype=dtype)
    k_rope_full = torch.cat([d["k_rope"] for d in layer_qkvh_data_list], dim=0).to("cuda", dtype=dtype)
    v_full = torch.cat([d["v"] for d in layer_qkvh_data_list], dim=0).to("cuda", dtype=dtype)
    return q_rope_full, k_rope_full, v_full


def main():
    args = parse_args()
    dtype = map_dtype(args.dtype)
    BS = int(args.BS)
    SBS = BS if args.SBS is None else int(args.SBS)
    delta = float(args.delta)
    target_length = int(args.target_length)
    iters = int(args.iters)
    warmup = int(args.warmup)
    bsz = int(args.bsz)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this benchmark.")

    torch.set_float32_matmul_precision("high")

    (kernel_module, attn_forward_decode, convert_to_triton_layout, pack_k_hi_lo) = load_kernel_components(args.kernel)
    attn_kernel_name = kernel_module.__name__.split(".")[-1]
    print(f"[Info] Using attention kernel: {attn_kernel_name} ({kernel_module.__name__})")

    exp_root = os.path.join(args.exp_root_dir, args.exp_root_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")
    layer_indices = list(range(1, 1 + bsz))

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, layer_indices, dtype)
    B, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, _ = k_rope_full.shape
    _, _, _, V = v_full.shape
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be divisible by Hkv ({Hkv})")

    L = min(target_length, T_full)
    scale = 1.0 / math.sqrt(K)

    print(
        f"[Setup] Using data {layer_data_root} | layer_indices={layer_indices} | "
        f"L={L}/{T_full} B={B} Hq={Hq} Hkv={Hkv} K={K} V={V} dtype={dtype} BS={BS} SBS={SBS} delta={delta}"
    )

    q_rope_1 = q_rope_full[:, :, L - 1 : L, :]
    k_rope = k_rope_full[:, :, :L, :]
    v = v_full[:, :, :L, :]

    q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
    k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

    o_ref, skip_ratio = attn_forward_decode(
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
    )
    torch.cuda.synchronize()

    o_flash = flash_attn_compute(q_rope_1, k_rope, v)
    if o_flash.dim() == 2:
        o_flash = o_flash.unsqueeze(0)

    max_abs = (o_ref.float() - o_flash.float()).abs().max().item()
    mean_abs = (o_ref.float() - o_flash.float()).abs().mean().item()

    def run_fused():
        return attn_forward_decode(
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
        )

    def run_flash():
        return flash_attn_compute(q_rope_1, k_rope, v)

    ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
    ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
    print(f"[Bench] attn_forward_decode: {ms_fused:.3f} ms (avg over {iters} iters, warmup={warmup})")
    print(f"[Bench] flash_attn:   {ms_flash:.3f} ms (avg over {iters} iters, warmup={warmup})")
    print(
        f"[Stats] skip_ratio={skip_ratio:.3%}, "
        f"output_norm={o_ref.float().norm().item():.4f}, "
        f"flash_norm={o_flash.float().norm().item():.4f}, "
        f"max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}"
    )


if __name__ == "__main__":
    main()
