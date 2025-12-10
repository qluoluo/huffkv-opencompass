# Benchmark-style test for 2-bit quantized kernel using the same data interface as run_attn_bench.py.
import argparse
import importlib
import math
import os
import re
import time

import torch

from utils.cache import dtype_key
from utils.load import load_qkvh
from utils.bench import benchmark
from utils.flash import flash_attn_compute

# Ensure attn_kernel package is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in os.sys.path:
    os.sys.path.append(THIS_DIR)

from attn_kernel.attn_kernel_v1210_fused_bsz_q2 import attn_forward_decode_quantized

EXP_ROOT_DIR = "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
EXP_ROOT_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"


def parse_args():
    p = argparse.ArgumentParser(description="Run quantized attn kernel test with recorded layer data.")
    p.add_argument("--kernel", type=str, default="attn_kernel.attn_kernel_v1210_fused_bsz_q2")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--BS", type=int, default=128)
    p.add_argument("--SBS", type=int, default=None)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--layer", type=int, default=1, help="Layer index to load")
    p.add_argument("--max-length", type=int, default=None, help="If set, truncate to this length")
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
    return k_q, scale, zero


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

    # Load kernel module (for sanity)
    kernel_module = importlib.import_module(args.kernel)
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode_quantized", attn_forward_decode_quantized)

    exp_root = os.path.join(EXP_ROOT_DIR, EXP_ROOT_SUBDIR)
    layer_data_root = os.path.join(exp_root, "layer_data")

    q_rope_full, k_rope_full, v_full = load_layer_batch(layer_data_root, args.layer, dtype, args.max_length)

    B, Hq, T_full, K = q_rope_full.shape
    _, Hkv, _, V = v_full.shape
    scale = 1.0 / math.sqrt(K)

    q_rope_1 = q_rope_full[:, :, T_full - 1 : T_full, :].contiguous()
    q, k, v = convert_layout(q_rope_1, k_rope_full, v_full)
    q_1 = q.unsqueeze(1)  # [B,1,HQ,K]

    k_q, k_scale, k_zero = quantize_k_2bit_no_token_dim(k)
    G = q.shape[1] // k.shape[2]

    def run_q2():
        return attn_forward_decode(
            q=q_1,
            k_q=k_q,
            k_scale=k_scale,
            k_zero=k_zero,
            v=v,
            k_bits=2,
            scale=scale,
            BS=BS,
            SBS=SBS,
            delta=delta,
            return_skip_ratio=True,
        )

    def run_flash():
        return flash_attn_compute(q, k, v)

    # One forward for correctness / skip ratio
    o_triton, skip_ratio = run_q2()
    o_flash = run_flash()

    ms_q2 = benchmark(lambda: run_q2()[0], iters=args.iters, warmup=args.warmup)
    ms_flash = benchmark(lambda: run_flash(), iters=args.iters, warmup=args.warmup)

    max_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().max().item()
    mean_abs_vs_flash = (o_triton.float() - o_flash.float()).abs().mean().item()

    gpu_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(gpu_idx)
    gpu_tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", props.name.strip())

    print(f"[Q2] GPU={gpu_tag} BS={BS} SBS={SBS} delta={delta} dtype={dtype_key(dtype)}")
    print(f"[Q2] Shape: B={B}, Hq={Hq}, Hkv={Hkv}, T={T_full}, K={K}, V={V}")
    print(f"[Q2] Kernel: {kernel_module.__name__.split('.')[-1]}, time={ms_q2:.3f} ms, skip_ratio={skip_ratio:.3%}")
    print(f"[Q2] Flash time: {ms_flash:.3f} ms")
    print(f"[Q2] triton vs flash: max_abs={max_abs_vs_flash:.3e}, mean_abs={mean_abs_vs_flash:.3e}")


if __name__ == "__main__":
    main()
