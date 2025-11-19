# utils/cache.py
import os
import json
import torch

__all__ = [
    "dtype_key", "to_k_str",
    "make_cache_file_path",
    "save_raw_cache", "load_raw_cache"
]

def dtype_key(dt: torch.dtype) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(dt, str(dt))

def to_k_str(n: int) -> str:
    val = n / 1024.0
    return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"

def make_cache_file_path(raw_data_dir, layer_idx, T_full, Hq, Hkv, D, Dv, BS, SBS, delta, dtype, step, iters, warmup, bsz=1):
    def _to_k(n: int) -> str:
        val = n / 1024.0
        return f"{int(val)}k" if abs(val - int(val)) < 1e-9 else f"{val:.1f}k"
    fname = (
        f"layer_{layer_idx}_Tmax{_to_k(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_BS{BS}_SBS{SBS}_delta{delta:g}_{dtype_key(dtype)}"
        f"_step{step}_it{iters}_wu{warmup}_bsz{bsz}.json"
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