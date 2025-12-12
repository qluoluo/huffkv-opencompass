#!/usr/bin/env python3
"""
Benchmark Quest decode attention vs FlashAttention2 on recorded layer data.

This mirrors run_attn_bench_q2.py: it sweeps sequence lengths on real Q/K/V
captures, benchmarks Quest decode against FlashAttention2, caches the raw
latency numbers, and plots speed curves under ./plot/.
"""
import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers.models.llama.modeling_llama import repeat_kv
from tqdm import tqdm

# Make ffa_triton_decode/utils importable (bench/cache/flash/load/plot helpers).
THIS_DIR = Path(__file__).resolve().parent
FFA_DECODE_DIR = THIS_DIR.parent / "ffa" / "ffa_triton_decode"
if str(FFA_DECODE_DIR) not in sys.path:
    sys.path.append(str(FFA_DECODE_DIR))

from utils.bench import benchmark  # noqa: E402
from utils.cache import dtype_key, load_raw_cache, save_raw_cache, to_k_str  # noqa: E402
from utils.flash import flash_attn_compute  # noqa: E402
from utils.load import load_qkvh  # noqa: E402

# Default recorded data root (same as run_attn_bench_q2.py)
EXP_ROOT_DIR = (
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/"
    "huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
EXP_ROOT_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quest decode vs FlashAttention2 benchmark on recorded Q/K/V."
    )
    parser.add_argument("--exp-root-dir", type=str, default=EXP_ROOT_DIR, help="Root directory that holds attn_analysis/result.")
    parser.add_argument("--exp-subdir", type=str, default=EXP_ROOT_SUBDIR, help="Subdir under exp-root-dir containing layer_data/.")
    parser.add_argument("--layer", type=int, default=0, help="Layer index to benchmark.")
    parser.add_argument("--batch-idx", type=int, default=0, help="Batch index inside the recorded tensors.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--page-size", type=int, default=16, help="Quest page size.")
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=1.0,
        help="Fraction of KV pages to keep (<=1.0). Ignored when --page-budget is set.",
    )
    parser.add_argument(
        "--page-budget",
        type=int,
        default=None,
        help="Optional fixed KV page budget; overrides --budget-ratio when set.",
    )
    parser.add_argument("--step", type=int, default=4096, help="Step size for length sweep.")
    parser.add_argument("--max-length", type=int, default=32768, help="Cap the max sequence length (<=0 means use full length).")
    parser.add_argument("--iters", type=int, default=50, help="Benchmark iterations for each length.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations before timing.")
    parser.add_argument("--no-cache", action="store_true", help="Do not load/save raw latency cache.")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (still runs benchmark).")
    parser.add_argument("--skip-validate", action="store_true", help="Skip correctness check vs FlashAttention2 at Tmax.")
    return parser.parse_args()


def map_dtype(dtype_str: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_str]


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


def compute_page_budget(total_len: int, page_size: int, budget_pages: Optional[int], budget_ratio: float) -> Tuple[int, int]:
    kv_pages = max(1, (total_len + page_size - 1) // page_size)
    if budget_pages is not None:
        return max(1, min(kv_pages, int(budget_pages))), kv_pages
    ratio = max(0.0, budget_ratio)
    pages = int(math.ceil(kv_pages * ratio))
    return max(1, min(kv_pages, pages)), kv_pages


def _expand_kv_to_mha(kv: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """
    Expand [T, num_kv_heads, head_dim] KV to [T, num_q_heads, head_dim] using HF repeat_kv.
    """
    num_kv_heads = kv.size(1)
    if num_q_heads == num_kv_heads:
        return kv
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads ({num_q_heads}) must be a multiple of num_kv_heads ({num_kv_heads}).")

    group_size = num_q_heads // num_kv_heads
    kv_4d = kv.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, T, D]
    expanded = repeat_kv(kv_4d, group_size)  # [1, num_q_heads, T, D]
    return expanded.squeeze(0).permute(1, 0, 2).contiguous()  # [T, num_q_heads, D]


def quest_decode_attention(
    q: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    page_size: int,
    page_budget: int,
    quest_utils,
) -> torch.Tensor:
    """
    Run Quest decode attention (q_len == 1) for a single layer.
    Inputs are expected in [T, H, D] layout for K/V and [1, H, D] for q.
    """
    if q.size(0) != 1:
        raise ValueError("Decode path expects q_len == 1.")

    num_q_heads = q.size(1)
    num_kv_heads = past_k.size(1)
    total_len = past_k.size(0) + new_k.size(0)

    controller = quest_utils.InferenceController(
        num_layers=1,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=q.size(2),
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=total_len,
        dtype=q.dtype,
        device=q.device,
    )

    controller.prepare_metadata(past_k.size(0))
    controller.begin_forward(past_k.size(0))
    quest_utils.append_kv(past_k, past_v, controller, layer_idx=0)
    controller.end_forward()

    controller.prepare_metadata(new_k.size(0))
    controller.begin_forward(new_k.size(0))
    quest_utils.append_kv(new_k, new_v, controller, layer_idx=0)

    if controller.need_estimate():
        estimated = quest_utils.decode_estimate(q, controller, layer_idx=0)
        quest_utils.decode_topk(estimated, controller)
        topk_indices = controller.topk_dindices_buffer
    else:
        topk_indices = controller.kv_indices_without_last

    out = quest_utils.decode_sparse_attn(
        q,
        controller,
        layer_idx=0,
        topk_indices=topk_indices,
    )
    controller.end_forward()
    return out


def build_plot_dirs(gpu_tag: str, layer: int, page_size: int, budget_tag: str, dtype_str: str, step: int, max_length: Optional[int]):
    lmax_part = f"_lmax{to_k_str(max_length)}" if max_length is not None and max_length > 0 else ""
    plot_root_dir = (
        THIS_DIR
        / "plot"
        / "quest_vs_flash"
        / gpu_tag
        / f"layer{layer}_ps{page_size}_{budget_tag}_step{step}_{dtype_str}{lmax_part}"
    )
    raw_data_dir = plot_root_dir / "raw"
    plot_root_dir.mkdir(parents=True, exist_ok=True)
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    return plot_root_dir, raw_data_dir


def make_cache_path(
    raw_dir: Path,
    layer: int,
    batch_idx: int,
    T_full: int,
    Hq: int,
    Hkv: int,
    D: int,
    Dv: int,
    page_size: int,
    budget_tag: str,
    dtype: torch.dtype,
    step: int,
    iters: int,
    warmup: int,
    max_length: Optional[int],
) -> Path:
    lmax_part = f"_lmax{to_k_str(max_length)}" if max_length is not None and max_length > 0 else ""
    fname = (
        f"quest_layer{layer}_b{batch_idx}_Tmax{to_k_str(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_ps{page_size}_{budget_tag}_{dtype_key(dtype)}_step{step}_it{iters}_wu{warmup}{lmax_part}_prepout.json"
    )
    return raw_dir / fname


def plot_speed_curve(
    x_lengths,
    quest_ms,
    flash_ms,
    skip_ratios,
    T_full,
    layer_idx,
    page_size,
    budget_tag,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(12, 8))

    line_quest, = ax1.plot(x_lengths, quest_ms, label="Quest decode", marker="o", markersize=2)
    line_flash, = ax1.plot(x_lengths, flash_ms, label="FlashAttention2", marker="o", markersize=2)
    ax1.set_xlabel("Sequence length (T)")
    ax1.set_ylabel("Latency per run (ms)")
    ax1.set_title(
        f"Layer {layer_idx} decode latency (Tmax={to_k_str(T_full)}, page_size={page_size}, budget={budget_tag})"
    )
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = [line_quest, line_flash]
    labels = ["Quest decode", "FlashAttention2"]

    if skip_ratios:
        ax2 = ax1.twinx()
        skip_pct = [sr * 100.0 for sr in skip_ratios]
        line_skip, = ax2.plot(
            x_lengths,
            skip_pct,
            label="Skipped pages (%)",
            color="tab:green",
            linestyle="--",
            marker="x",
            markersize=2,
        )
        ax2.set_ylabel("Skipped pages (%)")
        ax2.set_ylim(0, 100)
        lines.append(line_skip)
        labels.append("Skipped pages (%)")

    ax1.legend(lines, labels)
    plot_path = out_dir / f"layer_{layer_idx}_quest_vs_flash_Tmax{to_k_str(T_full)}_ps{page_size}_{budget_tag}.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


@torch.inference_mode()
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")

    dtype = map_dtype(args.dtype)
    max_length = None if args.max_length is not None and args.max_length <= 0 else int(args.max_length)
    step = max(1, int(args.step))
    page_size = int(args.page_size)

    import quest.utils as quest_utils  # noqa: E402

    gpu_tag, gpu_name, gpu_mem_gb, gpu_idx = get_gpu_info()
    print(f"[Info] Using GPU[{gpu_idx}]: {gpu_name} ({gpu_mem_gb}GB)")

    exp_root = os.path.join(args.exp_root_dir, args.exp_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    layer_iter = load_qkvh(layer_data_root, device="cpu", start_layer=args.layer)
    try:
        layer_data = next(layer_iter)
    except StopIteration:
        raise RuntimeError(f"No data found for layer_{args.layer} in {layer_data_root}")

    q_rope_full = layer_data["q_rope"]
    k_rope_full = layer_data["k_rope"]
    v_full = layer_data["v"]

    if max_length is not None and max_length > 0:
        q_rope_full = q_rope_full[..., :max_length, :]
        k_rope_full = k_rope_full[..., :max_length, :]
        v_full = v_full[..., :max_length, :]

    q_rope_full = q_rope_full.to("cuda", dtype=dtype)
    k_rope_full = k_rope_full.to("cuda", dtype=dtype)
    v_full = v_full.to("cuda", dtype=dtype)

    if args.batch_idx >= q_rope_full.size(0):
        raise ValueError(f"batch-idx {args.batch_idx} is out of range for batch size {q_rope_full.size(0)}.")

    q_batch = q_rope_full[args.batch_idx]  # [Hq, T, D]
    k_batch = k_rope_full[args.batch_idx]  # [Hkv, T, D]
    v_batch = v_full[args.batch_idx]  # [Hkv, T, Dv]

    Hq, T_full, D = q_batch.shape
    Hkv = k_batch.shape[0]
    Dv = v_batch.shape[2]
    if Hq % Hkv != 0:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({Hkv}) for GQA expansion.")
    if D != Dv:
        raise ValueError(f"Head dims differ: D={D}, Dv={Dv}.")

    lengths = list(range(step, T_full, step)) + [T_full]
    budget_tag = f"budget{args.page_budget}" if args.page_budget is not None else f"ratio{args.budget_ratio}"
    dtype_str = dtype_key(dtype)
    plot_root_dir, raw_data_dir = build_plot_dirs(
        gpu_tag, args.layer, page_size, budget_tag, dtype_str, step, max_length
    )
    cache_path = make_cache_path(
        raw_data_dir,
        args.layer,
        args.batch_idx,
        int(T_full),
        int(Hq),
        int(Hkv),
        int(D),
        int(Dv),
        page_size,
        budget_tag,
        dtype,
        step,
        args.iters,
        args.warmup,
        max_length,
    )

    def prepare_length_inputs(L: int):
        """
        Slice/expand/pad once per length so the benchmark only times the kernels.
        """
        q_decode = q_batch[:, L - 1, :].unsqueeze(0).contiguous()  # [1, Hq, D]
        k_prefix = k_batch[:, :L, :].permute(1, 0, 2).contiguous()  # [L, Hkv, D]
        v_prefix = v_batch[:, :L, :].permute(1, 0, 2).contiguous()  # [L, Hkv, Dv]

        past_k = k_prefix[:-1]
        past_v = v_prefix[:-1]
        new_k = k_prefix[-1:].contiguous()
        new_v = v_prefix[-1:].contiguous()

        past_k_mha = _expand_kv_to_mha(past_k, Hq)
        past_v_mha = _expand_kv_to_mha(past_v, Hq)
        new_k_mha = _expand_kv_to_mha(new_k, Hq)
        new_v_mha = _expand_kv_to_mha(new_v, Hq)

        page_budget, kv_pages = compute_page_budget(L, page_size, args.page_budget, args.budget_ratio)
        skip_ratio = 1.0 - min(1.0, page_budget / kv_pages)

        # quest.decode_topk currently expects num_heads == 32. When we use sparsity (page_budget < kv_pages),
        # pad heads to 32 for both Quest and FlashAttention to keep shapes aligned.
        effective_heads = Hq if page_budget >= kv_pages else max(Hq, 32)
        if effective_heads < Hq:
            raise ValueError(f"effective_heads {effective_heads} must be >= Hq {Hq}")
        if effective_heads > Hq:
            pad_h = effective_heads - Hq
            q_decode = torch.cat([q_decode, q_decode.new_zeros((1, pad_h, D))], dim=1)
            past_k_mha = torch.cat([past_k_mha, past_k_mha.new_zeros((past_k_mha.size(0), pad_h, past_k_mha.size(2)))], dim=1)
            past_v_mha = torch.cat([past_v_mha, past_v_mha.new_zeros((past_v_mha.size(0), pad_h, past_v_mha.size(2)))], dim=1)
            new_k_mha = torch.cat([new_k_mha, new_k_mha.new_zeros((new_k_mha.size(0), pad_h, new_k_mha.size(2)))], dim=1)
            new_v_mha = torch.cat([new_v_mha, new_v_mha.new_zeros((new_v_mha.size(0), pad_h, new_v_mha.size(2)))], dim=1)

        k_flash = torch.cat([past_k_mha, new_k_mha], dim=0).unsqueeze(0)  # [1, L, effective_heads, D]
        v_flash = torch.cat([past_v_mha, new_v_mha], dim=0).unsqueeze(0)  # [1, L, effective_heads, Dv]

        return dict(
            q_decode=q_decode,
            past_k_mha=past_k_mha,
            past_v_mha=past_v_mha,
            new_k_mha=new_k_mha,
            new_v_mha=new_v_mha,
            k_flash=k_flash,
            v_flash=v_flash,
            page_budget=page_budget,
            skip_ratio=skip_ratio,
        )

    def bench_one_length(L: int):
        prep = prepare_length_inputs(L)

        def run_quest():
            return quest_decode_attention(
                prep["q_decode"],
                prep["past_k_mha"],
                prep["past_v_mha"],
                prep["new_k_mha"],
                prep["new_v_mha"],
                page_size=page_size,
                page_budget=prep["page_budget"],
                quest_utils=quest_utils,
            )

        def run_flash():
            return flash_attn_compute(prep["q_decode"], prep["k_flash"], prep["v_flash"])

        ms_quest = benchmark(run_quest, iters=args.iters, warmup=args.warmup)
        ms_flash = benchmark(run_flash, iters=args.iters, warmup=args.warmup)
        return ms_quest, ms_flash, prep["skip_ratio"]

    if (not args.no_cache) and cache_path.exists():
        x_lengths, quest_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
        print(f"[Info] Loaded cached results from {cache_path}")
    else:
        quest_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc=f"Layer{args.layer} (Quest vs Flash)"):
            ms_quest, ms_flash, sr = bench_one_length(int(L))
            quest_ms_list.append(ms_quest)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)
        x_lengths = lengths
        meta = dict(
            layer_idx=int(args.layer),
            batch_idx=int(args.batch_idx),
            T_full=int(T_full),
            Hq=int(Hq),
            Hkv=int(Hkv),
            D=int(D),
            Dv=int(Dv),
            page_size=int(page_size),
            budget_tag=str(budget_tag),
            budget_ratio=float(args.budget_ratio),
            page_budget=None if args.page_budget is None else int(args.page_budget),
            dtype=dtype_str,
            step=int(step),
            iters=int(args.iters),
            warmup=int(args.warmup),
            gpu=str(gpu_tag),
            prep_outside_timing=True,
        )
        if not args.no_cache:
            save_raw_cache(cache_path, meta, x_lengths, quest_ms_list, flash_ms_list, skip_ratios)
            print(f"[Info] Saved raw benchmark data to {cache_path}")

    if not args.no_plot:
        plot_path = plot_speed_curve(
            x_lengths,
            quest_ms_list,
            flash_ms_list,
            skip_ratios,
            T_full,
            args.layer,
            page_size,
            budget_tag,
            plot_root_dir,
        )
        print(f"[Result] Saved plot to: {plot_path}")

    print(
        f"[Result] Layer {args.layer} | Tmax={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} | "
        f"page_size={page_size} budget={budget_tag} | Quest={quest_ms_list[-1]:.3f} ms, Flash={flash_ms_list[-1]:.3f} ms"
    )

    if not args.skip_validate:
        L = lengths[-1]
        prep = prepare_length_inputs(L)

        quest_out = quest_decode_attention(
            prep["q_decode"],
            prep["past_k_mha"],
            prep["past_v_mha"],
            prep["new_k_mha"],
            prep["new_v_mha"],
            page_size=page_size,
            page_budget=prep["page_budget"],
            quest_utils=quest_utils,
        )
        flash_out = flash_attn_compute(
            prep["q_decode"],
            prep["k_flash"],
            prep["v_flash"],
        )

        if flash_out.dim() == 2:  # flash_attn_compute squeezes to [H, D]
            flash_out = flash_out.unsqueeze(0)

        quest_slice = quest_out[:, :Hq, :]
        flash_slice = flash_out[:, :Hq, :]
        max_abs = (quest_slice.float() - flash_slice.float()).abs().max().item()
        mean_abs = (quest_slice.float() - flash_slice.float()).abs().mean().item()
        print(f"[Validate] Max abs diff={max_abs:.3e}, Mean abs diff={mean_abs:.3e}")


if __name__ == "__main__":
    main()
