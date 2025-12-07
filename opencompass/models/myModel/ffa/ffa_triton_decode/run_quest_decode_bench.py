#!/usr/bin/env python3
"""
Benchmark Quest decode attention against FlashAttention and plot latency curves.
This script mirrors the Quest API usage from example_attention_api.py and the
benchmarking/plotting utilities from run_attn_bench.py.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from utils.bench import benchmark
from utils.cache import dtype_key, save_raw_cache, load_raw_cache, to_k_str
from utils.flash import flash_attn_compute
from utils.load import load_qkvh


DEFAULT_EXP_ROOT_DIR = (
    "/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/"
    "huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
)
DEFAULT_EXP_SUBDIR = "Llama-3_2-3B/longbench_gov_report_48_68_256k"
DEFAULT_PAGE_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quest decode vs FlashAttention benchmark (length sweep)."
    )
    parser.add_argument("--exp-root-dir", type=str, default=DEFAULT_EXP_ROOT_DIR, help="Root directory containing experiment results.")
    parser.add_argument(
        "--exp-subdir",
        type=str,
        default=DEFAULT_EXP_SUBDIR,
        help="Subdirectory under exp-root-dir that holds layer_data.",
    )
    parser.add_argument("--layer-idx", type=int, default=1, help="Layer index to load from layer_data.")
    parser.add_argument("--batch-idx", type=int, default=0, help="Which batch item to use if stored tensors have batch > 1.")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Quest page size.")
    parser.add_argument(
        "--budget-ratio",
        type=float,
        default=1.0,
        help="Fraction of KV pages to keep (<=1.0 uses Top-K); ignored if --page-budget is set.",
    )
    parser.add_argument(
        "--page-budget",
        type=int,
        default=None,
        help="Fixed KV page budget (overrides --budget-ratio).",
    )
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations for each length.")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations before timing.")
    parser.add_argument("--step", type=int, default=1024, help="Step size for length sweep.")
    parser.add_argument("--max-length", type=int, default=None, help="Optional cap on sequence length.")
    parser.add_argument(
        "--quest-repo",
        type=str,
        default=None,
        help="Path to the Quest repository (defaults to ./quest or $QUEST_REPO).",
    )
    parser.add_argument(
        "--plot-root",
        type=str,
        default=None,
        help="Optional directory to store plots/results (defaults to ./plot/quest_decode/...).",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (still writes raw cache).")
    parser.add_argument("--no-cache", action="store_true", help="Disable reading/writing raw cache json.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip correctness check against FlashAttention.")
    return parser.parse_args()


def map_dtype(dtype_str: str):
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def resolve_quest_repo(user_path: Optional[str]) -> Path:
    """
    Add the Quest repo to sys.path. Preference: CLI path -> $QUEST_REPO -> ./quest.
    """
    candidates = []
    if user_path:
        candidates.append(Path(user_path).expanduser())
    env_repo = os.getenv("QUEST_REPO")
    if env_repo:
        candidates.append(Path(env_repo).expanduser())
    candidates.append(Path(__file__).resolve().parent / "quest")

    for repo in candidates:
        if repo is None:
            continue
        if repo.is_dir():
            if str(repo) not in sys.path:
                sys.path.append(str(repo))
            return repo

    raise FileNotFoundError(
        f"Cannot locate Quest repo. Tried: {', '.join(str(c) for c in candidates)}. "
        "Pass --quest-repo or set QUEST_REPO."
    )


def compute_page_budget(total_len: int, page_size: int, budget_pages: Optional[int], budget_ratio: float) -> Tuple[int, int]:
    kv_pages = max(1, (total_len + page_size - 1) // page_size)
    if budget_pages is not None:
        return max(1, min(kv_pages, int(budget_pages))), kv_pages
    ratio = max(0.0, budget_ratio)
    pages = int(math.ceil(kv_pages * ratio))
    return max(1, min(kv_pages, pages)), kv_pages


def make_decode_cache_path(raw_dir: str, layer_idx: int, batch_idx: int, T_full: int, Hq: int, Hkv: int, D: int, Dv: int,
                           page_size: int, budget_tag: str, dtype: torch.dtype, step: int, iters: int, warmup: int,
                           max_length: Optional[int]) -> str:
    lmax_part = f"_lmax{to_k_str(max_length)}" if max_length is not None else ""
    fname = (
        f"quest_layer{layer_idx}_b{batch_idx}_Tmax{to_k_str(T_full)}_Hq{Hq}_Hkv{Hkv}_D{D}_Dv{Dv}"
        f"_ps{page_size}_{budget_tag}_{dtype_key(dtype)}_step{int(step)}_it{int(iters)}_wu{int(warmup)}{lmax_part}.json"
    )
    return os.path.join(raw_dir, fname)


def plot_decode_curve(lengths, quest_ms, flash_ms, T_full, page_size, budget_label, layer_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(lengths, quest_ms, label="Quest decode", marker="o", markersize=2)
    plt.plot(lengths, flash_ms, label="FlashAttention", marker="o", markersize=2)
    plt.xlabel("Sequence length (T)")
    plt.ylabel("Latency per run (ms)")
    plt.title(
        f"Layer {layer_idx} decode latency (Tmax={to_k_str(T_full)}, page_size={page_size}, budget={budget_label})"
    )
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plot_path = os.path.join(
        out_dir,
        f"layer_{layer_idx}_decode_Tmax{to_k_str(T_full)}_ps{page_size}_{budget_label}.png",
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


def quest_decode_attention(q: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor, new_k: torch.Tensor, new_v: torch.Tensor,
                           page_size: int, page_budget: int, quest_utils) -> torch.Tensor:
    """
    Run Quest decode attention (q_len == 1) for a single layer using quest.utils API.
    """
    if q.size(0) != 1:
        raise ValueError("Decode path expects q_len == 1.")

    total_len = past_k.size(0) + new_k.size(0)
    controller = quest_utils.InferenceController(
        num_layers=1,
        num_heads=q.size(1),
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


@torch.inference_mode()
def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Quest decode benchmarking requires a CUDA-capable GPU.")

    torch.set_float32_matmul_precision("high")
    dtype = map_dtype(args.dtype)
    page_size = int(args.page_size)
    step = max(1, int(args.step))

    quest_repo = resolve_quest_repo(args.quest_repo)
    import quest.utils as quest_utils  # noqa: E402

    exp_root = os.path.join(args.exp_root_dir, args.exp_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    layer_iter = load_qkvh(layer_data_root, device="cpu", start_layer=args.layer_idx)
    try:
        layer_data = next(layer_iter)
    except StopIteration:
        raise RuntimeError(f"No data found for layer_{args.layer_idx} in {layer_data_root}")

    q_rope_full = layer_data["q_rope"].to("cuda", dtype=dtype)
    k_rope_full = layer_data["k_rope"].to("cuda", dtype=dtype)
    v_full = layer_data["v"].to("cuda", dtype=dtype)

    if args.batch_idx >= q_rope_full.size(0):
        raise ValueError(f"batch-idx {args.batch_idx} is out of range for batch size {q_rope_full.size(0)}.")

    if args.max_length is not None and args.max_length > 0:
        q_rope_full = q_rope_full[..., : args.max_length, :]
        k_rope_full = k_rope_full[..., : args.max_length, :]
        v_full = v_full[..., : args.max_length, :]

    _, Hq, T_full, D = q_rope_full.shape
    _, Hkv, _, Dv = v_full.shape
    if Hq != Hkv:
        raise ValueError(f"Hq ({Hq}) must match Hkv ({Hkv}) for Quest decode benchmark.")
    if D != Dv:
        raise ValueError(f"Head dims for K ({D}) and V ({Dv}) must match.")

    lengths = list(range(step, T_full, step)) + [T_full]

    budget_tag = f"budget{args.page_budget}" if args.page_budget is not None else f"ratio{args.budget_ratio}"
    lmax_tag = f"_lmax{to_k_str(args.max_length)}" if args.max_length else ""
    this_dir = Path(__file__).resolve().parent
    plot_root_dir = (
        Path(args.plot_root).expanduser()
        if args.plot_root
        else this_dir / "plot" / "quest_decode" / f"ps{page_size}_{budget_tag}"
    )
    plot_root_dir = plot_root_dir / f"layer{args.layer_idx}_b{args.batch_idx}_step{step}{lmax_tag}_{dtype_key(dtype)}"
    raw_data_dir = plot_root_dir / "raw"
    os.makedirs(raw_data_dir, exist_ok=True)

    cache_path = make_decode_cache_path(
        str(raw_data_dir),
        args.layer_idx,
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
        args.max_length,
    )

    q_batch = q_rope_full[args.batch_idx]
    k_batch = k_rope_full[args.batch_idx]
    v_batch = v_full[args.batch_idx]

    def bench_one_length(L: int):
        q_decode = q_batch[:, L - 1, :].unsqueeze(0).contiguous()
        past_k = k_batch[:, : L - 1, :].permute(1, 0, 2).contiguous() if L > 1 else k_batch.new_zeros((0, Hkv, D))
        past_v = v_batch[:, : L - 1, :].permute(1, 0, 2).contiguous() if L > 1 else v_batch.new_zeros((0, Hkv, Dv))
        new_k = k_batch[:, L - 1 : L, :].permute(1, 0, 2).contiguous()
        new_v = v_batch[:, L - 1 : L, :].permute(1, 0, 2).contiguous()

        page_budget, kv_pages = compute_page_budget(L, page_size, args.page_budget, args.budget_ratio)
        skip_ratio = 1.0 - min(1.0, page_budget / kv_pages)

        def run_quest():
            return quest_decode_attention(
                q_decode, past_k, past_v, new_k, new_v, page_size=page_size, page_budget=page_budget, quest_utils=quest_utils
            )

        def run_flash():
            return flash_attn_compute(
                q_batch[None, :, L - 1 : L, :],
                k_batch[None, :, :L, :],
                v_batch[None, :, :L, :],
            )

        ms_quest = benchmark(run_quest, iters=args.iters, warmup=args.warmup)
        ms_flash = benchmark(run_flash, iters=args.iters, warmup=args.warmup)
        return ms_quest, ms_flash, skip_ratio

    if (not args.no_cache) and os.path.exists(cache_path):
        x_lengths, quest_ms_list, flash_ms_list, skip_ratios, _meta = load_raw_cache(cache_path)
    else:
        quest_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L in tqdm(lengths, desc=f"Layer{args.layer_idx} (Quest decode)"):
            ms_quest, ms_flash, sr = bench_one_length(int(L))
            quest_ms_list.append(ms_quest)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)
        x_lengths = lengths

        meta = dict(
            layer_idx=int(args.layer_idx),
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
            dtype=dtype_key(dtype),
            step=int(step),
            iters=int(args.iters),
            warmup=int(args.warmup),
            quest_repo=str(quest_repo),
        )
        if not args.no_cache:
            save_raw_cache(cache_path, meta, x_lengths, quest_ms_list, flash_ms_list, skip_ratios)

    if not args.no_plot:
        plot_path = plot_decode_curve(
            x_lengths,
            quest_ms_list,
            flash_ms_list,
            T_full,
            page_size,
            budget_tag,
            args.layer_idx,
            plot_root_dir,
        )
    else:
        plot_path = None

    print(
        f"Layer {args.layer_idx} | Tmax={to_k_str(T_full)} H={Hq} D={D} | "
        f"page_size={page_size} budget={budget_tag} | Quest={quest_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms"
    )
    if plot_path:
        print(f"Saved plot to: {plot_path}")

    if not args.skip_validate:
        L = T_full
        q_decode = q_batch[:, L - 1, :].unsqueeze(0).contiguous()
        past_k = k_batch[:, : L - 1, :].permute(1, 0, 2).contiguous() if L > 1 else k_batch.new_zeros((0, Hkv, D))
        past_v = v_batch[:, : L - 1, :].permute(1, 0, 2).contiguous() if L > 1 else v_batch.new_zeros((0, Hkv, Dv))
        new_k = k_batch[:, L - 1 : L, :].permute(1, 0, 2).contiguous()
        new_v = v_batch[:, L - 1 : L, :].permute(1, 0, 2).contiguous()
        page_budget, _ = compute_page_budget(L, page_size, args.page_budget, args.budget_ratio)

        quest_out = quest_decode_attention(
            q_decode, past_k, past_v, new_k, new_v, page_size=page_size, page_budget=page_budget, quest_utils=quest_utils
        )
        flash_out = flash_attn_compute(
            q_batch[None, :, L - 1 : L, :],
            k_batch[None, :, :L, :],
            v_batch[None, :, :L, :],
        )

        max_abs = (quest_out.float() - flash_out.float()).abs().max().item()
        mean_abs = (quest_out.float() - flash_out.float()).abs().mean().item()
        print(f"[Validate] Max abs diff={max_abs:.3e}, Mean abs diff={mean_abs:.3e}")


if __name__ == "__main__":
    main()
