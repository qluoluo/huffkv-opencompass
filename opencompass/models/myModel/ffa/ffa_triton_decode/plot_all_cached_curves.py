#!/usr/bin/env python3
# plot_all_cached_curves.py
import os
import sys
import argparse
import math
import traceback
from typing import List, Tuple, Dict, Any, Optional

# 尝试将当前脚本目录加入 sys.path，以便导入 utils.*
THIS_FILE = os.path.abspath(__file__)
THIS_DIR = os.path.dirname(THIS_FILE)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 依赖 run_attn_bench.py 同目录的 utils
try:
    from utils.cache import load_raw_cache, to_k_str
except Exception:
    print("[Error] Cannot import utils.cache. Make sure this script is placed next to run_attn_bench.py.", file=sys.stderr)
    raise


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-process: collect all existing raw caches under a specific kernel and plot all curves into one figure."
    )
    parser.add_argument(
        "--kernel", type=str, required=True,
        help="Kernel module path (same format as run_attn_bench.py --kernel), e.g., attn_kernel.attn_kernel_v1022_unfused_grid2d_ht"
    )
    parser.add_argument(
        "--plot-root", type=str, default=None,
        help="Override search root for plot dir. Default: <this_dir>/plot/<kernel_name>"
    )
    parser.add_argument(
        "--layer", type=int, default=1,
        help="Layer index filter (default 1, align with run_attn_bench.py)"
    )
    parser.add_argument(
        "--dtype", type=str, default=None, choices=[None, "fp16", "bf16", "fp32"],
        help="Optional dtype filter, e.g. fp16. Default: None (no filter)"
    )
    parser.add_argument(
        "--save-name", type=str, default=None,
        help="Override output filename (without directory). Default: layer{layer}_summary.png"
    )
    parser.add_argument(
        "--max-legend", type=int, default=64,
        help="Max legend entries to show (avoid too large legends). Default: 64"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output image DPI. Default: 150"
    )
    parser.add_argument(
        "--width", type=float, default=10.0,
        help="Figure width in inches. Default: 10.0"
    )
    parser.add_argument(
        "--height", type=float, default=6.0,
        help="Figure height in inches. Default: 6.0"
    )
    parser.add_argument(
        "--hide-flash", action="store_true",
        help="If set, do not plot Flash curves (only fused curves)."
    )
    parser.add_argument(
        "--y-log", action="store_true",
        help="Use log-scale for Y axis."
    )
    return parser.parse_args()


def kernel_to_dirname(kernel_module: str) -> str:
    """
    Convert kernel module path to dir name used in run_attn_bench.py
    e.g. 'attn_kernel.attn_kernel_v1022_unfused_grid2d_ht' -> 'attn_kernel_v1022_unfused_grid2d_ht'
    """
    return kernel_module.split(".")[-1]


def default_plot_root(this_dir: str, kernel_name: str) -> str:
    """
    run_attn_bench.py 的输出结构：
    <this_dir>/plot/<kernel_name>/BS{BS}_SBS{SBS}_delta{delta}/raw/*.pkl
    """
    return os.path.join(this_dir, "plot", kernel_name)


def find_all_raw_cache_files(kernel_plot_root: str) -> List[str]:
    """
    递归扫描 kernel_plot_root 下的所有 raw 缓存文件。
    我们不强依赖具体的扩展名（.pkl 或 .pt 等），而是尝试用 load_raw_cache 读取。
    """
    if not os.path.isdir(kernel_plot_root):
        return []
    result = []
    for dirpath, _, filenames in os.walk(kernel_plot_root):
        # 仅限定 raw 子目录（按 run_attn_bench.py 约定）
        if os.path.basename(dirpath) != "raw":
            continue
        for fn in filenames:
            # 放宽匹配条件：只要是文件就尝试；load_raw_cache 失败时会被跳过
            full = os.path.join(dirpath, fn)
            if os.path.isfile(full):
                result.append(full)
    return result


def label_from_meta(meta: Dict[str, Any]) -> str:
    # 生成配置标签：BS{BS}_SBS{SBS}_d{delta}_{dtype}[+step{step}]
    # 也可根据需要把 step/iters/warmup 追加上去以区分不同配置
    bs = meta.get("BS", "?")
    sbs = meta.get("SBS", "?")
    delta = meta.get("delta", "?")
    dtype = meta.get("dtype", "?")
    step = meta.get("step", None)

    label = f"BS{bs}_SBS{sbs}_d{delta}_{dtype}"
    # 如果你在 batch 中改变了 step（常见），可以把 step 追加到标签避免误判
    if step is not None:
        label += f"_step{step}"
    return label


def load_one_cache(path: str) -> Optional[Tuple[List[int], List[float], List[float], List[float], Dict[str, Any]]]:
    try:
        x_lengths, fused_ms, flash_ms, skip_ratios, meta = load_raw_cache(path)
        return x_lengths, fused_ms, flash_ms, skip_ratios, meta
    except Exception:
        # 某些文件可能不是我们要的格式，跳过
        print(f"[Warn] Failed to load cache: {path}\n{traceback.format_exc()}")
        return None


def main():
    args = parse_args()

    kernel_name = kernel_to_dirname(args.kernel)
    if args.plot_root is None:
        plot_root = default_plot_root(THIS_DIR, kernel_name)
    else:
        plot_root = args.plot_root

    if not os.path.isdir(plot_root):
        print(f"[Error] Plot root not found: {plot_root}")
        sys.exit(1)

    # 收集所有 raw 缓存
    raw_files = find_all_raw_cache_files(plot_root)
    if not raw_files:
        print(f"[Info] No raw cache found under: {plot_root}")
        sys.exit(0)

    print(f"[Info] Found {len(raw_files)} raw file(s). Start loading...")
    curves = []  # list of dict: {x, fused, flash, meta, label}
    skipped = 0

    for f in sorted(raw_files):
        loaded = load_one_cache(f)
        if loaded is None:
            skipped += 1
            continue

        x_lengths, fused_ms, flash_ms, skip_ratios, meta = loaded

        # 按 layer 过滤
        if meta.get("layer_idx", None) != args.layer:
            continue
        # 按 dtype 过滤
        if args.dtype is not None and meta.get("dtype", None) != args.dtype:
            continue

        label = label_from_meta(meta)
        curves.append({
            "x": x_lengths,
            "fused": fused_ms,
            "flash": flash_ms,
            "meta": meta,
            "label": label,
            "path": f,
        })

    if not curves:
        print(f"[Info] No matched curves (after filtering). Raw files total={len(raw_files)}, skipped={skipped}.")
        sys.exit(0)

    print(f"[Info] Got {len(curves)} curve(s) to plot. (skipped={skipped})")

    # 画图
    os.makedirs(os.path.join(plot_root, "summary"), exist_ok=True)
    save_name = args.save_name or f"layer{args.layer}_summary.png"
    save_path = os.path.join(plot_root, "summary", save_name)

    plt.figure(figsize=(args.width, args.height))

    # 固定颜色循环，fused 实线，flash 同色虚线
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    # 按标签排序，保证颜色稳定
    curves.sort(key=lambda d: d["label"])

    # 标题信息：取 T_full/Hq/Hkv/D/Dv 可变，若不一致就只显示 kernel/layer
    # 尝试从第一个 meta 读出一些信息
    meta0 = curves[0]["meta"]
    try:
        T_full0 = meta0.get("T_full", None)
        title_left = f"{kernel_name} | layer {args.layer}"
        if T_full0 is not None:
            title_left += f" | T_full={to_k_str(T_full0)}"
        plt.title(title_left)
    except Exception:
        plt.title(f"{kernel_name} | layer {args.layer}")

    # X 轴：token length，Y 轴：ms
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Latency (ms)")
    if args.y_log:
        plt.yscale("log")

    legend_entries = 0
    color_idx = 0

    for c in curves:
        x = c["x"]
        fused = c["fused"]
        flash = c["flash"]
        label = c["label"]

        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

        # fused
        h1, = plt.plot(x, fused, color=color, linestyle="-", linewidth=2.0, label=label)
        legend_entries += 1

        # flash（可选）
        if not args.hide_flash:
            h2, = plt.plot(x, flash, color=color, linestyle="--", linewidth=1.5, label=label + "+flash")
            legend_entries += 1

        # 限制 legend 数量，避免太大
        if legend_entries >= args.max_legend:
            print(f"[Info] Legend entries reached max ({args.max_legend}). The rest will not appear in legend, but still plotted.")
            # 之后的线条不再带 label
            break

    # 对剩下的（如果超过 max_legend），继续画但不加 legend
    for c in curves[(legend_entries // (2 if not args.hide_flash else 1)):]:
        x = c["x"]
        fused = c["fused"]
        flash = c["flash"]
        color = color_cycle[color_idx % len(color_cycle)]
        color_idx += 1

        plt.plot(x, fused, color=color, linestyle="-", linewidth=2.0)
        if not args.hide_flash:
            plt.plot(x, flash, color=color, linestyle="--", linewidth=1.5)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=args.dpi)
    print(f"[Done] Saved summary figure to: {save_path}")


if __name__ == "__main__":
    main()
