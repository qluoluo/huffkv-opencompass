# run_attn_tabel.py
# -----------------------------------------------------------------------------
# 说明:
# - 仅测试固定的几个长度（不做 step sweep）。默认长度列表为 [8k, 32k, 64k, 128k, 256k]。
# - 不画图，不使用任何缓存：不加载、不保存 raw cache，所有数据均实时重算。
# - 仅提供 BS 和 SBS 两个列表，程序内部做笛卡尔积组合，并自动过滤仅保留 SBS <= BS 的组合。
# - 输出为 CSV 表格（每一行是 (BS, SBS) 配置，每一列是不同长度）：
#     - fused_times.csv    每行 (BS, SBS) + 各长度 fused 耗时（毫秒）
#     - flash_times.csv    每行 (BS, SBS) + 各长度 flash 耗时（毫秒）
#     - skip_ratios.csv    每行 (BS, SBS) + 各长度跳过比例
# - 表头中的长度列名以 k 为单位展示（例如 len_8k、len_32k 等）。
# - 输出目录：<this_dir>/tabel/<kernel_name>/
#
# 使用示例:
#   python run_attn_tabel.py \
#       --kernel attn_kernel.attn_kernel_v1022_fused_grid1d \
#       --dtype fp16 \
#       --BS-list 64,128,256 \
#       --SBS-list 64,128 \
#       --delta 5.0 \
#       --iters 200 \
#       --warmup 200 \
#       --lengths 8k,32k,64k,128k,256k
# -----------------------------------------------------------------------------

import os
import sys
import math
import argparse
import importlib
from typing import List, Tuple

from tqdm import tqdm
import torch

from utils.bench import benchmark
from utils.flash import flash_attn_compute
from utils.load import load_qkvh


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention benchmark over a grid of (BS, SBS) without cache or plotting.")
    parser.add_argument("--kernel", type=str,
                        default="attn_kernel.attn_kernel_v1022_fused_grid1d",
                        help="Python module path for attn_forward_decode")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--BS-list", type=str, default="128,256",
                        help="Comma-separated list of BS, e.g., '128,256,512'")
    parser.add_argument("--SBS-list", type=str, default="128,256",
                        help="Comma-separated list of SBS, e.g., '128,256,512'")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta parameter for skipping")
    parser.add_argument("--iters", type=int, default=1000, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=1000, help="Warmup iterations before timing")
    # 仅测固定长度，默认 [8k, 32k, 64k, 128k, 256k]
    parser.add_argument(
        "--lengths",
        type=str,
        default="8k,32k,64k,128k,256k",
        help="Comma-separated lengths with optional k suffix, e.g. '8k,32k,65536'"
    )
    # 为兼容保留 step 但不使用
    parser.add_argument("--step", type=int, default=1024, help="(Ignored) Step size for length sweep")
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


def dtype_key_of(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def to_k_str(T: int) -> str:
    # 展示成小写 k
    if T >= 1024:
        if T % 1024 == 0:
            return f"{T // 1024}k"
        return f"{T / 1024:.1f}k"
    return str(T)


def parse_pairs_from_lists(bs_list_str: str, sbs_list_str: str) -> List[Tuple[int, int]]:
    bs_list = [int(x.strip()) for x in bs_list_str.split(",") if x.strip()]
    sbs_list = [int(x.strip()) for x in sbs_list_str.split(",") if x.strip()]
    pairs = []
    for bs in bs_list:
        for sbs in sbs_list:
            if sbs <= bs:  # 只保留 SBS <= BS
                pairs.append((int(bs), int(sbs)))
    # 去重，保持顺序
    seen = set()
    uniq = []
    for p in pairs:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def parse_lengths_k(lengths_str: str) -> List[int]:
    # 支持形如 "8k, 32k, 65536"；k/K 表示 *1024
    vals = []
    seen = set()
    for tok in lengths_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        lower = tok.lower()
        if lower.endswith("k"):
            num = lower[:-1].strip()
            if not num:
                raise ValueError(f"Invalid length token: {tok}")
            L = int(float(num) * 1024)
        else:
            L = int(float(lower))
        if L <= 0:
            continue
        if L not in seen:
            vals.append(L)
            seen.add(L)
    return vals


if __name__ == "__main__":
    args = parse_args()

    # 动态加载内核模块
    kernel_module = importlib.import_module(args.kernel)
    if not hasattr(kernel_module, "attn_forward_decode"):
        raise AttributeError(f"Module {args.kernel} does not define 'attn_forward_decode'")
    attn_forward_decode = getattr(kernel_module, "attn_forward_decode")

    # 同时从内核模块使用合并后的 layout 工具
    if not hasattr(kernel_module, "convert_to_triton_layout") or not hasattr(kernel_module, "pack_k_hi_lo"):
        raise AttributeError(f"Module {args.kernel} must define 'convert_to_triton_layout' and 'pack_k_hi_lo'")

    convert_to_triton_layout = getattr(kernel_module, "convert_to_triton_layout")
    pack_k_hi_lo = getattr(kernel_module, "pack_k_hi_lo")

    attn_kernel_name = kernel_module.__name__.split('.')[-1]
    torch.set_float32_matmul_precision("high")

    dtype  = map_dtype(args.dtype)
    delta  = float(args.delta)
    iters  = int(args.iters)
    warmup = int(args.warmup)

    # 工程相关路径（与原脚本一致）
    exp_root_dir  = "/inspire/hdd/project/embodied-multimodality/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result"
    exp_root_subdir = "Llama-3_2-3B/longbench_gov_report_48_68_256k"
    exp_root = os.path.join(exp_root_dir, exp_root_subdir)
    layer_data_root = os.path.join(exp_root, "layer_data")

    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)

    # 输出 CSV 的目录（不画图、不缓存）：tabel/<kernel_name>/
    tabel_root = os.path.join(this_dir, "tabel")
    tabel_dir = os.path.join(tabel_root, attn_kernel_name)
    os.makedirs(tabel_dir, exist_ok=True)
    print(f"[Info] Output directory: {tabel_dir}")

    # 只使用 layer_1 的数据（写死）
    layer_idx = 1

    # 加载 layer_1 的数据（只加载一次）
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

    # 固定长度列表（过滤到不超过 T_full 的部分）
    raw_lengths = parse_lengths_k(args.lengths)
    lengths = [L for L in raw_lengths if L <= T_full]
    dropped = [L for L in raw_lengths if L > T_full]
    if not lengths:
        raise ValueError(
            f"All specified lengths exceed available sequence length T_full={T_full} "
            f"(requested: {raw_lengths})."
        )
    if dropped:
        print(f"[Warn] Dropped lengths (exceed T_full={T_full}): {', '.join(map(to_k_str, dropped))}")

    length_labels = [to_k_str(L) for L in lengths]

    # 生成 (BS, SBS) 组合（只保留 SBS <= BS）
    bs_sbs_pairs = parse_pairs_from_lists(args.BS_list, args.SBS_list)
    if not bs_sbs_pairs:
        print("No (BS, SBS) pairs after filtering (SBS <= BS). Exiting.")
        sys.exit(0)

    # 三个表：fused、flash、skip ratio
    # row key: (BS, SBS) -> list of values over lengths
    fused_table = {}
    flash_table = {}
    skip_table  = {}

    def bench_one_length(L, BS, SBS):
        q_rope_1 = q_rope_full[:, :, L-1:L, :]   # 取最后一个 query 步长
        k_rope   = k_rope_full[:, :, :L, :]
        v        = v_full[:, :, :L, :]

        # 使用合并到内核模块的布局工具，得到 K/V 为 [T, Hkv, D]
        q_triton, k_triton_fp16, v_triton = convert_to_triton_layout(q_rope_1, k_rope, v)
        k_hi8, k_lo8 = pack_k_hi_lo(k_triton_fp16)

        def run_fused():
            return attn_forward_decode(
                q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
                scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=False
            )

        def run_flash():
            return flash_attn_compute(q_rope_1, k_rope, v)

        # 单次执行以获取 skip ratio（不计时）
        _, sr = attn_forward_decode(
            q=q_triton, k_hi8=k_hi8, k_lo8=k_lo8, k_fp16=k_triton_fp16, v=v_triton,
            scale=scale, BS=BS, SBS=SBS, delta=delta, return_skip_ratio=True
        )

        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fused, ms_flash, float(sr)

    # 遍历每个 (BS, SBS): 不使用缓存，每次重算
    for (BS, SBS) in bs_sbs_pairs:
        fused_ms_list, flash_ms_list, skip_ratios = [], [], []
        for L, label in tqdm(list(zip(lengths, length_labels)), desc=f"(BS={BS},SBS={SBS})"):
            ms_fused, ms_flash, sr = bench_one_length(L, BS, SBS)
            fused_ms_list.append(ms_fused)
            flash_ms_list.append(ms_flash)
            skip_ratios.append(sr)

        fused_table[(BS, SBS)] = list(fused_ms_list)
        flash_table[(BS, SBS)] = list(flash_ms_list)
        skip_table[(BS, SBS)]  = list(skip_ratios)

        # 简单输出该组合在最后一个长度下的结果
        print(f"[Summary] Layer {layer_idx} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
              f"| BS={BS} SBS={SBS} delta={delta} | Kernel={attn_kernel_name} | "
              f"Fused={fused_ms_list[-1]:.3f} ms Flash={flash_ms_list[-1]:.3f} ms | Skip={skip_ratios[-1]:.3%}")

    # 写出 CSV 表（每个表一份文件）
    def write_table_csv(table_dict, filename: str):
        # 表头: BS,SBS, len_<L1_k>, len_<L2_k>, ...
        header_cols = ["BS", "SBS"] + [f"len_{lbl}" for lbl in length_labels]
        out_path = os.path.join(tabel_dir, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(",".join(header_cols) + "\n")
            for (BS, SBS) in bs_sbs_pairs:
                vals = table_dict[(BS, SBS)]
                row = [str(BS), str(SBS)] + [f"{v:.6f}" for v in vals]
                f.write(",".join(row) + "\n")
        print(f"[Info] Wrote CSV to: {out_path}")

    write_table_csv(fused_table, "fused_times.csv")
    write_table_csv(flash_table, "flash_times.csv")
    write_table_csv(skip_table,  "skip_ratios.csv")

    # 额外写出长度列表，便于查阅（包含原始数值与 k 标记）
    lengths_path = os.path.join(tabel_dir, "lengths.csv")
    with open(lengths_path, "w", encoding="utf-8") as f:
        f.write(",".join(["length", "length_k"])+ "\n")
        for L, lbl in zip(lengths, length_labels):
            f.write(f"{L},{lbl}\n")
    print(f"[Info] Wrote lengths to: {lengths_path}")

    print("[Done]")
