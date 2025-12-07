# run_attn_bench_prefill.py
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from utils.pack import pack_k_hi_lo

def parse_args():
    parser = argparse.ArgumentParser(description="Run attention benchmark (prefill) with configurable hyperparameters.")
    parser.add_argument("--kernel", type=str,
                        default="attn_kernel.prefill_kernel_v0926",
                        help="Python module path for attn_forward_prefill")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--BS", type=int, default=128, help="Block size (BS)")
    parser.add_argument("--BT", type=int, default=128, help="Block size (BT)")
    parser.add_argument("--BK", type=int, default=128, help="Block size for K dimension (BK)")
    parser.add_argument("--delta", type=float, default=5.0, help="Delta parameter for skipping")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations before timing")
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
    if not hasattr(kernel_module, "attn_forward_prefill"):
        raise AttributeError(f"Module {args.kernel} does not define 'attn_forward_prefill'")
    attn_forward_prefill = getattr(kernel_module, "attn_forward_prefill")

    # 可选：外部阈值计算函数（用于 --no-thres-time）
    compute_threshold_external = getattr(kernel_module, "compute_threshold_external", None)

    attn_kernel_name = kernel_module.__name__.split('.')[-1]
    torch.set_float32_matmul_precision("high")

    dtype  = map_dtype(args.dtype)
    BS     = int(args.BS)
    BT     = int(args.BT)
    BK     = int(args.BK)
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
    plot_root_dir = os.path.join(this_dir, "plot_result", f"{attn_kernel_name}",
                                 f"BS{BS}_BK{BK}_delta{delta}_bsz{bsz}" + 
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

    print(f"[Info][Prefill] Using batch size: {bsz_actual}, Hq: {Hq}, Hkv: {Hkv}, T_full: {T_full}, D: {D}, Dv: {Dv}")

    # Prefill：q/k/v 均取前 L 个 token，q_len = kv_len = L
    def bench_one_length(L):
        torch.cuda.empty_cache()
        # [Prefill] 使用 0:L 的前缀，q 长度与 k/v 相同
        q_rope_L = q_rope_full[:, :, :L, :]      # [bsz, Hq, L, D]
        k_rope_L = k_rope_full[:, :, :L, :]      # [bsz, Hkv, L, D]
        v_L      = v_full[:, :, :L, :]           # [bsz, Hkv, L, Dv]

        k_hi8, k_lo8 = None, None

        # 可选：外部预计算阈值（不计入 fused 计时）
        pre_th = None
        if args.no_thres_time:
            if compute_threshold_external is None:
                raise AttributeError(f"Module {args.kernel} does not define 'compute_threshold_external' required by --no-thres-time")
            # 注意：NTB 的计算逻辑与内核保持一致（按 K 轴分块）
            NTB = (L + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_rope_L, k=k_rope_L, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        def run_fused():
            return attn_forward_prefill(
                q=q_rope_L, k=k_rope_L, v=v_L,
                scale=scale, BT=BT, BS=BS, BK=BK, delta=delta, return_skip_ratio=False,
                precomputed_threshold=pre_th
            )

        def run_flash():
            return flash_attn_compute(q_rope_L, k_rope_L, v_L)

        # 运行一次获取 skip ratio（维度随内核实现，通常聚合到一个标量）
        output, sr = attn_forward_prefill(
            q=q_rope_L, k=k_rope_L, v=v_L, k_hi8=k_hi8, k_lo8=k_lo8,
            scale=scale, BT=BT, BS=BS, BK=BK, delta=delta, return_skip_ratio=True,
            precomputed_threshold=pre_th
        )

        ms_fused = benchmark(run_fused, iters=iters, warmup=warmup)
        ms_flash = benchmark(run_flash, iters=iters, warmup=warmup)
        return ms_fused, ms_flash, float(sr)

    def validate_full():
        # [Prefill] 全长 q/k/v，一次性前向

        pre_th = None
        if args.no_thres_time:
            NTB = (T_full + BS - 1) // BS
            pre_th = compute_threshold_external(
                q=q_rope_full, k=k_rope_full, scale=scale, NTB=NTB, delta=delta, HKV=Hkv, HQ=Hq
            )

        o_triton, skip_ratio = attn_forward_prefill(
            q=q_rope_full, k=k_rope_full, v=v_full,
            scale=scale, BT=BT, BS=BS, BK=BK, delta=delta, return_skip_ratio=True,
            precomputed_threshold=pre_th
        )
        o_flash = flash_attn_compute(q_rope_full, k_rope_full, v_full)
        
        # import ipdb; ipdb.set_trace()

        max_abs  = (o_triton.float() - o_flash.float()).abs().max().item()
        mean_abs = (o_triton.float() - o_flash.float()).abs().mean().item()
        print(f"[Prefill] Skipped block ratio: {skip_ratio:.3%} (over HKV x NTBS)")
        print(f"[Prefill] Value diff vs Flash(GQA): max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")

    # 长度 sweep
    step = int(args.step)
    lengths = list(range(step, T_full, step)) + [T_full]

    cache_path = make_cache_file_path(
        raw_data_dir, f"prefill_layers_{layer_indices[0]}-{layer_indices[-1]}", T_full, Hq, Hkv, D, Dv, BS, BK, delta, dtype, step, iters, warmup, bsz=bsz
    )

    # 初始化结果列表
    fused_ms_list, flash_ms_list, skip_ratios = [], [], []
    x_lengths = []

    # 检查是否有现有的缓存文件
    if os.path.exists(cache_path):
        try:
            # 加载现有缓存
            existing_x_lengths, existing_fused_ms, existing_flash_ms, existing_skip_ratios, _meta = load_raw_cache(cache_path)
            
            # 找到最后一个完成的长度点
            last_completed_idx = -1
            for i, (fused, flash, skip) in enumerate(zip(existing_fused_ms, existing_flash_ms, existing_skip_ratios)):
                if fused is not None and flash is not None and skip is not None:
                    last_completed_idx = i
                else:
                    break
            
            if last_completed_idx >= 0:
                # 使用已完成的测试结果
                x_lengths = existing_x_lengths[:last_completed_idx + 1]
                fused_ms_list = existing_fused_ms[:last_completed_idx + 1]
                flash_ms_list = existing_flash_ms[:last_completed_idx + 1]
                skip_ratios = existing_skip_ratios[:last_completed_idx + 1]
                print(f"[Info] Found existing cache with {len(x_lengths)} completed measurements")
                
                # 找出需要继续测试的长度点
                remaining_lengths = [L for L in lengths if L not in x_lengths]
                if remaining_lengths:
                    print(f"[Info] Continuing from length {x_lengths[-1] if x_lengths else 'start'}, {len(remaining_lengths)} lengths remaining")
                else:
                    print(f"[Info] All lengths already measured")
            else:
                # 缓存文件存在但没有有效数据，从头开始
                remaining_lengths = lengths
                print(f"[Info] Cache file exists but contains no valid data, starting from scratch")
        except Exception as e:
            print(f"[Warning] Failed to load cache file {cache_path}: {e}, starting from scratch")
            remaining_lengths = lengths
    else:
        # 没有缓存文件，从头开始
        remaining_lengths = lengths
        print(f"[Info] No existing cache found, starting from scratch")

    # 准备元数据
    meta = dict(
        mode="prefill",
        layer_indices=layer_indices, T_full=int(T_full),
        Hq=int(Hq), Hkv=int(Hkv), D=int(D), Dv=int(Dv),
        BS=int(BS), BK=int(BK), delta=float(delta),
        dtype=dtype_key(dtype), step=int(step),
        iters=int(iters), warmup=int(warmup),
        attn_kernel=attn_kernel_name,
        no_thres_time=bool(args.no_thres_time),
        bsz=int(bsz),
    )

    # 测试剩余的长度点
    if remaining_lengths:
        for L in tqdm(remaining_lengths, desc=f"[Prefill] Layers{layer_indices[0]}_{bsz}"):
            try:
                ms_fused, ms_flash, sr = bench_one_length(L)
                
                # 添加到结果列表
                x_lengths.append(L)
                fused_ms_list.append(ms_fused)
                flash_ms_list.append(ms_flash)
                skip_ratios.append(sr)
                
                # 立即保存到缓存
                save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)
                # print(f"[Info] Saved measurement for L={L}: fused={ms_fused:.3f}ms, flash={ms_flash:.3f}ms, skip_ratio={sr:.3f}")
                
            except Exception as e:
                print(f"[Error] Failed to benchmark length {L}: {e}")
                # 即使出错也保存当前进度，但标记这个点为None
                x_lengths.append(L)
                fused_ms_list.append(None)
                flash_ms_list.append(None)
                skip_ratios.append(None)
                save_raw_cache(cache_path, meta, x_lengths, fused_ms_list, flash_ms_list, skip_ratios)
                continue

    # 过滤掉失败的测量点（值为None的）
    valid_indices = [i for i, (fused, flash, skip) in enumerate(zip(fused_ms_list, flash_ms_list, skip_ratios)) 
                    if fused is not None and flash is not None and skip is not None]
    
    if valid_indices:
        valid_x_lengths = [x_lengths[i] for i in valid_indices]
        valid_fused_ms = [fused_ms_list[i] for i in valid_indices]
        valid_flash_ms = [flash_ms_list[i] for i in valid_indices]
        valid_skip_ratios = [skip_ratios[i] for i in valid_indices]
    else:
        print("[Warning] No valid measurements found!")
        valid_x_lengths, valid_fused_ms, valid_flash_ms, valid_skip_ratios = [], [], [], []

    plot_path = plot_speed_curve(
        valid_x_lengths, valid_fused_ms, valid_flash_ms,
        T_full, BS, BK, delta, f"prefill_layers{layer_indices[0]}_{bsz}", plot_root_dir, attn_kernel_name
    )

    if valid_fused_ms and valid_flash_ms:
        print(f"[Prefill] Layers {layer_indices} | bsz={bsz} | T={to_k_str(T_full)} Hq={Hq} Hkv={Hkv} D={D} Dv={Dv} "
              f"| BS={BS} BK={BK} delta={delta} | Kernel={attn_kernel_name} | "
              f"Fused={valid_fused_ms[-1]:.3f} ms Flash={valid_flash_ms[-1]:.3f} ms")
    else:
        print(f"[Prefill] No valid final measurement available")
    
    print(f"[Prefill] Saved plot to: {plot_path}")

    validate_full()