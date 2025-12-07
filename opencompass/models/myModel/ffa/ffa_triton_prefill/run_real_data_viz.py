import os
import sys
import math
import torch
import argparse
import importlib
import matplotlib.pyplot as plt

# 复用原本的工具函数
from utils.load import load_qkvh
from utils.flash import flash_attn_compute

# 导入你的 Kernel 中的可视化函数
# 注意：假设你的 kernel 文件名为 prefill_kernel_v1203.py
from attn_kernel.prefill_kernel_v1203 import attn_forward_prefill, plot_attention_heatmap

def parse_args():
    parser = argparse.ArgumentParser(description="Run attention kernel on real data for validation and visualization.")
    
    # 核心参数
    parser.add_argument("--layer-idx", type=int, default=10, help="要分析的 Transformer 层索引")
    parser.add_argument("--pattern", type=str, default="ffa", choices=["dense", "local", "ffa"], help="稀疏模式")
    parser.add_argument("--delta", type=float, default=0.1, help="阈值 Delta")
    
    # 路径配置 (保留了你原始文件中的路径)
    parser.add_argument("--data-root", type=str, 
                        default="/inspire/hdd/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/huffkv-opencompass/opencompass/models/myModel/ffa/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_48_68_256k/layer_data",
                        help="Data directory path")
    
    # Kernel 参数
    parser.add_argument("--BS", type=int, default=128, help="Block size (BS)")
    parser.add_argument("--BT", type=int, default=128, help="Block size (BT)")
    parser.add_argument("--BK", type=int, default=128, help="Block size for K dimension (BK)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 环境设置
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Running on {device} | Layer: {args.layer_idx} | Pattern: {args.pattern}")

    # 2. 加载真实数据
    if not os.path.exists(args.data_root):
        raise FileNotFoundError(f"Data path not found: {args.data_root}")

    print(f"[Info] Loading data from {args.data_root}...")
    # load_qkvh 返回的是一个 iterator，我们需要跳过前面不需要的层
    layer_iter = load_qkvh(args.data_root, device='cpu', start_layer=args.layer_idx)
    
    try:
        layer_data = next(layer_iter) # 获取指定层的数据
    except StopIteration:
        print(f"[Error] Layer {args.layer_idx} not found in data.")
        sys.exit(1)

    # 提取 QKV 并转到 GPU
    q_rope = layer_data["q_rope"].to(device) # [1, Hq, T, D]
    k_rope = layer_data["k_rope"].to(device) # [1, Hkv, T, D]
    v = layer_data["v"].to(device)           # [1, Hkv, T, Dv]

    # 确保数据是半精度 (根据你的 kernel 要求)
    if q_rope.dtype == torch.float32:
        q_rope = q_rope.half()
        k_rope = k_rope.half()
        v = v.half()

    # 获取维度信息
    bsz, Hq, T, D = q_rope.shape
    _, Hkv, _, Dv = v.shape
    
    print(f"[Info] Data Shape: B={bsz}, T={T}, Hq={Hq}, Hkv={Hkv}, D={D}")

    # 3. 运行 Triton Kernel (开启可视化和 Skip Ratio)
    print(f"[Info] Running Kernel (Pattern={args.pattern}, Delta={args.delta})...")
    
    try:
        # 调用 Kernel
        # 注意：这里我们只跑一次全长 Forward，不做分段 length sweep
        o_triton, skip_ratio, viz_mask = attn_forward_prefill(
            q=q_rope, k=k_rope, v=v,
            BT=args.BT, BS=args.BS, BK=args.BK, # 确保这里的 BS 参数与 kernel 定义一致
            delta=args.delta,
            pattern=args.pattern,
            return_skip_ratio=True,
            return_viz=True
        )
    except Exception as e:
        print(f"[Error] Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. 运行 Flash Attention (Ground Truth) 进行对比
    print("[Info] Running Ground Truth (Flash Attention)...")
    o_ref = flash_attn_compute(q_rope, k_rope, v)

    # 5. 计算误差
    # 转换 float 进行比较以避免溢出
    diff = (o_triton.float() - o_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print("=" * 40)
    print(f"Validation Results (Layer {args.layer_idx}):")
    print(f"Pattern      : {args.pattern}")
    print(f"Skip Ratio   : {skip_ratio:.2%}")
    print(f"Max Diff     : {max_diff:.6f}")
    print(f"Mean Diff    : {mean_diff:.6f}")
    print("=" * 40)

    # 6. 保存可视化热力图
    # 创建输出目录
    output_dir = "viz_results"
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"L{args.layer_idx}_{args.pattern}_d{args.delta}_T{T}.png"
    save_path = os.path.join(output_dir, filename)
    
    print(f"[Info] Generating Heatmap: {save_path}")
    # 调用 kernel 文件里写好的 plot 函数
    # 注意：viz_mask 在 GPU 上，plot 函数内部通常会转 cpu，但如果是多 Batch 需要取 [0]
    # 你的 plot_attention_heatmap 似乎处理 [HQ, NQ, NB] 维度，这里我们要确保传入正确的维度
    
    # 如果 batch size > 1，我们只画第一个样本
    viz_to_plot = viz_mask[0:Hq] if bsz == 1 else viz_mask.reshape(bsz, Hq, -1, -1)[0]
    
    plot_attention_heatmap(viz_to_plot, filename=save_path)
    print("[Info] Done.")

if __name__ == "__main__":
    main()