# utils/plot.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .cache import to_k_str

__all__ = ["plot_speed_curve"]

def plot_speed_curve(x_lengths, fused_ms_list, flash_ms_list,
                     T_full, BS, SBS, delta, layer_idx, out_dir, attn_kernel_name=None):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.plot(x_lengths, fused_ms_list, label="Triton fused", marker="o", markersize=2)
    plt.plot(x_lengths, flash_ms_list, label="FlashAttn", marker="o", markersize=2)
    plt.xlabel("Sequence length (T)")
    plt.ylabel("Latency per run (ms)")
    plt.ylim(0, 0.4)
    Tmax_k_str = to_k_str(T_full)
    
    # 在标题中包含 attn kernel 名称
    kernel_info = f" | Kernel: {attn_kernel_name}" if attn_kernel_name else ""
    plt.title(f"Layer {layer_idx} Speed vs Length (Tmax={Tmax_k_str}, BS={BS}, SBS={SBS}, delta={delta}{kernel_info})")
    
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    
    # 在文件名中包含 attn kernel 名称
    if attn_kernel_name:
        plot_path = os.path.join(out_dir, f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}_{attn_kernel_name}.png")
    else:
        plot_path = os.path.join(out_dir, f"layer_{layer_idx}_speed_Tmax{Tmax_k_str}.png")
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path