# 观察固定的某个k相对于后续进来的所有q的相似度的变化

import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_qkvh
from transformers.models.llama.modeling_llama import repeat_kv

if __name__ == "__main__":
    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
    layer_data_root = os.path.join(exp_root, 'layer_data')
    plot_output_root = os.path.join(exp_root, 'k_attn_weights_change_plot-minus-max')

    os.makedirs(plot_output_root, exist_ok=True)

    # Iterate through the layers and plot the attention weights and their distribution
    with torch.no_grad():
        for layer_idx, layer_qkvh_data in enumerate(load_qkvh(layer_data_root)):
            q_rope = layer_qkvh_data["q_rope"]  # [B, H, T, D]
            k_rope = layer_qkvh_data["k_rope"]  # [B, HKV, T, D]

            sample_seq_len = 2048
            if sample_seq_len > 0:
                q_rope = q_rope[:, :, :sample_seq_len, :]
                k_rope = k_rope[:, :, :sample_seq_len, :]
            
            sample_num_heads = 4
            if sample_num_heads > 0:
                q_rope = q_rope[:, :sample_num_heads, :, :]
                k_rope = k_rope[:, :sample_num_heads, :, :]

            bsz, num_heads, seq_len, head_dim = q_rope.shape
            _, num_kv_heads, _, _ = k_rope.shape
            head_group = num_heads // num_kv_heads
            k_rope = repeat_kv(k_rope, head_group)  # -> [B, H, T, D]

            assert bsz == 1, f"Batch size must be 1, but got {bsz}"

            print(f"plot max weights in {layer_idx}")
            max_weights = torch.zeros(num_heads, seq_len)
            for head_idx in range(num_heads):
                for seq_idx in range(seq_len):
                    q_select = q_rope[0, head_idx, seq_idx:seq_idx+1, :]
                    k_select = k_rope[0, head_idx, 0:seq_idx+1, :]
                    attn_weights = torch.matmul(q_select, k_select.transpose(-1, -2)) / math.sqrt(head_dim)
                    max_weights[head_idx, seq_idx] = attn_weights.max()
                
                save_dir = os.path.join(plot_output_root, f"layer_{layer_idx}", f"head_{head_idx}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"max_weights.png")
                plt.figure(figsize=(10, 5))
                plt.plot(max_weights[head_idx])
                plt.savefig(save_path)
                plt.close()

            start_idx = 256
            end_idx = 260
            
            print(f"plot k weights change in {layer_idx}")
            for head_idx in range(num_heads):
                for seq_idx in range(start_idx, end_idx):
                    # 选定某一时刻的 k（形状 [1, D]）
                    k_select = k_rope[0, head_idx, seq_idx:seq_idx+1, :]                     # [1, D]
                    # 取该时刻及之后所有 q（形状 [L_future, D]）
                    q_select = q_rope[0, head_idx, seq_idx:, :]                               # [L_future, D]

                    # scaled dot-product 相似度（未归一化为概率）
                    # 结果形状 [L_future, 1]
                    attn_weights = torch.matmul(q_select, k_select.transpose(-1, -2)) / math.sqrt(head_dim)

                    attn_values = attn_weights.squeeze(-1)                                 # [L_future]
                    attn_values -= max_weights[head_idx, seq_idx:]

                    xs = np.arange(seq_idx, seq_len)                                           # 真实 token 位置
                    ys = attn_values.detach().cpu().float().numpy()

                    # 画图并保存
                    fig = plt.figure(figsize=(40, 8))

                    # 子图1：随位置变化的相似度
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.plot(xs, ys)
                    ax1.set_title(f"Layer {layer_idx} • Head {head_idx}\nfixed k@{seq_idx} vs q@[{seq_idx}..{seq_len-1}]")
                    ax1.set_xlabel("Token position")
                    ax1.set_ylabel("Similarity")

                    # 子图2：分布直方图
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.hist(ys, bins=50)
                    ax2.set_title("Distribution")
                    ax2.set_xlabel("Similarity value")
                    ax2.set_ylabel("Count")

                    plt.tight_layout()

                    save_dir = os.path.join(plot_output_root, f"layer_{layer_idx}", f"head_{head_idx}")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"seq_{seq_idx}.png")
                    plt.savefig(save_path, dpi=300)
                    plt.close(fig)
