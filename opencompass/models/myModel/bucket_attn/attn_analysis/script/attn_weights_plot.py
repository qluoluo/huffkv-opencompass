import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_qkvh
from transformers.models.llama.modeling_llama import repeat_kv

def plot_attention_weights(attn_weights, layer_idx, output_dir):
    """
    Plot the attention weights as a heatmap and save it as a PNG image.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(attn_weights.squeeze().cpu().float().numpy(), cmap='viridis', aspect='auto')
    ax.set_title(f'Attention Weights for Layer {layer_idx}')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_attn_weights.png'))
    plt.close(fig)

def plot_attention_weights_distribution(attn_weights, layer_idx, output_dir, exp=False):
    """
    Plot the distribution (histogram) of the attention weights' values after applying exp.
    """
    if exp:
        attn_weights = torch.exp(attn_weights)  # Apply exponential transformation

    flattened_weights = attn_weights.flatten().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(30, 18))
    
    # Plot histogram and calculate bin counts
    counts, bins, patches = ax.hist(flattened_weights, bins=100, color='skyblue', edgecolor='black')

    # Label each bin with the count
    for count, bin_edge in zip(counts, bins):
        ax.text(bin_edge + (bins[1] - bins[0]) / 2, count, str(int(count)), ha='center', va='bottom', fontsize=6)

    ax.set_title(f'Attention Weights Distribution for Layer {layer_idx}')
    ax.set_xlabel('Attention Weight Value')
    ax.set_ylabel('Frequency')

    # Save the distribution plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'layer_{layer_idx}_attn_weights_distribution{"_exp" if exp else ""}.png'), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')
    plot_output_root = os.path.join(exp_root, 'attn_weights_plot')

    os.makedirs(plot_output_root, exist_ok=True)

    # Iterate through the layers and plot the attention weights and their distribution
    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        q_rope = layer_qkvh_data["q_rope"]
        k_rope = layer_qkvh_data["k_rope"]

        sample_seq_len = -1
        if sample_seq_len > 0:
            q_rope = q_rope[:, :, :sample_seq_len, :]
            k_rope = k_rope[:, :, :sample_seq_len, :]

        q_rope = q_rope[..., -1:, :]  # Focusing on the last position (query part)

        bsz, num_heads, seq_len, head_dim = q_rope.shape
        _, num_kv_heads, _, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"

        # Calculate attention weights
        attn_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_weights -= torch.max(attn_weights, dim=-1, keepdim=True)[0]

        # Plot the attention weights heatmap
        # plot_attention_weights(attn_weights, layer_idx, plot_output_root)

        # Plot the distribution of attention weights values (with exp applied)
        plot_attention_weights_distribution(attn_weights, layer_idx, plot_output_root, exp=False)
        plot_attention_weights_distribution(attn_weights, layer_idx, plot_output_root, exp=True)

    print("Plots generated successfully!")
