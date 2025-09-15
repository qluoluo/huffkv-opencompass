import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_qkvh
from transformers.models.llama.modeling_llama import repeat_kv
from flash_attn import flash_attn_func

def bucket_attn(q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    accurate_bound: float= -5,
    lost_bound: float= -20,
    step: float= 0.1,
):

    accurate_bound = -5
    lost_bound = -20
    step = 0.1

    bsz, num_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape
    head_group = num_heads // num_kv_heads
    k = repeat_kv(k, head_group)
    v = repeat_kv(v, head_group)
    
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    attn_weights -= torch.max(attn_weights, dim=-1, keepdim=True)[0]

    # sort
    attn_weights, new_indices = torch.sort(attn_weights, dim=-1)
    k, v = k[..., new_indices, :], v[..., new_indices, :]
    
    bucket_num = (accurate_bound - lost_bound) / step
    bound_list = torch.arange(lost_bound, accurate_bound, step, device='cuda')
    interval_mid_list = torch.arange(lost_bound + step / 2, accurate_bound, step, device='cuda')
    interval_mid_exp = torch.exp(interval_mid_list)

    bound_idx = torch.bucketize(attn_weights, bound_list)

    value_sum_bucket = torch.zeros((bsz, num_heads, bucket_num, head_dim), device='cuda', dtype=torch.bfloat16)
    value_count_bucket = torch.zeros((bsz, num_heads, bucket_num, 1), device='cuda', dtype=torch.bfloat16)
    
    k_accurate = k[..., bound_idx == 0, :]
    v_accurate = v[..., bound_idx == 0, :]

    # accurate_attn_output_up, accurate_attn_output_down = compute_part_attn(q, k_accurate, v_accurate)

    for i in range(bucket_num):
        value_sum_bucket[..., i, :] = torch.sum(v[..., bound_idx == i, :], dim=-2, keepdim=True)
        value_count_bucket[..., i, :] = torch.sum(bound_idx == i, dim=-1, keepdim=True)

    import ipdb; ipdb.set_trace()

    bucket_attn_output_down = torch.sum(value_count_bucket * interval_mid_exp.unsqueeze(0).unsqueeze(0), dim=-1, keepdim=True)
    
    return value_sum_bucket


def compute_attn_from_bucket(attn_weights: torch.Tensor, v_sum: torch.Tensor, split_output: bool=False):
    attn_output_down = torch.sum(attn_weights, dim=-1, keepdim=True)
    attn_output_up = attn_weights * v_sum


if __name__ == "__main__":
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    layer_data_root = os.path.join(exp_root, 'layer_data')

    # Iterate through the layers and plot the attention weights and their distribution
    for layer_idx, layer_qkvh_data in tqdm(enumerate(load_qkvh(layer_data_root))):
        q_rope = layer_qkvh_data["q_rope"].to('cuda')
        k_rope = layer_qkvh_data["k_rope"].to('cuda')
        v = layer_qkvh_data["v"].to('cuda')

        sample_seq_len = -1
        if sample_seq_len > 0:
            q_rope = q_rope[:, :, :sample_seq_len, :]
            k_rope = k_rope[:, :, :sample_seq_len, :]

        q_rope = q_rope[..., -1:, :]  # Focusing on the last position (query part)

        bsz, num_heads, seq_len, head_dim = q_rope.shape
        _, num_kv_heads, _, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"

        # Calculate attention weights
        attn_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_weights -= torch.max(attn_weights, dim=-1, keepdim=True)[0]

        attn_output = flash_attn_func(q_rope.transpose(1,2), k_rope.transpose(1,2), v.transpose(1,2), causal=True).transpose(1,2)

        attn_output_bucket = bucket_attn(q_rope, k_rope, v)