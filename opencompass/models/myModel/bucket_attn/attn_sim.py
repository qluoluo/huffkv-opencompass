import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from .utils import load_qkvh
from transformers.models.llama.modeling_llama import repeat_kv
from flash_attn import flash_attn_func
# from attn_utils import flash_part_attn, matmul_part_attn

@torch.no_grad()
def bucket_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    accurate_bound: float = -3.0,
    lost_bound: float = -10.0,
    bucket_step: float = 0.1,
    split_output: bool = False,
):
    """
    近似注意力：对 logits ∈ [accurate_bound, 0] 做精确；(lost_bound, accurate_bound) 做分桶近似；
    ≤ lost_bound 的区域权重极小（近似丢弃）。要求 q_len == 1。

    输入:
      q: (B, Hq, 1, D)
      k: (B, Hkv, K, D)
      v: (B, Hkv, K, D)
    输出:
      (B, Hq, 1, D)  或  (up, down) 若 split_output=True
    """
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "q/k/v must be 4D tensors"
    bsz, num_heads, q_len, head_dim = q.shape
    _, num_kv_heads, k_len, _ = k.shape
    assert q_len == 1, f"q_len must be 1, got {q_len}"
    assert accurate_bound < 0.0 and lost_bound < accurate_bound, "Require lost_bound < accurate_bound < 0"
    assert bucket_step > 0.0, "step must be > 0"

    # 重复 KV 头以匹配注意力头
    assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
    head_group = num_heads // num_kv_heads
    k = repeat_kv(k, head_group)  # (B, Hq, K, D)
    v = repeat_kv(v, head_group)  # (B, Hq, K, D)

    dev = q.device
    dt = q.dtype

    # 计算对数注意力，做稳定化（最大值平移）
    logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)  # (B,H,1,K)
    logits = logits - logits.max(dim=-1, keepdim=True).values           # <= 0

    # -------- 精确区间 [accurate_bound, 0] --------
    accurate_mask = (logits >= accurate_bound)                           # (B,H,1,K)
    if accurate_mask.any():
        logits_acc = logits.masked_fill(~accurate_mask, float("-inf"))
        w_up = torch.exp(logits_acc)                                     # (B,H,1,K)
        up_acc = torch.matmul(w_up, v)                                   # (B,H,1,D)
        down_acc = w_up.sum(dim=-1, keepdim=True)                        # (B,H,1,1)
    else:
        up_acc = torch.zeros((bsz, num_heads, 1, head_dim), device=dev, dtype=v.dtype)
        down_acc = torch.zeros((bsz, num_heads, 1, 1), device=dev, dtype=v.dtype)

    # -------- 分桶区间 (lost_bound, accurate_bound) --------
    # 桶个数与边界
    bucket_num = int(math.ceil((accurate_bound - lost_bound) / bucket_step))
    # 用 linspace 生成 bucket_num+1 个边界点（包含两端）
    bounds = torch.linspace(lost_bound, accurate_bound, bucket_num + 1, device=dev, dtype=logits.dtype)
    mids = 0.5 * (bounds[:-1] + bounds[1:])                              # (bucket_num,)
    mids_exp = torch.exp(mids).view(1, 1, bucket_num, 1)                 # (1,1,B,1)

    # 压掉 q 轴
    logits_b = logits.squeeze(-2)                                        # (B,H,K)
    # 只在 (lost_bound, accurate_bound) 内做分桶
    mask_b = (logits_b < accurate_bound) & (logits_b > lost_bound)       # (B,H,K)

    # 为了稳妥，先把 logits clamp 到 [lost_bound, accurate_bound)
    # bucketize 返回 [1, bucket_num]，减 1 后在 [0, bucket_num-1]
    logits_clamped = logits_b.clamp(min=lost_bound, max=accurate_bound - 1e-12)
    bucket_idx = torch.bucketize(logits_clamped, bounds) - 1             # (B,H,K)
    bucket_idx = torch.clamp(bucket_idx, 0, bucket_num - 1)

    # 只对分桶区域累计（mask=False 的位置当作 0）
    v_masked = v * mask_b.unsqueeze(-1)                                  # (B,H,K,D)

    # sum(v) 按桶累加 -> (B,H,Bucket,D)
    idx_exp = bucket_idx.unsqueeze(-1).expand_as(v_masked)               # (B,H,K,D)
    sum_bucket = torch.zeros((bsz, num_heads, bucket_num, head_dim), device=dev, dtype=v.dtype)
    sum_bucket.scatter_add_(dim=2, index=idx_exp, src=v_masked)

    # count 按桶累加 -> (B,H,Bucket,1)
    ones = mask_b.unsqueeze(-1).to(v.dtype)                              # (B,H,K,1)
    cnt_bucket = torch.zeros((bsz, num_heads, bucket_num, 1), device=dev, dtype=v.dtype)
    cnt_bucket.scatter_add_(dim=2, index=bucket_idx.unsqueeze(-1), src=ones)

    # 用区间中点的 exp 近似 softmax 权重：沿桶维求和，得到 (B,H,1,D) / (B,H,1,1)
    up_bucket = (sum_bucket * mids_exp).sum(dim=2, keepdim=True)         # (B,H,1,D)
    down_bucket = (cnt_bucket * mids_exp).sum(dim=2, keepdim=True)       # (B,H,1,1)

    # -------- 合并并输出 --------
    up = up_acc + up_bucket
    down = down_acc + down_bucket

    # 数值稳定性
    eps = torch.finfo(up.dtype).eps
    out = up / torch.clamp(down, min=eps)

    return (up, down) if split_output else out



if __name__ == "__main__":
    # exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_narrativeqa_42'
    exp_root = '/inspire/hdd/project/heziweiproject/liuxiaoran-240108120089/projects_zgliu/projects/huffkv/attn_analysis/result/Llama-3_2-3B/longbench_gov_report_46'
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

        bsz, num_heads, _, head_dim = q_rope.shape
        _, num_kv_heads, seq_len, _ = k_rope.shape
        head_group = num_heads // num_kv_heads
        k_rope = repeat_kv(k_rope, head_group)
        v = repeat_kv(v, head_group)

        assert bsz == 1, f"Batch size must be 1, but got {bsz}"
        print(f"input {seq_len=}")

        # Calculate attention weights
        # attn_weights = torch.matmul(q_rope, k_rope.transpose(-1, -2)) / math.sqrt(head_dim)
        # attn_weights -= torch.max(attn_weights, dim=-1, keepdim=True)[0]

        attn_output_flash = flash_attn_func(q_rope.transpose(1,2), k_rope.transpose(1,2), v.transpose(1,2), causal=True).transpose(1,2)
        attn_output_torch = torch.nn.functional.scaled_dot_product_attention(q_rope, k_rope, v, is_causal=True)
        attn_output_bucket = bucket_attn(q_rope, k_rope, v)

        print(f'{attn_output_flash.shape=}, {attn_output_bucket.shape=}\n')

        print("torch vs flash")
        print(f'max {torch.max(torch.abs(attn_output_torch - attn_output_flash))} mean {torch.mean(torch.abs(attn_output_torch - attn_output_flash))}\n')

        print("flash vs bucket")
        print(f'max {torch.max(torch.abs(attn_output_flash - attn_output_bucket))} mean {torch.mean(torch.abs(attn_output_flash - attn_output_bucket))}\n')

        print("torch vs bucket")
        print(f'max {torch.max(torch.abs(attn_output_torch - attn_output_bucket))} mean {torch.mean(torch.abs(attn_output_torch - attn_output_bucket))}\n')

        exit()