import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.llama.modeling_llama import repeat_kv
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func
import math
from typing import Tuple

def generate_casual_mask(q_len, k_len, device='cpu') -> torch.Tensor:
    # 生成滑动窗口掩码，允许每个位置i仅关注前window_size个位置（不包括i自身）
    max_len = max(q_len, k_len)
    mask = torch.ones(max_len, max_len, dtype=torch.bool).triu(1)
    mask = mask[None, None, -q_len:, -k_len:].to(device)
    return mask

def matmul_part_attn(
        q:torch.Tensor,
        k:torch.Tensor, 
        v:torch.Tensor, 
        mask=None
    )-> Tuple[torch.Tensor, torch.Tensor]:
    '''
    qkv形状 (bsz, num_heads, seq_len, head_dim)
    return: (bsz, num_heads, seq_len, head_dim), (bsz, num_heads, seq_len, 1)
    '''
    bsz, num_heads, seq_len, head_dim = q.shape
    attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)

    if mask is not None:
        print(f"{attn_weights.shape=}, {mask.shape=}")

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, -float('inf'))

    attn_weights_up = torch.exp(attn_weights)
    attn_weights_down = torch.sum(attn_weights_up, dim=-1, keepdim=True)
    attn_weights_up = torch.matmul(attn_weights_up, v)

    # attn_weights_up = attn_weights_up.transpose(1,2)
    # attn_weights_down = attn_weights_down.transpose(1,2)

    return attn_weights_up, attn_weights_down

def flash_part_attn(
        q:torch.Tensor,
        k:torch.Tensor, 
        v:torch.Tensor, 
        causal:bool=True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    qkv形状 (bsz, seq_len, num_heads, head_dim)
    返回形状 (bsz, seq_len, num_heads, head_dim), (bsz, seq_len, num_heads, 1)
    '''

    bsz, num_heads, seq_len, head_dim = q.shape
    output, softmax_lse, _ = flash_attn_func(
        q,
        k,
        v,
        causal=causal,
        return_attn_probs=True
    )
    output_down = torch.exp(softmax_lse[..., None].transpose(1,2))
    output_up = output * output_down

    return output_up, output_down

if __name__ == '__main__':

    bsz, num_heads, seq_len, head_dim = 1, 8, 128, 64

    Q = torch.randn(bsz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    full_output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    full_output_up, full_output_down = matmul_part_attn(Q, K, V, mask=generate_casual_mask(seq_len, seq_len, Q.device))

    print(f'{full_output.shape=}\n{full_output_up.shape=}, {full_output_down.shape=}')

    flash_output_up, flash_output_down = flash_part_attn(
        Q.transpose(1,2), 
        K.transpose(1,2), 
        V.transpose(1,2), 
        causal=True
    )

    flash_output_up = flash_output_up.transpose(1,2)
    flash_output_down = flash_output_down.transpose(1,2)

    print(f'{flash_output_up.shape=}, {flash_output_down.shape=}')

    # print(f'{torch.mean(torch.abs(full_output_up - flash_output_up))=}')
    # print(f'{torch.mean(torch.abs(full_output_down - flash_output_down))=}')
    # print(f'{full_output_up - flash_output_up=}')
    # print(f'{full_output_down - flash_output_down=}')

    # print(f'{full_output - flash_output_up/flash_output_down=}')
    print(f'{torch.max(torch.abs(full_output - flash_output_up/flash_output_down))=}')
    print(f'{torch.max(torch.abs(full_output - full_output_up/full_output_down))=}')