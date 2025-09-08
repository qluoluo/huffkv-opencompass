import os
os.environ['FLASH_ATTENTION_TRITON_AMD_DEBUG'] = "1"

import torch

from flash_attn import flash_attn_func
from flash_attn.flash_attn_triton_amd.interface_fa import fw                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                d

if __name__ == "__main__":
    bsz, num_heads, seq_len, head_dim = 1, 8, 134, 64
    Q = torch.randn(bsz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    K = torch.randn(bsz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)
    V = torch.randn(bsz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.bfloat16)

    attn_output = flash_attn_func(
        Q.transpose(1,2),
        K.transpose(1,2),
        V.transpose(1,2),
        causal=True,
    )

    attn_output2 = fwd(
        Q.transpose(1,2),
        K.transpose(1,2),
        V.transpose(1,2),
        causal=True,
    )

    print(torch.max(torch.abs(attn_output - attn_output2)))