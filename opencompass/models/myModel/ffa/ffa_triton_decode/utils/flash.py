# utils/flash.py
def flash_attn_compute(q_rope_1, k_rope, v):
    from flash_attn import flash_attn_func
    # flash_attn_func expects [B, T, H, D]
    # out = flash_attn_func(q_rope_1.transpose(1, 2),
    #                       k_rope.transpose(1, 2),
    #                       v.transpose(1, 2),
    #                       causal=False)
    
    out = flash_attn_func(q_rope_1.unsqueeze(1),
                          k_rope,
                          v,
                          causal=False)
    
    return out.squeeze(0).squeeze(0)