# utils/flash.py
def flash_attn_compute(q_rope, k_rope, v, causal=True):
    from flash_attn import flash_attn_func
    # flash_attn_func expects [B, T, H, D]
    out = flash_attn_func(q_rope.transpose(1, 2),
                          k_rope.transpose(1, 2),
                          v.transpose(1, 2),
                          causal=causal)
    return out