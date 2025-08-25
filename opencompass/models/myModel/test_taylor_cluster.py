import torch
from taylor_kv.cache_storage import RemainKVCacheStorage

bsz, num_heads, seq_len, head_dim = 1, 12, 1234, 64
K = torch.randn(bsz, num_heads, seq_len, head_dim)
V = torch.randn(bsz, num_heads, seq_len, head_dim)

RKV = RemainKVCacheStorage(
    name="test",
    cluster_k=10,
    group_size=10,
    order=1,
    u_mode="diag",
    debug=True,
)

RKV.append(K, V)
