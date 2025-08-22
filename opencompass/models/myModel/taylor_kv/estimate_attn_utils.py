import torch

def _unsqueeze_before_last(x: torch.Tensor, n: int, idx_from_end: int = 1):
    for _ in range(n):
        x = x.unsqueeze(-idx_from_end-1)
    return x

# @torch.no_grad()
# def preprocess_stats_bh(K: torch.Tensor, V: torch.Tensor):
#     # print(f"preprocess_stats_bh input {K.shape=} {V.shape=}")
#     assert K.dim() == 4 and V.dim() == 4, "K,V 应为 [B, H, N, d]"

#     B, H, N, d_k = K.shape
#     d_v = V.shape[-1]
#     device, dtype = K.device, K.dtype

#     kappa = K.mean(dim=2)          # [B,H,d_k]
#     Vsum  = V.sum(dim=2)           # [B,H,d_v]

#     A   = torch.einsum('bhnd,bhnk->bhdk', V, K)                # [B,H,d_v,d_k]
#     B_t = torch.einsum('bhnd,bhnk,bhnl->bhdkl', V, K, K)       # [B,H,d_v,d_k,d_k]
#     D   = torch.einsum('bhnk,bhnl->bhkl', K, K)                # [B,H,d_k,d_k]

#     M = A - Vsum[..., None] * kappa[..., None, :]              # [B,H,d_v,d_k]

#     KA_T = torch.einsum('bhk,bhdl->bhdkl', kappa, A)
#     AK_T = torch.einsum('bhdk,bhl->bhdkl', A, kappa)
#     KK_T = torch.einsum('bhk,bhl->bhkl', kappa, kappa)

#     U = B_t - KA_T - AK_T + Vsum[..., None, None] * KK_T.unsqueeze(2)  # [B,H,d_v,d_k,d_k]
#     Sigma = D - (N * KK_T)                                             # [B,H,d_k,d_k]

#     return {
#         "n": torch.tensor(float(N), device=device, dtype=dtype),
#         "kappa": kappa,    # [B,H,d_k]
#         "V": Vsum,         # [B,H,d_v]
#         "M": M,            # [B,H,d_v,d_k]
#         "U": U,            # [B,H,d_v,d_k,d_k]
#         "Sigma": Sigma,    # [B,H,d_k,d_k]
#     }

import torch

@torch.no_grad()
def preprocess_stats_bh(K: torch.Tensor, V: torch.Tensor):
    """
    通用版：
      K: [..., N, d_k]
      V: [..., N, d_v]
    要求：K 与 V 的前缀维度一致，且第 -2 维 N 相同
    返回：
      n:       标量(同 dtype/device)，值为 N
      kappa:   [..., d_k]
      V:       [..., d_v]
      M:       [..., d_v, d_k]
      U:       [..., d_v, d_k, d_k]
      Sigma:   [..., d_k, d_k]
    """
    assert K.dim() >= 2 and V.dim() >= 2, "K,V 应为 [..., N, d]"
    assert K.shape[:-2] == V.shape[:-2], "K,V 前缀维度必须一致"
    assert K.shape[-2] == V.shape[-2], "K,V 的第 -2 维 N 必须相同"

    N   = K.shape[-2]
    d_k = K.shape[-1]
    d_v = V.shape[-1]
    device, dtype = K.device, K.dtype

    # 逐 N 维聚合
    kappa = K.mean(dim=-2)          # [..., d_k]
    Vsum  = V.sum(dim=-2)           # [..., d_v]

    # 基本张量
    A   = torch.einsum('...nd,...nk->...dk',   V, K)                 # [..., d_v, d_k]
    B_t = torch.einsum('...nd,...nk,...nl->...dkl', V, K, K)         # [..., d_v, d_k, d_k]
    D   = torch.einsum('...nk,...nl->...kl',  K, K)                  # [..., d_k, d_k]

    M = A - Vsum[..., None] * kappa[..., None, :]                    # [..., d_v, d_k]

    # 外积项（注意使用不同的爱因斯坦指标避免与 kappa 收缩）
    KA_T = torch.einsum('...k,...dl->...dkl', kappa, A)              # [..., d_v, d_k, d_k]
    AK_T = torch.einsum('...dk,...l->...dkl', A, kappa)              # [..., d_v, d_k, d_k]
    KK_T = torch.einsum('...k,...l->...kl',   kappa, kappa)          # [..., d_k, d_k]

    U     = B_t - KA_T - AK_T + Vsum[..., None, None] * KK_T.unsqueeze(-3)  # [..., d_v, d_k, d_k]
    Sigma = D - (N * KK_T)                                                  # [..., d_k, d_k]

    return {
        "n": torch.tensor(float(N), device=device, dtype=dtype),
        "kappa": kappa,    # [..., d_k]
        "V": Vsum,         # [..., d_v]
        "M": M,            # [..., d_v, d_k]
        "U": U,            # [..., d_v, d_k, d_k]
        "Sigma": Sigma,    # [..., d_k, d_k]
    }


@torch.no_grad()
def taylor_num_estimate(q: torch.Tensor, stats: dict):
    """
    N(q) ≈ e^{q·kappa} [ V + M q + 1/2 * (U : (q⊗q)) ]
    q: [B, H, S, d_k]
    返回: [..., d_v]
    """
    # print(f"taylor_num_estimate input {q.shape=}")
    assert q.shape[-2] == 1, f"{q.shape=}"
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"

    # ✅ 修正：不减 1
    extra = q.ndim - stats["kappa"].ndim
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)      # [..., d_k]
    Vsum  = _unsqueeze_before_last(stats["V"],     extra, idx_from_end=1)      # [..., d_v]
    M     = _unsqueeze_before_last(stats["M"],     extra, idx_from_end=2)      # [..., d_v, d_k]
    U     = _unsqueeze_before_last(stats["U"],     extra, idx_from_end=3)      # [..., d_v, d_k, d_k]

    # 更稳妥的写法，避免省略号对不齐
    e = torch.exp((q * kappa).sum(dim=-1))                                     # [...]

    term0 = Vsum                                                               # [..., d_v]
    term1 = torch.einsum('...vk,...k->...v', M, q)                             # [..., d_v]
    term2 = 0.5 * torch.einsum('...vkl,...k,...l->...v', U, q, q)             # [..., d_v]
    return e[..., None] * (term0 + term1 + term2)                              # [..., d_v]


@torch.no_grad()
def taylor_den_estimate(q: torch.Tensor, stats: dict):
    """
    Z(q) ≈ e^{q·kappa} [ n + 1/2 * q^T Σ q ]
    q: [B, H, S, d_k]
    返回: [..., 1]  （方便与分子 [..., d_v] 相除）
    """
    assert q.shape[-2] == 1, f"{q.shape=}"
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"

    # ✅ 修正：不减 1
    extra = q.ndim - stats["kappa"].ndim
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)      # [..., d_k]
    Sigma = _unsqueeze_before_last(stats["Sigma"], extra, idx_from_end=2)      # [..., d_k, d_k]

    e = torch.exp((q * kappa).sum(dim=-1))                                     # [...]
    quad = torch.einsum('...kl,...k,...l->...', Sigma, q, q)                   # [...]
    den = e * (stats["n"].to(q.dtype) + 0.5 * quad)                            # [...]
    return den[..., None]      

if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, Dk, Dv = 1, 8, 128, 16, 16
    K = torch.randn(B, H, N, Dk)
    V = torch.randn(B, H, N, Dv)

    stats = preprocess_stats_bh(K, V)
    print("kappa:", stats["kappa"].shape)   # [B,H,Dk]
    print("Vsum:", stats["V"].shape)        # [B,H,Dv]
    print("M:", stats["M"].shape)           # [B,H,Dv,Dk]
    print("U:", stats["U"].shape)           # [B,H,Dv,Dk,Dk]
    print("Sigma:", stats["Sigma"].shape)   # [B,H,Dk,Dk]

    # 单个查询
    q = torch.randn(B, H, 1, Dk)
    num1 = taylor_num_estimate(q, stats)
    den1 = taylor_den_estimate(q, stats)
    print(f"{num1.shape=}, {den1.shape=}")

    from attn_utils import matmul_part_attn
    up, down = matmul_part_attn(q, K, V)
    print(f"{up.shape=}, {down.shape=}")

    attn_output1 = num1 / den1
    attn_output2 = up / down

    print(f"{attn_output1.shape=}, {attn_output2.shape=}")
    print(f"{attn_output1 - attn_output2=}")