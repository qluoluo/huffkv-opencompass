import torch
from typing import Literal

def _unsqueeze_before_last(x: torch.Tensor, n: int, idx_from_end: int = 1):
    for _ in range(n):
        x = x.unsqueeze(-idx_from_end-1)
    return x

@torch.no_grad()
def preprocess_stats_bh(
    K: torch.Tensor, 
    V: torch.Tensor, 
    order: Literal[1, 2] = 1,
    u_mode: Literal["full","diag","none"]="diag",
):
    """
    通用版 & 可选阶数：
      K: [..., N, d_k]
      V: [..., N, d_v]
    要求：K 与 V 的前缀维度一致，且第 -2 维 N 相同
    返回（按需计算）：
      n:       标量(同 dtype/device)，值为 N
      kappa:   [..., d_k]                    —— 所有阶都需要
      V:       [..., d_v]                    —— 所有阶都需要（∑V_i）
      M:       [..., d_v, d_k]               —— 一阶/二阶都需要
      U:       [..., d_v, d_k, d_k] 或 None  —— 仅二阶需要
      Sigma:   [..., d_k, d_k] 或 None       —— 仅二阶需要
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
    # A = ∑ V_i k_i^T
    A = torch.einsum('...nd,...nk->...dk', V, K)   # [..., d_v, d_k]
    M = A - Vsum[..., None] * kappa[..., None, :]
    U = Sigma = None

    if order == 2:
        # 总是算 Sigma（分母用得到），它只有 d_k^2
        D     = torch.einsum('...nk,...nl->...kl', K, K)
        KK_T  = torch.einsum('...k,...l->...kl', kappa, kappa)
        Sigma = D - (N * KK_T)

        if u_mode == "full":
            B_t  = torch.einsum('...nd,...nk,...nl->...dkl', V, K, K)
            KA_T = torch.einsum('...k,...dl->...dkl', kappa, A)
            AK_T = torch.einsum('...dk,...l->...dkl', A, kappa)
            U = B_t - KA_T - AK_T + Vsum[..., None, None]*KK_T.unsqueeze(-3)

        elif u_mode == "diag":
            # 仅保留二阶的对角向量，极大省内存
            Ssq = torch.einsum('...nd,...nk->...dk', V, K*K)  # [..., d_v, d_k]
        # "none" 什么也不存（配合上面的“用KV精确”或“共享近似”）
    out = {"n": torch.tensor(float(N), device=device, dtype=dtype),
           "kappa": kappa, "V": Vsum, "M": M,
           "U": U, "Sigma": Sigma, "order": order}
    if order==2 and u_mode=="diag":
        out["Ssq"] = Ssq
    return out


@torch.no_grad()
def taylor_num_estimate(q: torch.Tensor, stats: dict):
    """
    分子 N(q) 估计：
      一阶：N(q) ≈ e^{q·kappa} [ V + M q ]
      二阶：N(q) ≈ e^{q·kappa} [ V + M q + 1/2 * (U : (q⊗q)) ]
    q: [B, H, S, d_k]，当前实现要求 B=1, S=1
    返回: [..., d_v]
    """
    # assert q.shape[0] == 1, f"{q.shape=}"
    # q = q.squeeze(0)
    assert q.shape[-2] == 1, f"{q.shape=}"
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"

    # 维度对齐
    extra = q.ndim - stats["kappa"].ndim
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    order = stats['order']

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)      # [..., d_k]
    Vsum  = _unsqueeze_before_last(stats["V"],     extra, idx_from_end=1)      # [..., d_v]
    M     = _unsqueeze_before_last(stats["M"],     extra, idx_from_end=2)      # [..., d_v, d_k]

    e = torch.exp((q * kappa).sum(dim=-1))                                     # [...]

    term0 = Vsum                                                               # [..., d_v]
    term1 = torch.einsum('...vk,...k->...v', M, q)                             # [..., d_v]

    if order == 1:
        return e[..., None] * (term0 + term1)

    # 二阶
    assert stats["U"] is not None, "需要在 preprocess_stats_bh(..., order=2) 下预计算 U"
    U = _unsqueeze_before_last(stats["U"], extra, idx_from_end=3)              # [..., d_v, d_k, d_k]
    term2 = 0.5 * torch.einsum('...vkl,...k,...l->...v', U, q, q)             # [..., d_v]
    return e[..., None] * (term0 + term1 + term2)


@torch.no_grad()
def taylor_den_estimate(q: torch.Tensor, stats: dict):
    """
    分母 Z(q) 估计：
      一阶：Z(q) ≈ e^{q·kappa} [ n ]
      二阶：Z(q) ≈ e^{q·kappa} [ n + 1/2 * q^T Σ q ]
    q: [B, H, S, d_k]，当前实现要求 B=1, S=1
    返回: [..., 1]  （方便与分子 [..., d_v] 相除）
    """
    # assert q.shape[0] == 1, f"{q.shape=}"
    # q = q.squeeze(0)
    assert q.shape[-2] == 1, f"{q.shape=}"
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"

    extra = q.ndim - stats["kappa"].ndim
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    order = stats['order']

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)      # [..., d_k]
    e = torch.exp((q * kappa).sum(dim=-1))                                     # [...]

    if order == 1:
        den = e * stats["n"].to(q.dtype)                                       # [...]
        return den[..., None]

    # 二阶
    assert stats["Sigma"] is not None, "需要在 preprocess_stats_bh(..., order=2) 下预计算 Sigma"
    Sigma = _unsqueeze_before_last(stats["Sigma"], extra, idx_from_end=2)      # [..., d_k, d_k]
    quad = torch.einsum('...kl,...k,...l->...', Sigma, q, q)                   # [...]
    den = e * (stats["n"].to(q.dtype) + 0.5 * quad)                            # [...]
    return den[..., None]


# @torch.no_grad()
# def taylor_attn_estimate(q: torch.Tensor, stats: dict):
#     """
#     直接给出注意力输出的泰勒估计： N(q) / Z(q)
#     """
#     num = taylor_num_estimate(q, stats)
#     den = taylor_den_estimate(q, stats)
#     return num / den


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, Dk, Dv = 1, 8, 128, 16, 16
    K = torch.randn(B, H, N, Dk)
    V = torch.randn(B, H, N, Dv)

    # 选择展开阶数：1 或 2
    order = 2 # 改成 1 即用一阶
    stats = preprocess_stats_bh(K, V, order=order)

    print("kappa:", stats["kappa"].shape)           # [B,H,Dk]
    print("Vsum:", stats["V"].shape)                # [B,H,Dv]
    print("M:", stats["M"].shape)                   # [B,H,Dv,Dk]
    if order == 2:
        print("U:", stats["U"].shape)               # [B,H,Dv,Dk,Dk]
        print("Sigma:", stats["Sigma"].shape)       # [B,H,Dk,Dk]

    # 单个查询
    q = torch.randn(B, H, 1, Dk)
    num = taylor_num_estimate(q, stats)
    den = taylor_den_estimate(q, stats)
    print(f"{num.shape=}, {den.shape=}")

    # 与精确（部分）注意力对比
    from attn_utils import matmul_part_attn
    up, down = matmul_part_attn(q, K, V)
    attn_output_taylor = num / den
    attn_output_exact  = up / down

    print(f"{attn_output_taylor.shape=}, {attn_output_exact.shape=}")
    print(f"误差范数: {(attn_output_taylor - attn_output_exact).norm()}")
    print(f"{attn_output_taylor - attn_output_exact=}")
