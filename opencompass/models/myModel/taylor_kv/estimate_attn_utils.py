import torch

def _unsqueeze_before_last(x: torch.Tensor, n: int, idx_from_end: int = 1):
    """
    在张量末尾第 idx_from_end 个维度之前插入 n 个维度。
    例如：x[..., Dk] -> idx_from_end=1；x[..., Dv, Dk] -> idx_from_end=2。
    """
    for _ in range(n):
        x = x.unsqueeze(-idx_from_end-0)
    return x

@torch.no_grad()
def preprocess_stats_bh(K: torch.Tensor, V: torch.Tensor):
    """
    预处理中心化统计量（一次性），供二阶泰勒近似使用。
    K: [B, H, N, d_k], V: [B, H, N, d_v]
    返回:
      {
        "n": 标量 float(N),
        "kappa": [B,H,d_k],
        "V":     [B,H,d_v],
        "M":     [B,H,d_v,d_k],
        "U":     [B,H,d_v,d_k,d_k],
        "Sigma": [B,H,d_k,d_k]
      }
    """
    assert K.dim() == 4 and V.dim() == 4, "K,V 应为 [B, H, N, d]"
    B, H, N, d_k = K.shape
    d_v = V.shape[-1]
    device = K.device

    # 均值、求和（沿 seq_len=N）
    kappa = K.mean(dim=2)                 # [B,H,d_k]
    Vsum  = V.sum(dim=2)                  # [B,H,d_v]

    # 原始矩（按 batch/head 分开求）
    # A = sum_i v_i ⊗ k_i     -> [B,H,d_v,d_k]
    A = torch.einsum('bhnd,bhnk->bhdk', V, K)

    # B = sum_i v_i ⊗ (k_i k_i^T) -> [B,H,d_v,d_k,d_k]
    B_t = torch.einsum('bhnd,bhnk,bhnl->bhdkl', V, K, K)

    # D = sum_i k_i k_i^T -> [B,H,d_k,d_k]
    D = torch.einsum('bhnk,bhnl->bhkl', K, K)

    # 中心化
    # M = A - Vsum[:, :, :, None] * kappa[:, :, None, :]
    M = A - Vsum[..., :, None] * kappa[..., None, :]

    # U_j = B_j - kappa A_j^T - A_j kappa^T + V_j (kappa kappa^T)
    k_row = kappa[..., None, :]             # [B,H,1,d_k]
    k_col = kappa[..., :, None]             # [B,H,d_k,1]
    A_row = A[..., None, :]                 # [B,H,d_v,1,d_k]
    A_col = A[..., :, None]                 # [B,H,d_v,d_k,1]

    U = B_t - (A_col * k_row) - (k_col * A_row) \
        + Vsum[..., None, None] * (k_col * k_row)  # [B,H,d_v,d_k,d_k]

    Sigma = D - (N * torch.einsum('bhk,bhl->bhkl', kappa, kappa))  # [B,H,d_k,d_k]

    return {
        "n": torch.tensor(float(N), device=device),
        "kappa": kappa,   # [B,H,d_k]
        "V": Vsum,        # [B,H,d_v]
        "M": M,           # [B,H,d_v,d_k]
        "U": U,           # [B,H,d_v,d_k,d_k]
        "Sigma": Sigma    # [B,H,d_k,d_k]
    }

@torch.no_grad()
def taylor_num_estimate(q: torch.Tensor, stats: dict):
    """
    二阶泰勒近似的分子:
      N(q) = sum_i e^{q·k_i} v_i  ≈ e^{q·kappa} [ V + M q + 1/2 * (U : (q⊗q)) ]
    q: [..., d_k]，其前导维需与 stats 的 [B,H] 对齐，或包含相同的 [B,H] 并可再带任意额外维度
    返回: [..., d_v]
    """
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"
    # 需要在 stats 张量上为 q 的“额外查询维”插入轴，以便广播
    extra = q.ndim - 1 - stats["kappa"].ndim  # 去掉 q 的最后一维 d_k
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)      # [..., d_k]
    Vsum  = _unsqueeze_before_last(stats["V"],     extra, idx_from_end=1)      # [..., d_v]
    M     = _unsqueeze_before_last(stats["M"],     extra, idx_from_end=2)      # [..., d_v, d_k]
    U     = _unsqueeze_before_last(stats["U"],     extra, idx_from_end=3)      # [..., d_v, d_k, d_k]

    # e^{q·kappa}
    e = torch.exp(torch.einsum('...k,...k->...', q, kappa))                    # [...]

    term0 = Vsum                                                                 # [..., d_v]
    term1 = torch.einsum('...vk,...k->...v', M, q)                               # [..., d_v]
    term2 = 0.5 * torch.einsum('...vkl,...k,...l->...v', U, q, q)               # [..., d_v]
    return e[..., None] * (term0 + term1 + term2)                                # [..., d_v]

@torch.no_grad()
def taylor_den_estimate(q: torch.Tensor, stats: dict):
    """
    二阶泰勒近似的分母:
      Z(q) = sum_i e^{q·k_i} ≈ e^{q·kappa} [ n + 1/2 * q^T Σ q ]
    q: [..., d_k] （同上）
    返回: [...], 与 q 的前导维一致
    """
    assert q.shape[-1] == stats["kappa"].shape[-1], "q 的最后一维应等于 d_k"

    extra = q.ndim - 1 - stats["kappa"].ndim
    if extra < 0:
        raise ValueError("q 的维度应至少与 [B,H,d_k] 等长")

    kappa = _unsqueeze_before_last(stats["kappa"], extra, idx_from_end=1)       # [..., d_k]
    Sigma = _unsqueeze_before_last(stats["Sigma"], extra, idx_from_end=2)       # [..., d_k, d_k]

    e = torch.exp(torch.einsum('...k,...k->...', q, kappa))                     # [...]
    quad = torch.einsum('...kl,...k,...l->...', Sigma, q, q)                    # [...]
    return e * (stats["n"] + 0.5 * quad)                                        # [...]

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
    q1 = torch.randn(B, H, 1, Dk)
    num1 = taylor_num_estimate(q1, stats)
    den1 = taylor_den_estimate(q1, stats)
    print("num1:", num1.shape)  # [B,H,Dv]
    print("den1:", den1.shape)  # [B,H]

    from .attn_utils import matmul_part_attn
    up, down = matmul_part_attn(q1, K, V)