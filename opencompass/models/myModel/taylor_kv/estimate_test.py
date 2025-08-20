import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SumExpStats:
    n: int
    d: int
    order: int
    mu: np.ndarray               # (d,)
    Sigma: Optional[np.ndarray]  # (d,d) or None
    M3: Optional[np.ndarray]     # (d,d,d) or None
    dtype: str = "float64"

def preprocess_k(
    k: np.ndarray,
    order: int = 2,
    dtype: str = "float64",
    max_m3_dim: int = 64,
) -> SumExpStats:
    """
    预处理 k_i（仅计算到指定阶）：
      order=1 -> 仅 mu
      order=2 -> mu + Sigma
      order=3 -> mu + Sigma + M3
    """
    if order not in (1, 2, 3):
        raise ValueError("order 必须在 {1,2,3} 中选择")
    k = np.asarray(k, dtype=dtype)
    if k.ndim != 2:
        raise ValueError("k 必须是 (n, d) 的二维数组")
    n, d = k.shape
    if n == 0:
        raise ValueError("k 至少包含一个样本")

    # 一阶：只算 mu
    mu = k.mean(axis=0)
    Sigma = None
    M3 = None

    # 二阶：Sigma = E[xx^T] - mu mu^T
    if order >= 2:
        ExxT = (k.T @ k) / n
        Sigma = ExxT - np.outer(mu, mu)

    # 三阶：仅当需要时才计算中心化张量
    if order == 3:
        if d > max_m3_dim:
            raise MemoryError(
                f"d={d} 超过 max_m3_dim={max_m3_dim}；为避免 O(d^3) 内存，请降低阶数或调大阈值（谨慎）。"
            )
        Xc = k - mu
        # M3 = E[(x-mu)⊗(x-mu)⊗(x-mu)]
        M3 = np.einsum('ni,nj,nk->ijk', Xc, Xc, Xc, optimize=True) / n

    return SumExpStats(n=n, d=d, order=order, mu=mu, Sigma=Sigma, M3=M3, dtype=dtype)

def estimate_sumexp(
    q: np.ndarray,
    stats: SumExpStats,
    return_log: bool = False,
    return_diag: bool = False,
) -> Dict[str, Any]:
    """
    用预处理统计量估计 S = sum_i exp(q^T k_i)。
    根据 stats.order 自动截断到 1/2/3 阶。
    """
    q = np.asarray(q, dtype=stats.dtype)
    if q.ndim != 1 or q.shape[0] != stats.d:
        raise ValueError(f"q 维度应为 ({stats.d},)")

    lin = float(q @ stats.mu)  # 一阶
    quad = 0.0
    cubic = 0.0

    if stats.order >= 2 and stats.Sigma is not None:
        quad = 0.5 * float(q @ stats.Sigma @ q)

    if stats.order >= 3 and stats.M3 is not None:
        kappa3 = float(np.einsum('ijk,i,j,k->', stats.M3, q, q, q, optimize=True))
        cubic = (1.0 / 6.0) * kappa3

    log_S_hat = np.log(stats.n) + lin + quad + cubic
    out = {
        "S_hat": None if return_log else float(np.exp(log_S_hat)),
        "log_S_hat": float(log_S_hat),
    }
    if return_diag:
        out.update({"lin": lin, "quad": quad, "cubic": cubic})
    return out

# ========================== 示例入口 ==========================
if __name__ == "__main__":
    # 1) 造一批随机的 k_i 和一个查询向量 q
    rng = np.random.default_rng(0)
    n, d = 2000, 64
    A = rng.normal(size=(d, d))
    Sigma_true = (A @ A.T) / (20 * d)    # 控制尺度，便于二/三阶近似更准
    mu_true = rng.normal(size=d)

    k = rng.multivariate_normal(mu_true, Sigma_true, size=n)  # (n,d)
    q = rng.normal(size=d) / 2.0                               # (d,)

    # 2) 真值（仅用于验证）
    exact = float(np.exp(k @ q).sum())

    # 3) 预处理 + 估计，比较 1/2/3 阶
    print("Exact =", f"{exact:.6e}")
    for ord_ in (1, 2, 3):
        stats = preprocess_k(k, order=ord_, max_m3_dim=64)
        out = estimate_sumexp(q, stats, return_log=False, return_diag=True)
        rel_err = abs(out["S_hat"] - exact) / exact
        print(
            f"order={ord_}: "
            f"S_hat={out['S_hat']:.6e}, "
            f"log_S_hat={out['log_S_hat']:.6f}, "
            f"rel_err={rel_err:.3e}, "
            f"terms(lin/quad/cubic)=({out['lin']:.4f}, {out['quad']:.4f}, {out['cubic']:.4f})"
        )
