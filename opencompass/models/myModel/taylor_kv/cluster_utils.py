# batched_kmeans_eval.py
# pip install torch torch-kmeans
import math
import torch
from torch_kmeans import KMeans

@torch.no_grad()
def kmeans_seq(x: torch.Tensor, k: int, iters: int = 50):
    """
    用 torch-kmeans 按 L 维做 K-Means（逐样本批处理）。
    输入:  x: [..., L, D]
    返回:  labels:  [..., L]
          centers: [..., K, D]
    """
    assert x.dim() >= 2, "x 必须是 [..., L, D]"
    *prefix, L, D = x.shape
    x2 = x.reshape(-1, L, D)                 # (BS_flat, L, D)

    km = KMeans(n_clusters=k, max_iter=iters)
    res = km(x2)                             # ClusterResult: labels, centers, inertia 等
    labels = res.labels.reshape(*prefix, L)
    centers = res.centers.reshape(*prefix, k, D)
    return labels, centers

@torch.no_grad()
def cluster_tightness(
    x: torch.Tensor,           # [..., L, D]
    labels: torch.Tensor,      # [..., L]
    centers: torch.Tensor,     # [..., K, D]
    metric: str = "euclidean", # "euclidean" 或 "cosine"
    tau: float | None = None   # 阈值；欧氏=距离阈值，余弦=1-cos 阈值
):
    """
    评估每个样本的每个簇是否“够近”。返回列表（长度=展平后的 batch 数）：
      item = {
        'per_cluster': [{'count','mean','std','p90','p95','max',('tight')}...],
        'inertia': ...,           # 欧氏=平方距离和；余弦= (1-cos) 之和
        'mean_dist': ...,         # 所有点到各自中心的平均距离
        'p95_dist': ...,          # 全局 P95
        'davies_bouldin': ...     # 越小越好；<1 常被认为不错
      }
    """
    assert x.dim() >= 2
    *prefix, L, D = x.shape
    K = centers.shape[-2]
    P = int(torch.tensor(prefix).prod()) if prefix else 1

    x2 = x.reshape(P, L, D)
    y2 = labels.reshape(P, L)
    c2 = centers.reshape(P, K, D)

    out = []
    for i in range(P):
        xi, yi, ci = x2[i], y2[i], c2[i]

        if metric == "euclidean":
            # di: 每个点到其所属簇心的欧氏距离
            dist = torch.cdist(xi, ci, p=2)                   # [L, K]
            di = dist.gather(1, yi.unsqueeze(1)).squeeze(1)   # [L]
            inertia = float((di ** 2).sum())
            centroid_d = torch.cdist(ci, ci, p=2)             # [K, K]
        elif metric == "cosine":
            xin = torch.nn.functional.normalize(xi, dim=-1)
            cin = torch.nn.functional.normalize(ci, dim=-1)
            cos = xin @ cin.T                                  # [L, K]
            di = 1.0 - cos.gather(1, yi.unsqueeze(1)).squeeze(1)
            inertia = float(di.sum())
            centroid_d = 1.0 - (cin @ cin.T)                  # [K, K]
        else:
            raise ValueError("metric 需为 'euclidean' 或 'cosine'")

        # 每簇统计
        per_cluster = []
        for j in range(K):
            m = (yi == j)
            if m.any():
                d = di[m]
                s = {
                    "count": int(m.sum()),
                    "mean": float(d.mean()),
                    "std": float(d.std(unbiased=False)),
                    "p90": float(d.quantile(0.90)),
                    "p95": float(d.quantile(0.95)),
                    "max": float(d.max()),
                }
                if tau is not None:
                    s["tight"] = bool(s["p95"] <= tau)  # 用 P95 当半径判据
            else:
                s = {"count": 0, "mean": math.nan, "std": math.nan,
                     "p90": math.nan, "p95": math.nan, "max": math.nan}
                if tau is not None:
                    s["tight"] = False
            per_cluster.append(s)

        # Davies–Bouldin 指数
        Sj = []
        for j in range(K):
            m = (yi == j)
            Sj.append(float(di[m].mean()) if m.any() else math.nan)
        Sj = torch.tensor(Sj, device=xi.device, dtype=xi.dtype)
        M = centroid_d.clone()
        M.fill_diagonal_(float("inf"))
        Sj_mat = Sj.unsqueeze(0) + Sj.unsqueeze(1)            # [K, K]
        R = Sj_mat / M
        R[torch.isnan(R)] = 0.0
        dbi = float(R.max(dim=1).values.nanmean())

        out.append({
            "per_cluster": per_cluster,
            "inertia": inertia,
            "mean_dist": float(di.mean()),
            "p95_dist": float(di.quantile(0.95)),
            "davies_bouldin": dbi,
        })
    return out

def _pretty_print(report_item, prefix="sample"):
    pcs = report_item["per_cluster"]
    header = f"{prefix}: inertia={report_item['inertia']:.4f}, mean_dist={report_item['mean_dist']:.4f}, " \
             f"p95_dist={report_item['p95_dist']:.4f}, DBI={report_item['davies_bouldin']:.4f}"
    print(header)
    print("  cluster | count |   mean   |    std   |   p90   |   p95   |   max   | tight")
    for j, s in enumerate(pcs):
        tight = s.get("tight", None)
        tight_str = ("YES" if tight else "NO") if tight is not None else "-"
        print(f"     {j:>3} | {s['count']:>5} | {s['mean']:>7.4f} | {s['std']:>7.4f} | "
              f"{s['p90']:>7.4f} | {s['p95']:>7.4f} | {s['max']:>7.4f} | {tight_str}")

if __name__ == "__main__":
    # 最小演示
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(2, 3, 128, 64, device=device)   # [B=2, H=3, L=128, D=64]
    k = 8

    labels, centers = kmeans_seq(x, k=k, iters=30)
    print("labels:", tuple(labels.shape))    # -> (2, 3, 128)
    print("centers:", tuple(centers.shape))  # -> (2, 3, 8, 64)

    # 评估（欧氏距离）；比如要求 P95 距离 <= 0.8 视为“够近”
    report = cluster_tightness(x, labels, centers, metric="euclidean")

    # 打印第一条样本（B=0,H=0）
    _pretty_print(report[0], prefix="sample[0]")
