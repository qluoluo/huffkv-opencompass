# batched_kmeans_eval.py
# pip install torch torch-kmeans scikit-learn matplotlib
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


# ========= 新增：按“样本”把每个类的成员组合成 list =========
@torch.no_grad()
def group_by_cluster_per_sample(x_sample: torch.Tensor,
                                labels_sample: torch.Tensor,
                                k: int):
    """
    x_sample: (L, D) —— 单个样本
    labels_sample: (L,)
    返回: list[torch.Tensor]，长度为 K，
          第 i 个元素是该样本中属于簇 i 的所有向量，形状 (n_i, D)
    """
    assert x_sample.dim() == 2 and labels_sample.dim() == 1
    groups = [x_sample[labels_sample == i] for i in range(k)]
    return groups


@torch.no_grad()
def group_by_cluster_batched(x: torch.Tensor,
                             labels: torch.Tensor,
                             k: int):
    """
    仅支持三维输入：
      x: (B, L, D)
      labels: (B, L)
    返回：
      list[长度为 B]，其中每个元素是长度为 K 的 list[Tensor(n_i, D)]
    """
    assert x.dim() == 3, "x 必须是三维 (B, L, D)"
    assert labels.dim() == 2, "labels 必须是二维 (B, L)"
    B, L, D = x.shape
    assert labels.shape == (B, L), "labels 形状应为 (B, L)"

    groups_per_batch = []
    for b in range(B):
        xb = x[b]           # (L, D)
        lb = labels[b]      # (L,)
        groups = [xb[lb == i] for i in range(k)]
        groups_per_batch.append(groups)
    return groups_per_batch


# ========= 新增：t-SNE 到二维并可视化（逐样本） =========
def tsne_plot_per_sample(x_sample: torch.Tensor,
                         labels_sample: torch.Tensor,
                         title: str = None,
                         perplexity: int = 30,
                         random_state: int = 0,
                        #  show: bool = True,
                         save_path: str = None):
    """
    对单个样本做 t-SNE 可视化。
    x_sample: (L, D)
    labels_sample: (L,)
    """
    import numpy as np
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    x_np = x_sample.detach().cpu().numpy()
    y_np = labels_sample.detach().cpu().numpy()

    # t-SNE 到二维
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        learning_rate="auto",
        random_state=random_state,
    )
    xy = tsne.fit_transform(x_np)  # (L, 2)

    # 画散点：用 cluster label 作为颜色
    plt.figure(figsize=(6, 5), dpi=140)
    plt.scatter(xy[:, 0], xy[:, 1], c=y_np, s=16, alpha=0.9)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if title:
        plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    # if show:
    #     plt.show()
    plt.close()


# ========= 使用示例 =========
if __name__ == "__main__":
    torch.manual_seed(0)
    # 假设有一个 batch，形状 (B, L, D)
    B, L, D, K = 4, 128, 32, 5
    x = torch.randn(B, L, D)

    labels, centers = kmeans_seq(x, k=K, iters=50)

    # 1) 把每个样本的每个簇组合成 list
    grouped = group_by_cluster_batched(x, labels, k=K)
    # grouped 是长度为 B 的 list；其中 grouped[b] 是长度为 K 的 list，
    # grouped[b][i] 就是第 b 个样本里簇 i 的所有向量 (n_i, D)

    # 2) 对第 0 个样本做 t-SNE 可视化
    tsne_plot_per_sample(
        x_sample=x[0],
        labels_sample=labels[0],
        title="Sample 0 - t-SNE of token embeddings (colored by cluster)",
        perplexity=30,
        random_state=42,
        # show=True,
        save_path="tsne_sample.png",  # 想保存就填路径，例如 "tsne_sample0.png"
    )

    # 如果你有更多前缀维度（例如 (B, H, L, D)），
    # 先把其中一个样本切出来成 (L, D) 再调用 tsne_plot_per_sample 即可。
