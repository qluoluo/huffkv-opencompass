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
    assert L > 0 and D > 0, "L 和 D 必须 > 0"

    x2 = x.reshape(-1, L, D)  # (BS_flat, L, D)

    km = KMeans(n_clusters=k, max_iter=iters)
    res = km(x2)  # ClusterResult: labels, centers, inertia 等
    labels = res.labels.reshape(*prefix, L)
    centers = res.centers.reshape(*prefix, k, D)
    return labels, centers


# ========= 新增：按“样本”把每个类的成员组合成 list =========
@torch.no_grad()
def group_by_cluster_per_sample(
    x_sample: torch.Tensor, labels_sample: torch.Tensor, k: int
):
    """
    x_sample: (L, D) —— 单个样本
    labels_sample: (L,)
    返回: list[torch.Tensor]，长度为 K，
          第 i 个元素是该样本中属于簇 i 的所有向量，形状 (n_i, D)
    """
    assert x_sample.dim() == 2 and labels_sample.dim() == 1
    L, D = x_sample.shape
    assert labels_sample.shape == (L,)
    groups = [x_sample[labels_sample == i] for i in range(k)]
    return groups


def _reshape_list(lst, shape):
    """
    把一维列表 lst 重新“嵌套”为给定 shape 的多层列表。
    例如 shape=(B,H) 时，返回形如 [ [ ... ] * H ] * B。
    """
    if len(shape) == 0:
        # 无前缀维度时，返回单个元素（例如单样本时）
        return lst[0]
    size = shape[0]
    step = len(lst) // size
    return [
        _reshape_list(lst[i * step : (i + 1) * step], shape[1:]) for i in range(size)
    ]


@torch.no_grad()
def group_by_cluster_batched(
    x: torch.Tensor, labels: torch.Tensor, k: int, keep_prefix_shape: bool = True
):
    """
    通用化支持：x 的形状为 [..., L, D]，labels 为 [..., L] —— 任意前缀维度均可。
    参数：
      x:       [..., L, D]
      labels:  [..., L]
      k:       簇数
      keep_prefix_shape: 若为 True，则返回的结构会按前缀形状嵌套；
                         若为 False，则返回拍扁后的一维列表（长度为 ∏prefix）。
    返回：
      - 当 keep_prefix_shape=False（默认）：list[长度为 ∏prefix]，
        其中每个元素是长度为 K 的 list[Tensor(n_i, D)]；
      - 当 keep_prefix_shape=True：嵌套列表，其嵌套层级与前缀维度一致，
        每个叶子节点是长度为 K 的 list[Tensor(n_i, D)]。
    """
    assert x.dim() >= 2, "x 必须是 [..., L, D]"
    *prefix, L, D = x.shape
    assert labels.shape == (
        *prefix,
        L,
    ), f"labels 形状应为 {(*prefix, L)}, 实际为 {labels.shape}"

    # 展平成 (N, L, D) 与 (N, L)
    N = math.prod(prefix) if len(prefix) > 0 else 1
    x_flat = x.reshape(N, L, D)
    labels_flat = labels.reshape(N, L)

    groups_flat = []
    for n in range(N):
        xb = x_flat[n]  # (L, D)
        lb = labels_flat[n]  # (L,)
        groups = [xb[lb == i] for i in range(k)]  # list 长度 K
        groups_flat.append(groups)

    if keep_prefix_shape:
        return _reshape_list(groups_flat, list(prefix))
    else:
        return groups_flat


# ========= 新增：t-SNE 到二维并可视化（逐样本） =========
def tsne_plot_per_sample(
    x_sample: torch.Tensor,
    labels_sample: torch.Tensor,
    title: str = None,
    perplexity: int = 30,
    random_state: int = 0,
    #  show: bool = True,
    save_path: str = None,
):
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

    # 假设有一个 batch，形状 (bsz, seq_len, head_dim)
    bsz, seq_len, head_dim, K = 4, 128, 32, 5
    x = torch.randn(bsz, seq_len, head_dim)

    labels, centers = kmeans_seq(x, k=K, iters=50)

    # 1) 把每个样本的每个簇组合成 list
    grouped_flat = group_by_cluster_batched(x, labels, k=K, keep_prefix_shape=False)
    # grouped_flat 是长度为 bsz 的 list；其中 grouped_flat[b] 是长度为 K 的 list，
    # grouped_flat[b][i] 就是第 b 个样本里簇 i 的所有向量 (n_i, D)

    # 也可以保留前缀形状（这里就是 [bsz]）
    grouped_nested = group_by_cluster_batched(x, labels, k=K, keep_prefix_shape=True)

    # 2) 对第 0 个样本做 t-SNE 可视化
    tsne_plot_per_sample(
        x_sample=x[0],
        labels_sample=labels[0],
        title="Sample 0 - t-SNE of token embeddings (colored by cluster)",
        perplexity=30,
        random_state=42,
        # show=True,
        save_path="tsne_sample.png",
    )

    # 3) 更多前缀维度的例子：例如 (B, H, L, D)
    B, H, L, D = 2, 3, 64, 16
    x2 = torch.randn(B, H, L, D)
    labels2, centers2 = kmeans_seq(x2, k=K, iters=30)

    # a) 拍扁返回（长度为 B*H）
    grouped2_flat = group_by_cluster_batched(x2, labels2, k=K, keep_prefix_shape=False)

    # b) 保留前缀形状，得到 [B][H] 两层嵌套，每个叶子是长度为 K 的 list[Tensor(n_i, D)]
    grouped2_nested = group_by_cluster_batched(x2, labels2, k=K, keep_prefix_shape=True)

    # c) 做 t-SNE：选取一个样本（例如 b=1, h=2），切成 (L, D) 再可视化
    b, h = 1, 2
    tsne_plot_per_sample(
        x_sample=x2[b, h],
        labels_sample=labels2[b, h],
        title=f"Sample (b={b}, h={h}) - t-SNE",
        save_path="tsne_sample_b1h2.png",
    )
