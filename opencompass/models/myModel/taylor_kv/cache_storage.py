import os
import torch
from typing import Optional, Tuple, List, Dict, Literal

from .estimate_attn_utils import (
    preprocess_stats_bh,
    # taylor_den,  # 由上层负责具体 attn 计算，这里不再使用
    # taylor_num
)

class RemainKVCacheStorage:
    """
    分层 KV Cache 存储管理器（批次×多头）

    约定形状：
      - K: [B, H, S, Dk]
      - V: [B, H, S, Dv]

    规则：
      - 第一次 append() 作为 prefill：仅保存二阶泰勒统计量（不保留原始 prefill K/V）
      - 后续 append() 作为 decode：按 seq 维（dim=2）把原始 K/V 拼接保存
      - cache_length：每次 append 只累加 seq_len（S）
      - get_cache()：返回 (decode_K, decode_V)
      - get_states()：返回 (prefill_stats, prefill_len)
                    由上层完成具体 attention 计算
    """

    def __init__(self, 
                 name: str = "unnamed", 
                 cluster_k: int = 0,
                 group_size: int = 0,
                 order: int = 1,
                 u_mode: Literal["full","diag","none"]="full",
                 debug: bool = False
                ):
        self.name = name
        self.cluster_k = cluster_k
        self.group_size = group_size
        self.debug = debug

        # 首次 append 后锁定的形状信息
        self._B: Optional[int] = None
        self._H: Optional[int] = None
        self._Dk: Optional[int] = None
        self._Dv: Optional[int] = None

        # prefill：按 [B][H] 保存 stats，便于上层按 (b,h) 访问
        self._prefill_stats: Optional[List[List[Dict]]] = None
        self._prefill_len: int = 0  # prefill 的 seq_len（S）

        # decode：仅保存原始 K/V
        self._decode_K: Optional[torch.Tensor] = None  # [B, H, Sd, Dk]
        self._decode_V: Optional[torch.Tensor] = None  # [B, H, Sd, Dv]

        # 只累加 seq_len
        self.cache_length: int = 0

    # def _check_and_set_shapes(self, K: torch.Tensor, V: torch.Tensor) -> None:
    #     assert K.dim() == 4 and V.dim() == 4, "K,V 需为 4-D: [B,H,S,D]"
    #     assert K.shape[:3] == V.shape[:3], "K,V 的 B/H/S 必须一致"
    #     B, H, S, Dk = K.shape
    #     _, _, _, Dv = V.shape
    #     if self._B is None:
    #         self._B, self._H, self._Dk, self._Dv = B, H, Dk, Dv
    #     else:
    #         assert (B, H, Dk, Dv) == (self._B, self._H, self._Dk, self._Dv), \
    #             f"后续 append 形状需与首次一致，期望(B={self._B},H={self._H},Dk={self._Dk},Dv={self._Dv})，实际(B={B},H={H},Dk={Dk},Dv={Dv})"

    def prefill_process(self, K: torch.Tensor, V: torch.Tensor) -> None:

        if self.debug:
            print(f"prefill_process_without_cluster: K.shape={K.shape}, V.shape={V.shape}")

        num_heads, seq_len, head_dim = K.shape
        self._prefill_len += seq_len
        self._prefill_stats = preprocess_stats_bh(K, V)

    def prefill_process_byfixgroup(self, K: torch.Tensor, V: torch.Tensor) -> None:
        if self.debug:
            print(f"prefill_process_byfixgroup: K.shape={K.shape}, V.shape={V.shape}")

        assert self.group_size > 0, "group_size must > 0 in prefill_process_byfixgroup"
        assert K.dim() == 4 and V.dim() == 4, "K/V must be [bsz, num_heads, seq_len, head_dim]"
        assert K.shape == V.shape, "K and V must have the same shape"

        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, "This prefill path assumes bsz == 1"

        # 统计预填长度
        self._prefill_len += seq_len

        # 处理末尾不足一组的残余 token，留给 decode 阶段
        remain_len = seq_len % self.group_size
        if remain_len > 0:
            # 注意包含 heads 维度
            self._decode_K = K[:, :, -remain_len:, :].contiguous()
            self._decode_V = V[:, :, -remain_len:, :].contiguous()
            K_work = K[:, :, :-remain_len, :]
            V_work = V[:, :, :-remain_len, :]
        else:
            # 没有残余则清空 decode 暂存（避免陈旧值）
            self._decode_K = None
            self._decode_V = None
            K_work, V_work = K, V

        # 可能全部都是残余（seq_len < group_size），此时无需预填
        if K_work.numel() == 0:
            self._prefill_stats = None
            return

        # 现在长度可被整除：seq_len_work = groups * group_size
        seq_len_work = K_work.shape[2]
        groups = seq_len_work // self.group_size
        assert seq_len_work == groups * self.group_size

        # 先 reshape 出组维度：[B, H, G, S, D]
        K_work = K_work.view(bsz, num_heads, groups, self.group_size, head_dim)
        V_work = V_work.view(bsz, num_heads, groups, self.group_size, head_dim)

        # 将 组别(G) 和 batch 维 合并到 batch 上：
        # [B, H, G, S, D] -> [B, G, H, S, D] -> [B*G, H, S, D]
        K_work = K_work.permute(0, 2, 1, 3, 4).reshape(bsz * groups, num_heads, self.group_size, head_dim).contiguous()
        V_work = V_work.permute(0, 2, 1, 3, 4).reshape(bsz * groups, num_heads, self.group_size, head_dim).contiguous()

        # 下游仍按 [B', H, T, D]（这里 T=group_size）处理
        self._prefill_stats = preprocess_stats_bh(K_work, V_work)

        # import ipdb; ipdb.set_trace()
        

    def prefill_process_bycluster(self, K: torch.Tensor, V: torch.Tensor) -> None:

        if self.debug:
            print(f"prefill_process_bycluster: K.shape={K.shape}, V.shape={V.shape}")

        assert self.cluster_k > 0, "cluster_k must be > 0 in prefill_process_bycluster"

        num_heads, seq_len, head_dim = K.shape
        self._prefill_len += seq_len

        from .cluster_utils import kmeans_seq, group_by_cluster_batched, tsne_plot_per_sample
        K_labels, K_centers = kmeans_seq(K, self.cluster_k, iters=50)

        K_grouped = group_by_cluster_batched(K, K_labels, k=self.cluster_k)
        V_grouped = group_by_cluster_batched(V, K_labels, k=self.cluster_k)

        for i in range(num_heads):
            save_dir = os.path.join(os.path.dirname(__file__), "tsne_plots")
            os.makedirs(save_dir, exist_ok=True)
            tsne_plot_per_sample(K[i], K_labels[i], title="K", save_path=os.path.join(save_dir, f"K_{i}.png"))

        group_stats = []
        for group_idx in range(len(K_grouped)):
            # 下面实际上是每个头的list
            K_idx = K_grouped[group_idx]
            V_idx = V_grouped[group_idx]

            append_data = []
            for cluster_idx in range(len(K_idx)):
                append_data.append(preprocess_stats_bh(K_idx[cluster_idx], V_idx[cluster_idx]))


        self._prefill_stats = group_stats

    @torch.no_grad()
    def append(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        追加一段 KV。
        - 首次调用：prefill → 仅保存二阶泰勒统计量（逐 (b,h) 计算）
        - 其后调用：decode → 原始 K/V 直接沿 seq_len 维拼接
        """
        # self._check_and_set_shapes(K, V)
        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, f"batch size must be 1, but got {bsz}"

        # K = K.squeeze(0)
        # V = V.squeeze(0)

        # 只累加 seq_len
        self.cache_length += int(seq_len)

        if self._prefill_len == 0:
            # self.prefill_process(K, V)
            self.prefill_process_byfixgroup(K, V)
        else:
            # 作为 decode：沿 seq_len 维拼接
            if self._decode_K is None:
                self._decode_K = K.contiguous()
                self._decode_V = V.contiguous()
            else:
                self._decode_K = torch.cat([self._decode_K, K], dim=-2)
                self._decode_V = torch.cat([self._decode_V, V], dim=-2)

    @torch.no_grad()
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        返回 decode 部分的原始 KV：
          - 若没有 decode，则为 (None, None)
          - 形状：decode_K ∈ [B, H, Sd, Dk], decode_V ∈ [B, H, Sd, Dv]
        """
        return self._decode_K, self._decode_V

    @torch.no_grad()
    def get_states(self) -> Tuple[Optional[List[List[Dict]]], int]:
        """
        返回 prefill 的统计量与长度
        """
        return self._prefill_stats

    def get_length(self) -> int:
        """返回累计的 seq_len 之和（每次 append 只加当前段的 S）。"""
        return self.cache_length

    def get_split_length(self) -> int:
        return self._prefill_len, self.cache_length - self._prefill_len