import os
import torch
from typing import Optional, Tuple, List, Dict, Literal, Any

from .estimate_attn_utils import preprocess_stats_bh


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
      - get_states()：返回 prefill_stats
                    由上层完成具体 attention 计算
    """

    def __init__(
        self,
        name: str = "unnamed",
        cluster_k: int = -1,
        group_size: int = -1,
        order: int = 1,
        u_mode: Literal["full", "diag", "none"] = "full",
        debug: bool = False,
        save_full_prefill_cache: bool = False,
        kmeans_args: Dict[str, Any] = {},
    ):
        self.name = name
        self.cluster_k = cluster_k
        self.group_size = group_size
        self.order = order
        self.u_mode = u_mode
        self.debug = debug
        self.save_full_prefill_cache = save_full_prefill_cache

        # 首次 append 后锁定的形状信息
        self._B: Optional[int] = None
        self._H: Optional[int] = None
        self._Dk: Optional[int] = None
        self._Dv: Optional[int] = None

        # prefill：按 [B][H] 保存 stats，便于上层按 (b,h) 访问
        self._prefill_stats: Optional[List[List[Dict]]] = None
        self._prefill_len: int = 0  # prefill 的 seq_len（S）
        self._prefill_full_key_cache: Optional[torch.Tensor] = None
        self._prefill_full_value_cache: Optional[torch.Tensor] = None

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
            print(
                f"prefill_process_without_cluster: K.shape={K.shape}, V.shape={V.shape}"
            )

        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, "This prefill path assumes bsz == 1"

        self._prefill_len += seq_len
        self._prefill_stats = preprocess_stats_bh(
            K, V, order=self.order, u_mode=self.u_mode
        )

    def prefill_process_byfixgroup(self, K: torch.Tensor, V: torch.Tensor) -> None:
        if self.debug:
            print(f"prefill_process_byfixgroup: K.shape={K.shape}, V.shape={V.shape}")

        # assert self.group_size > 0, "group_size must > 0 in prefill_process_byfixgroup"
        assert (
            K.dim() == 4 and V.dim() == 4
        ), "K/V must be [bsz, num_heads, seq_len, head_dim]"
        assert K.shape == V.shape, "K and V must have the same shape"

        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, "This prefill path assumes bsz == 1"

        # 统计预填长度
        self._prefill_len += seq_len

        # 处理末尾不足一组的残余 token，留给 decode 阶段
        group_size = self.group_size if self.group_size > 0 else seq_len

        remain_len = seq_len % group_size
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
        groups = seq_len_work // group_size
        assert seq_len_work == groups * group_size

        # 先 reshape 出组维度：[B, H, G, S, D]
        K_work = K_work.view(bsz, num_heads, groups, group_size, head_dim)
        V_work = V_work.view(bsz, num_heads, groups, group_size, head_dim)

        # 将 组别(G) 和 batch 维 合并到 batch 上：
        # [B, H, G, S, D] -> [B, G, H, S, D] -> [B*G, H, S, D]
        K_work = (
            K_work.permute(0, 2, 1, 3, 4)
            .reshape(bsz * groups, num_heads, group_size, head_dim)
            .contiguous()
        )
        V_work = (
            V_work.permute(0, 2, 1, 3, 4)
            .reshape(bsz * groups, num_heads, group_size, head_dim)
            .contiguous()
        )

        # 下游仍按 [B', H, T, D]（这里 T=group_size）处理
        self._prefill_stats = preprocess_stats_bh(
            K_work, V_work, order=self.order, u_mode=self.u_mode
        )

        # import ipdb; ipdb.set_trace()

    def prefill_process_bycluster(self, K: torch.Tensor, V: torch.Tensor) -> None:

        if self.debug:
            print(f"prefill_process_bycluster: K.shape={K.shape}, V.shape={V.shape}")

        assert self.cluster_k > 0, "cluster_k must be > 0 in prefill_process_bycluster"

        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, "assumes bsz == 1"

        self._prefill_len += seq_len

        from .cluster_utils import (
            kmeans_seq,
            group_by_cluster_batched,
            tsne_plot_per_sample,
        )

        K_labels, K_centers = kmeans_seq(K, self.cluster_k, **self.kmeans_args)

        K_grouped = group_by_cluster_batched(
            K, K_labels, k=self.cluster_k, keep_prefix_shape=True
        )
        V_grouped = group_by_cluster_batched(
            V, K_labels, k=self.cluster_k, keep_prefix_shape=True
        )

        # 保存 t-SNE 可视化图
        # save_dir = os.path.join(os.path.dirname(__file__), "tsne_plots")
        # os.makedirs(save_dir, exist_ok=True)
        # for h in range(num_heads):
        #     tsne_plot_per_sample(
        #         K[0, h],
        #         K_labels[0, h],
        #         title=f"K (head={h})",
        #         save_path=os.path.join(save_dir, f"K_{h}.png"),
        #     )

        # 构建每个头的簇统计量
        cluster_stats = []
        for bsz_idx in range(bsz):
            bsz_stats = []
            for head_idx in range(num_heads):
                head_stats = []
                for cluster_idx in range(self.cluster_k):
                    K_cluster = K_grouped[bsz_idx][head_idx][cluster_idx]
                    V_cluster = V_grouped[bsz_idx][head_idx][cluster_idx]
                    head_stats.append(
                        preprocess_stats_bh(
                            K_cluster,
                            V_cluster,
                            order=self.order,
                            u_mode=self.u_mode,
                        )
                    )
                bsz_stats.append(head_stats)
            cluster_stats.append(bsz_stats)

        self._prefill_stats = cluster_stats

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
            # 首次 append, prefill阶段
            assert not (self.group_size > 0 and self.cluster_k > 0), f"group_size: {self.group_size}, cluster_k: {self.cluster_k}"

            # 保存 prefill 的 full cache
            if self.save_full_prefill_cache:
                self._prefill_full_key_cache = K
                self._prefill_full_value_cache = V

            # self.prefill_process(K, V)
            if self.group_size > 0:
                print("Fix Group Prefill Append")
                self.prefill_process_byfixgroup(K, V)
            elif self.cluster_k > 0:
                print("Cluster Prefill Append")
                self.prefill_process_bycluster(K, V)
            else:
                print("Normal Prefill Append")
                self.prefill_process(K, V)
        else:
            # 作为 decode 阶段：沿 seq_len 维拼接
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
    def get_full_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        返回 prefill 的 full cache
        """
        if not self.save_full_prefill_cache:
            raise ValueError("save_full_prefill_cache is False")
        
        full_key_cache_list = [x for x in [self._prefill_full_key_cache, self._decode_K] if x is not None]
        full_value_cache_list = [x for x in [self._prefill_full_value_cache, self._decode_V] if x is not None]
        
        if len(full_key_cache_list) == 0:
            return None, None

        full_key_cache = torch.cat(full_key_cache_list, dim=-2)
        full_value_cache = torch.cat(full_value_cache_list, dim=-2)

        return full_key_cache, full_value_cache

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
