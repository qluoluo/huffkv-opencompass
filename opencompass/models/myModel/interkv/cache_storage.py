import os
import torch
from typing import Optional, Tuple, List, Dict, Literal, Any

from .inter_utils_v0907 import precompute_coefficients_inter, approximate_qk, approximate_qkv

class InterKVCacheStorage:
    """
    分层 KV Cache 存储管理器（批次×多头）

    约定形状：
      - K: [B, H, S, Dk]
      - V: [B, H, S, Dv]

    规则：
      - 第一次 append() 作为 prefill
      - 后续 append() 作为 decode：按 seq 维（dim=2）把原始 K/V 拼接保存
      - cache_length：每次 append 只累加 seq_len（S）
      - get_cache()：返回 (decode_K, decode_V)
      - get_states()：返回 prefill_stats
                    由上层完成具体 attention 计算
    """

    def __init__(
        self,
        name: str = "unnamed",
        inter_q_len: int = 8,
        inter_method: Literal["avg", "near", "kernel"] = "avg",
        save_full_prefill_cache: bool = False,
        debug: bool = False
    ):
        self.name = name
        self.inter_q_len = inter_q_len
        self.inter_method = inter_method
        self.save_full_prefill_cache = save_full_prefill_cache
        self.debug = debug

        # 首次 append 后锁定的形状信息
        self._B: Optional[int] = None
        self._H: Optional[int] = None
        self._Dk: Optional[int] = None
        self._Dv: Optional[int] = None

        # prefill：按 [B][H] 保存 stats，便于上层按 (b,h) 访问
        self._prefill_stats = None
        self._prefill_len: int = 0
        self._prefill_full_key_cache: Optional[torch.Tensor] = None
        self._prefill_full_value_cache: Optional[torch.Tensor] = None

        # decode：仅保存原始 K/V
        self._decode_K: Optional[torch.Tensor] = None  # [B, H, Sd, Dk]
        self._decode_V: Optional[torch.Tensor] = None  # [B, H, Sd, Dv]

        # 只累加 seq_len
        self.cache_length: int = 0

    def prefill_process(
            self, 
            Q: torch.Tensor,
            K: torch.Tensor, 
            V: torch.Tensor
        ) -> None:

        if self.debug:
            print(
                f"prefill_process_without_cluster: K.shape={K.shape}, V.shape={V.shape}"
            )

        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, "This prefill path assumes bsz == 1"

        self._prefill_len += seq_len
        q_sample = Q[..., -self.inter_q_len:, :]

        self._prefill_stats = []
        for i in range(self.inter_q_len):
            self._prefill_stats.append(
                precompute_coefficients_inter(
                    q_sample[..., i:i+1, :], K, V
                )
            )

    @torch.no_grad()
    def append(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor
    ) -> None:
        """
        追加一段 KV。
        - 首次调用：prefill → 仅保存二阶泰勒统计量（逐 (b,h) 计算）
        - 其后调用：decode → 原始 K/V 直接沿 seq_len 维拼接
        """
        # self._check_and_set_shapes(K, V)
        bsz, num_heads, seq_len, head_dim = K.shape
        assert bsz == 1, f"batch size must be 1, but got {bsz}"

        # 只累加 seq_len
        self.cache_length += int(seq_len)

        if self._prefill_len == 0:
            # 首次 append, prefill阶段
            print("Prefill Append")

            # 保存 prefill 的 full cache
            if self.save_full_prefill_cache:
                self._prefill_full_key_cache = K
                self._prefill_full_value_cache = V

            self.prefill_process(Q, K, V)

        else:
            # 作为 decode 阶段：沿 seq_len 维拼接
            print("Decode Append")

            if self._decode_K is None:
                self._decode_K = K.contiguous()
                self._decode_V = V.contiguous()
            else:
                self._decode_K = torch.cat([self._decode_K, K], dim=-2)
                self._decode_V = torch.cat([self._decode_V, V], dim=-2)
                
    @torch.no_grad()
    def get_attn_result(self, Q: torch.Tensor):
        bsz, num_heads, seq_len, head_dim = Q.shape

        assert seq_len == 1, "Only support seq_len=1"

        qkv_list, qk_list = [], []
        for i in range(len(self._prefill_stats)):
            qkv = approximate_qkv(Q, self._prefill_stats[i])
            qk = approximate_qk(Q, self._prefill_stats[i])

            print(f"{qkv.shape=} {qk.shape=}")

            qkv_list.append(qkv)
            qk_list.append(qk)

        qkv = torch.stack(qkv_list).mean(dim=0)
        qk = torch.stack(qk_list).mean(dim=0)

        return qkv, qk

    @torch.no_grad()
    def get_decode_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        返回decode部分的原始 KV
        """
        return self._decode_K, self._decode_V

    @torch.no_grad()
    def get_full_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        返回所有的cache
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
    def get_states(self):
        """
        返回 prefill 的统计量与长度
        """
        return self._prefill_stats

    def get_length(self) -> int:
        """返回累计的 seq_len 之和（每次 append 只加当前段的 S）。"""
        return self.cache_length

    def get_split_length(self) -> int:
        return self._prefill_len, self.cache_length - self._prefill_len
