import torch
from typing import Optional, Tuple, List, Dict

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

    def __init__(self, name: str = "unnamed", debug: bool = False):
        self.name = name
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

    def _check_and_set_shapes(self, K: torch.Tensor, V: torch.Tensor) -> None:
        assert K.dim() == 4 and V.dim() == 4, "K,V 需为 4-D: [B,H,S,D]"
        assert K.shape[:3] == V.shape[:3], "K,V 的 B/H/S 必须一致"
        B, H, S, Dk = K.shape
        _, _, _, Dv = V.shape
        if self._B is None:
            self._B, self._H, self._Dk, self._Dv = B, H, Dk, Dv
        else:
            assert (B, H, Dk, Dv) == (self._B, self._H, self._Dk, self._Dv), \
                f"后续 append 形状需与首次一致，期望(B={self._B},H={self._H},Dk={self._Dk},Dv={self._Dv})，实际(B={B},H={H},Dk={Dk},Dv={Dv})"

    @torch.no_grad()
    def append(self, K: torch.Tensor, V: torch.Tensor) -> None:
        """
        追加一段 KV。
        - 首次调用：prefill → 仅保存二阶泰勒统计量（逐 (b,h) 计算）
        - 其后调用：decode → 原始 K/V 直接沿 seq_len 维拼接
        """
        self._check_and_set_shapes(K, V)
        B, H, S, _ = K.shape

        # 只累加 seq_len
        self.cache_length += int(S)

        if self._prefill_stats is None:
            # 作为 prefill：逐 (b,h) 保存二阶泰勒统计量（输入 [S,D*]）
            # stats_grid: List[List[Dict]] = []
            # for b in range(B):
            #     row: List[Dict] = []
            #     for h in range(H):
            #         stats = preprocess_stats(K[b, h], V[b, h])  # K[b,h]: [S,Dk], V[b,h]: [S,Dv]
            #         row.append(stats)
            #     stats_grid.append(row)
            # self._prefill_stats = stats_grid
            self._prefill_stats = preprocess_stats_bh(K, V)
            self._prefill_len = S
            if self.debug:
                print(f"[{self.name}] prefill saved -> B={B}, H={H}, S={S}")
        else:
            # 作为 decode：沿 seq_len 维拼接
            if self._decode_K is None:
                self._decode_K = K.contiguous()
                self._decode_V = V.contiguous()
            else:
                self._decode_K = torch.cat([self._decode_K, K], dim=2)
                self._decode_V = torch.cat([self._decode_V, V], dim=2)
            if self.debug:
                Sd = self._decode_K.shape[2]
                print(f"[{self.name}] decode appended -> B={B}, H={H}, Sd={Sd}")

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
        return self._prefill_stats, self._prefill_len

    def get_length(self) -> int:
        """返回累计的 seq_len 之和（每次 append 只加当前段的 S）。"""
        return self.cache_length
