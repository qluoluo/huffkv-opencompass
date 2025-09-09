from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv

from .cache_storage import InterKVCacheStorage

class InterKVCache(DynamicCache):
    """
    Custom cache implementation with three-tier storage based on attention importance:
    - Sparse: Selected sparse KV pairs based on attention scores
    - Compressed: Quantized less important KV pairs
    - Window: Recent KV pairs in full precision
    """

    InterKVCache_init = False

    def __init__(self, config) -> None:
        super().__init__()
        

        # Cache configuration
        self.kvcache_settings = config.kvcache_settings
        self.window_size = self.kvcache_settings["window_size"]
        self.sparse_num = self.kvcache_settings["sparse_num"]
        self.use_remain = self.kvcache_settings["use_remain"]

        self.debug = self.kvcache_settings.get("debug", False)
        if type(self.debug) is str:
            print(f"pass {type(self.debug)=} {self.debug=}")
            self.debug = self.debug.lower() == "true"
        # self.debug = True
        # self.debug = False

        self.key_cache = []
        self.value_cache = []

        if self.use_remain:
            if self.debug:
                print("Cache use Remain")
            self.remain_cache: List[InterKVCacheStorage] = []
            self.remain_cache_kwargs = self.kvcache_settings['remain_settings']
        else:
            if self.debug:
                print("Cache NOT use Remain")

        self._seen_tokens = 0
        # self._prefill_computed = False

        # self.remain_cluster_k = self.kvcache_settings["remain_cluster_k"]
        # self.remain_group_size = self.kvcache_settings["remain_group_size"]


        # Minimum sequence length required for meaningful importance selection
        self.min_seq_length = self.sparse_num + self.window_size

        if not InterKVCache.InterKVCache_init:
            InterKVCache.InterKVCache_init = True
            print(f"-------------------- {self.__class__.__name__} init --------------------")
            print(f"{self.debug=}")

    def _initialize_layer_cache(self, layer_idx: int):
        """Initialize cache dictionaries for a new layer if not present."""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(
                {
                    "sparse": None,
                    "window": None,
                }
            )
            self.value_cache.append(
                {
                    "sparse": None,
                    "window": None,
                }
            )
            if self.use_remain:
                self.remain_cache.append(
                    InterKVCacheStorage(
                        name=f"remain_cache_{layer_idx}",
                        debug=self.debug if layer_idx==0 else False,
                        **self.remain_cache_kwargs,
                    )
                )

    def _select_sparse_kv(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select sparse and less important KV pairs based on attention scores.

        Returns:
            sparse_keys, sparse_values, remain_keys, remain_values
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        sparse_num = self.kvcache_settings["sparse_num"]
        pool_kernel_size = self.kvcache_settings.get("pool_kernel_size", -1)
        window_size = self.kvcache_settings["window_size"]

        if sparse_num == 0:
            return None, None, key_states, value_states

        q_window_mean = query_states[..., -window_size:, :].mean(dim=-2, keepdim=True)

        # import ipdb; ipdb.set_trace()

        scores = torch.matmul(q_window_mean, key_states.transpose(-2, -1)).squeeze(2)

        if pool_kernel_size > 0:
            pool = torch.nn.AvgPool1d(
                kernel_size=pool_kernel_size, padding=pool_kernel_size // 2, stride=1
            )
            scores = pool(scores)

        actual_sparse_num = min(sparse_num, seq_len)

        _, topk_indices = torch.topk(scores, k=actual_sparse_num, dim=-1)
        topk_indices, _ = torch.sort(topk_indices, dim=-1)
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        sparse_k = torch.gather(key_states, dim=-2, index=indices_expanded)
        sparse_v = torch.gather(value_states, dim=-2, index=indices_expanded)

        # Generate mask for remaining indices
        mask = torch.zeros(
            (bsz, num_heads, seq_len), dtype=torch.bool, device=key_states.device
        )
        mask.scatter_(-1, topk_indices, True)

        # Get remaining indices
        remain_indices = torch.arange(seq_len, device=key_states.device).expand(
            bsz, num_heads, seq_len
        )
        remain_indices = remain_indices.masked_select(~mask).view(bsz, num_heads, -1)
        remain_indices_expanded = remain_indices.unsqueeze(-1).expand(
            -1, -1, -1, head_dim
        )

        remain_k = torch.gather(key_states, dim=-2, index=remain_indices_expanded)
        remain_v = torch.gather(value_states, dim=-2, index=remain_indices_expanded)

        return sparse_k, sparse_v, remain_k, remain_v

    def _setup_sparse_based_cache(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """
        Setup sparse-based cache partitioning (only done once).
        """
        seq_len = key_states.shape[-2]
        window_size = self.window_size

        self.key_cache[layer_idx]['window'] = key_states[..., -window_size:, :]
        self.value_cache[layer_idx]['window'] = value_states[..., -window_size:, :]

        # Select sparse and remain KV pairs
        sparse_key, sparse_value, remain_key, remain_value = self._select_sparse_kv(
            query_states[..., -window_size:, :],
            key_states[..., :-window_size, :],
            value_states[..., :-window_size, :],
        )
        self.key_cache[layer_idx]['sparse'] = sparse_key
        self.value_cache[layer_idx]['sparse'] = sparse_value

        if self.use_remain:
            self.remain_cache[layer_idx].append(query_states, remain_key, remain_value)

    def _update_window_cache(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        """Update window cache with new tokens, maintaining window size."""
        window_size = self.window_size

        if self.key_cache[layer_idx]["window"] is not None:
            # Concatenate new states
            new_window_keys = torch.cat(
                [self.key_cache[layer_idx]["window"], key_states], dim=-2
            )
            new_window_values = torch.cat(
                [self.value_cache[layer_idx]["window"], value_states], dim=-2
            )
        else:
            new_window_keys = key_states
            new_window_values = value_states

        self.key_cache[layer_idx]["window"] = new_window_keys[..., -window_size:, :]
        self.value_cache[layer_idx]["window"] = new_window_values[..., -window_size:, :]

        # Keep only the last window_size tokens
        if self.use_remain and new_window_keys.shape[-2] > self.window_size:
            self.remain_cache[layer_idx].append(
                query_states,
                new_window_keys[..., : -self.window_size, :],
                new_window_values[..., : -self.window_size, :]
            )
            # self.remain_cache[layer_idx].append(
            #     new_window_values[..., : -self.window_size, :]
            # )

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and reconstruct full key-value cache for a layer."""
        key_cache = self._reconstruct_cache(layer_idx, is_key=True)
        value_cache = self._reconstruct_cache(layer_idx, is_key=False)

        # remain_states = None
        # if self.use_remain and len(self.remain_cache) > layer_idx:
        #     remain_states = self.remain_cache[layer_idx].get_states()

        return key_cache, value_cache

    def get_remain_attn_result(self, layer_idx: int, query_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.remain_cache[layer_idx].get_attn_result(query_states)

    def _reconstruct_cache(self, layer_idx: int, is_key: bool) -> torch.Tensor:
        """Reconstruct full cache tensor from sparse and window parts."""
        if len(self.key_cache) <= layer_idx:
            return None

        cache_dict = (
            self.key_cache[layer_idx] if is_key else self.value_cache[layer_idx]
        )

        # Get cache components
        sparse_cache = cache_dict.get("sparse")
        window_cache = cache_dict.get("window")

        remain_cache = None
        if self.use_remain:
            remain_cache = self.remain_cache[layer_idx].get_decode_cache()
            remain_cache = remain_cache[0] if is_key else remain_cache[1]

        # Concatenate cache parts in order: sparse -> remain -> window
        cache_parts = [x for x in [sparse_cache, remain_cache, window_cache] if x is not None]

        return torch.cat(cache_parts, dim=-2) if cache_parts else None

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        """Return the number of layers in the model."""
        ret_len = 0
        if self.key_cache[0]["window"] is not None:
            ret_len += self.key_cache[0]["window"].shape[-2]
        if self.key_cache[0]["sparse"] is not None:
            ret_len += self.key_cache[0]["sparse"].shape[-2]

        # ret_len += self.key_cache[0]["compressed"].get_length()
        if self.use_remain:
            ret_len += self.remain_cache[0].get_length()

        return ret_len

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Update cache with new key-value states.

        The cache uses a three-tier system:
        - Sparse: Selected sparse KV pairs based on attention scores (full precision)
        - Compressed: Quantized less important KV pairs
        - Window: Recent KV pairs in full precision
        """
        # print(f"Updating cache in {layer_idx=}...")
        # if len(self.key_cache) > layer_idx and layer_idx == 0:
        #     print(f"sparse_len = {self.key_cache[layer_idx]['sparse'].shape[-2] if self.key_cache[layer_idx]['sparse'] is not None else 'None'}")
        #     print(f"window_len = {self.key_cache[layer_idx]['window'].shape[-2] if self.key_cache[layer_idx]['window'] is not None else 'None'}")
        #     print(f"compressed_len = {self.key_cache[layer_idx]['compressed'].get_length()}")

        # Update token count on first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if self.debug and layer_idx == 0:
            print(f"{self.__class__.__name__} update input {layer_idx=}, {query_states.shape[-2]=}")

        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        _, num_heads, _, _ = query_states.shape
        if num_heads != num_kv_heads:
            num_groups = num_heads // num_kv_heads
            assert (
                num_groups * num_kv_heads == num_heads
            ), "num_heads must be divisible by num_kv_heads"
            key_states = repeat_kv(key_states, num_groups)
            value_states = repeat_kv(value_states, num_groups)

        store_key_cache, store_value_cache = self[layer_idx]

        if self.debug and layer_idx == 0:
            print(
                f"{self.__class__.__name__} get store len {store_key_cache.shape[-2] if store_value_cache is not None else 0}"
            )

        ret_key_cache, ret_value_cache = None, None
        if store_key_cache is None:
            ret_key_cache = key_states
            ret_value_cache = value_states
        else:
            ret_key_cache = torch.cat([store_key_cache, key_states], dim=-2)
            ret_value_cache = torch.cat([store_value_cache, value_states], dim=-2)

        # Initialize layer cache if needed
        self._initialize_layer_cache(layer_idx)

        # First time: setup sparse-based partitioning
        if self.key_cache[layer_idx]["window"] is None:
            # Prefill stage
            # Assert that input sequence is long enough for meaningful partitioning
            assert seq_len >= self.min_seq_length, (
                f"Input sequence length ({seq_len}) must be at least {self.min_seq_length} "
                f"(sparse_num: {self.sparse_num} + window_size: {self.window_size}) "
                f"for meaningful sparse-based selection."
            )

            self._setup_sparse_based_cache(
                query_states,
                key_states,
                value_states,
                layer_idx,
            )

            if self.debug and layer_idx == 0:
                print("prefill_stage fill")
                self.print_cache_length(layer_idx)
        else:
            # Decoding stage
            # Subsequent updates: only update window cache
            self._update_window_cache(
                query_states,
                key_states, 
                value_states,
                layer_idx,
            )

            if self.debug and layer_idx == 0:
                print("decode_stage fill")
                self.print_cache_length(layer_idx)

        return ret_key_cache, ret_value_cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the sequence length of the cached states for a given layer."""
        # Check if cache is empty
        if len(self.key_cache) == 0 or len(self.key_cache) <= layer_idx:
            return 0

        key_cache = self.key_cache[layer_idx]

        # Calculate lengths of each cache component
        sparse_length = (
            key_cache["sparse"].shape[-2] if key_cache["sparse"] is not None else 0
        )
        window_length = (
            key_cache["window"].shape[-2] if key_cache["window"] is not None else 0
        )

        # Calculate remain length
        remain_length = 0
        if self.use_remain:
            remain_length = self.remain_cache[layer_idx].get_length()

        return sparse_length + remain_length + window_length

    def print_cache_length(self, layer_idx):
        print(
                f"sparse: {self.key_cache[layer_idx]['sparse'].shape[-2] if self.key_cache[layer_idx]['sparse'] is not None else 0} \
                compressed: {self.remain_cache[layer_idx].get_length() if self.use_remain else 0} \
                window: {self.key_cache[layer_idx]['window'].shape[-2] if self.key_cache[layer_idx]['window'] is not None else 0}"
            )