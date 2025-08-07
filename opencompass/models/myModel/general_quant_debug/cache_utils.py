from typing import Any, Dict, List, Optional, Tuple, Union
from functools import partial

import torch
from transformers.cache_utils import DynamicCache
from .quant_utils import quantize_tensor, dequantize_tensor


class CustomCache(DynamicCache):
    """
    Custom cache implementation with three-tier storage (global, mid, local) 
    and independent quantization for key and value tensors.
    """
    
    CustomCache_init = False
    
    def __init__(self, config) -> None:
        super().__init__()
        if not CustomCache.CustomCache_init:
            CustomCache.CustomCache_init = True
            print("-------------------- CustomCache init --------------------")

        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

        self.kvcache_settings = config.kvcache_settings
        
        # Initialize quantization functions
        self._init_quantization_functions()
        
        # Cache configuration
        self.group_size = self.kvcache_settings.group_size
        self.global_length = self.kvcache_settings.global_residual_length
        self.local_length = self.kvcache_settings.local_residual_length

        self.prefill_stage = True

    def _init_quantization_functions(self):
        """Initialize quantization and dequantization functions."""
        self.key_quant_func = partial(
            quantize_tensor, 
            nbit=self.kvcache_settings.k_bits, 
            dim=self.kvcache_settings.k_quant_dim
        )
        self.value_quant_func = partial(
            quantize_tensor, 
            nbit=self.kvcache_settings.v_bits, 
            dim=self.kvcache_settings.v_quant_dim
        )
        self.key_dequant_func = partial(dequantize_tensor)
        self.value_dequant_func = partial(dequantize_tensor)

    def _update_local_cache(self, layer_idx, key_states: torch.Tensor, value_states: torch.Tensor):
        """Update local cache by concatenating new states."""
        key_cache_layer = self.key_cache[layer_idx]
        value_cache_layer = self.value_cache[layer_idx]
        
        if key_cache_layer["local"] is not None:
            key_cache_layer["local"] = torch.cat([key_cache_layer["local"], key_states], dim=-2)
            value_cache_layer["local"] = torch.cat([value_cache_layer["local"], value_states], dim=-2)
        else:
            key_cache_layer["local"] = key_states
            value_cache_layer["local"] = value_states

    def _process_excess_cache(self, layer_idx: int, cache_type: str, local_length: int, 
                            group_size: int, quant_func):
        """
        Process excess tokens in local cache by moving them to mid cache with quantization.
        
        Args:
            layer_idx: Layer index to process
            cache_type: "key" or "value" for error messages
            local_length: Maximum allowed local cache length
            group_size: Group size for quantization
            quant_func: Quantization function to use
        """
        if group_size <= 0 and not self.prefill_stage:
            return

        cache_dict = self.key_cache[layer_idx] if cache_type == "key" else self.value_cache[layer_idx]
        
        if cache_dict["local"].shape[-2] <= local_length:
            return
        if group_size > 0 and cache_dict["local"].shape[-2] < local_length + group_size:
            return

        excess_length = cache_dict["local"].shape[-2] - local_length
        if group_size > 0:
            remain_length = excess_length % group_size
            excess_length -= remain_length


        # Split local cache into remaining and excess parts
        excess_tensor, cache_dict["local"] = (
            cache_dict["local"][..., :excess_length, :],
            cache_dict["local"][..., excess_length:, :],
        )

        # Apply grouping if specified
        if group_size > 0:
            excess_tensor = self._apply_grouping(excess_tensor, group_size, cache_type)

        # Quantize excess tensor
        excess_quant_data = quant_func(excess_tensor)
        
        if layer_idx == 8:
            print(f"{layer_idx=}, {excess_length=}, compress shape = {excess_tensor.shape}")

        # Update mid cache
        self._update_mid_cache(cache_dict, excess_quant_data, layer_idx)

    def _apply_grouping(self, tensor: torch.Tensor, group_size: int, cache_type: str) -> torch.Tensor:
        """Apply grouping to tensor for quantization."""
        num_groups = tensor.shape[-2] // group_size
        if num_groups * group_size != tensor.shape[-2]:
            raise ValueError(
                f"tensor.shape[-2] ({tensor.shape[-2]}) cannot be evenly divided "
                f"by {cache_type} group size ({group_size})"
            )
        
        return tensor.view(
            *tensor.shape[:-2], num_groups, group_size, tensor.shape[-1]
        )

    def _update_mid_cache(self, cache_dict: dict, quant_data, layer_idx=-1):

        """Update mid cache with quantized data."""
        if cache_dict["mid"] is not None:
            for i, data in enumerate(quant_data):
                cache_dict["mid"][i] = torch.cat([cache_dict["mid"][i], data], dim=2)
        else:
            cache_dict["mid"] = list(quant_data)

        # if layer_idx == 0:
        #     print("update mid cache")
        #     for i, data in enumerate(quant_data):
        #         print(f"{data.shape} ", end='')
        #     print('')
        #     for i, data in enumerate(quant_data):
        #         print(f"{cache_dict['mid'][i].shape} ", end='')
        #     print('\n')

    def _reconstruct_cache(self, layer_idx: int, is_key: bool) -> torch.Tensor:
        """Reconstruct full cache tensor from global, mid, and local parts."""
        cache_dict = self.key_cache[layer_idx] if is_key else self.value_cache[layer_idx]
        dequant_func = self.key_dequant_func if is_key else self.value_dequant_func
        group_size = self.group_size

        # Get cache components
        global_cache = cache_dict.get("global")
        mid_cache_quant = cache_dict.get("mid")
        local_cache = cache_dict.get("local")

        # Dequantize mid cache
        mid_cache = dequant_func(*mid_cache_quant) if mid_cache_quant else None

        # Apply group size reshaping
        if mid_cache is not None and group_size > 0:
            mid_cache = mid_cache.view(
                *mid_cache.shape[:-3], 
                mid_cache.shape[-3] * mid_cache.shape[-2], 
                mid_cache.shape[-1]
            )

        # Concatenate cache parts
        cache_parts = []
        if global_cache is not None:
            cache_parts.append(global_cache)
        if mid_cache is not None:
            cache_parts.append(mid_cache)
        if local_cache is not None:
            cache_parts.append(local_cache)

        return torch.cat(cache_parts, dim=-2) if cache_parts else None

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and reconstruct full key-value cache for a layer."""
        key_cache = self._reconstruct_cache(layer_idx, is_key=True)
        value_cache = self._reconstruct_cache(layer_idx, is_key=False)
        return key_cache, value_cache

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        """Return the number of layers in the model."""
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states.
        
        The cache uses a three-tier system:
        - Global: Fixed-size cache for the beginning tokens
        - Mid: Quantized cache for intermediate tokens (with grouping)
        - Local: Full-precision cache for recent tokens
        """
        # Update token count on first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if layer_idx == 8:
            print(f"{layer_idx=}, now update kv cache {key_states.shape}")
            self.print_cache_stats(layer_idx)

        
        if len(self.key_cache) <= layer_idx:
            self.prefill_stage = True
            self.key_cache.append({"global": None, "mid": None, "local": None})
            self.value_cache.append({"global": None, "mid": None, "local": None})
        else:
            self.prefill_stage = False

        # Get current cache state before update
        store_key_cache, store_value_cache = self[layer_idx]
        ret_key_cache = torch.cat([store_key_cache, key_states], dim=-2) if store_key_cache is not None else key_states
        ret_value_cache = torch.cat([store_value_cache, value_states], dim=-2) if store_value_cache is not None else value_states
        
        if self.prefill_stage:
            # Calculate cache lengths
            assert key_states.shape[-2] >= self.global_length + self.local_length, \
                f"prefill input length too short: {key_states.shape[-2]=} <= {self.global_length=} + {self.local_length=}"

            # Setup global cache
            if self.global_length > 0:
                self.key_cache[layer_idx]["global"] = key_states[..., :self.global_length, :]
                self.value_cache[layer_idx]["global"] = value_states[..., :self.global_length, :]
            key_states = key_states[..., self.global_length:, :]
            value_states = value_states[..., self.global_length:, :]

        # Update local cache
        self._update_local_cache(layer_idx, key_states, value_states)

        # Process excess in key cache
        self._process_excess_cache(
            layer_idx, "key", self.local_length, 
            self.group_size, self.key_quant_func
        )
        # Process excess in value cache
        self._process_excess_cache(
            layer_idx, "value", self.local_length,
            self.group_size, self.value_quant_func
        )

        if layer_idx == 8:
            print(f"{layer_idx=}, now return kv cache {ret_key_cache.shape}")
            self.print_cache_stats(layer_idx)

        return ret_key_cache, ret_value_cache

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def get_seq_length(self, layer_idx: int = 0) -> int:
        return (
            self.get_cache_length(layer_idx, "global")
            + self.get_cache_length(layer_idx, "mid")
            + self.get_cache_length(layer_idx, "local")
        )

    def get_cache_length(self, layer_idx: int, cache_key: str) -> int:
        if layer_idx >= len(self.key_cache):
            return 0
        cache_dict = self.key_cache[layer_idx]
        if cache_dict[cache_key] is None:
            return 0

        if cache_key in {"global", "local"}:
            return cache_dict[cache_key].shape[-2]

        # cache_key == 'mid'
        if self.group_size > 0:
            # mid[0] shape: [B, G, T_mid, H]
            return cache_dict["mid"][0].shape[-3] * cache_dict["mid"][0].shape[-2]
        return cache_dict["mid"][0].shape[-2]

    def print_cache_stats(self, layer_idx: int) -> None:
        if layer_idx >= len(self.key_cache):
            print(f"Layer {layer_idx}: global={0}  mid={0}  local={0}")
            return
        g = self.get_cache_length(layer_idx, "global")
        m = self.get_cache_length(layer_idx, "mid")
        l = self.get_cache_length(layer_idx, "local")
        print(f"Layer {layer_idx}: global={g}  mid={m}  local={l}")