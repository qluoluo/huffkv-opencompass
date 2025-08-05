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

        # self.key_cache: List[dict] = []
        # self.value_cache: List[dict] = []
        self.key_cache = []
        self.value_cache = []
        self._seen_tokens = 0

        self.kvcache_settings = config.kvcache_settings
        
        # Initialize quantization functions
        self._init_quantization_functions()
        
        # Cache configuration
        self.key_group_size = self.kvcache_settings.key_group_size
        self.value_group_size = self.kvcache_settings.value_group_size

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

    def _initialize_layer_cache(self, layer_idx: int):
        """Initialize cache dictionaries for a new layer if not present."""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append({"global": None, "mid": None, "local": None})
            self.value_cache.append({"global": None, "mid": None, "local": None})

    def _setup_global_cache(self, key_states: torch.Tensor, value_states: torch.Tensor, 
                          current_key_cache: dict, current_value_cache: dict, 
                          global_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set up global cache with the first `global_length` tokens.
        Returns the remaining key_states and value_states after global extraction.
        """
        if current_key_cache["global"] is None:
            if key_states.shape[-2] < global_length:
                raise ValueError(
                    f"global_length ({global_length}) must be less than or equal to "
                    f"key_states.shape[-2] ({key_states.shape[-2]})"
                )

            current_key_cache["global"] = key_states[..., :global_length, :]
            current_value_cache["global"] = value_states[..., :global_length, :]

            key_states = key_states[..., global_length:, :]
            value_states = value_states[..., global_length:, :]

        return key_states, value_states

    def _update_local_cache(self, key_states: torch.Tensor, value_states: torch.Tensor,
                          current_key_cache: dict, current_value_cache: dict):
        """Update local cache by concatenating new states."""
        if current_key_cache["local"] is not None:
            current_key_cache["local"] = torch.cat([current_key_cache["local"], key_states], dim=-2)
            current_value_cache["local"] = torch.cat([current_value_cache["local"], value_states], dim=-2)
        else:
            current_key_cache["local"] = key_states
            current_value_cache["local"] = value_states

    def _process_excess_cache(self, cache_dict: dict, cache_type: str, local_length: int, 
                            group_size: int, quant_func):
        """
        Process excess tokens in local cache by moving them to mid cache with quantization.
        
        Args:
            cache_dict: The cache dictionary (key or value)
            cache_type: "key" or "value" for error messages
            local_length: Maximum allowed local cache length
            group_size: Group size for quantization
            quant_func: Quantization function to use
        """
        if cache_dict["local"].shape[-2] <= local_length:
            return

        excess_length = cache_dict["local"].shape[-2] - local_length
        if excess_length <= 0:
            return

        # Split local cache into remaining and excess parts
        cache_dict["local"], excess_tensor = (
            cache_dict["local"][..., -local_length:, :],
            cache_dict["local"][..., :excess_length, :],
        )

        # Apply grouping if specified
        if group_size > 0:
            excess_tensor = self._apply_grouping(excess_tensor, group_size, cache_type)

        # Quantize excess tensor
        excess_quant_data = quant_func(excess_tensor)

        # Update mid cache
        self._update_mid_cache(cache_dict, excess_quant_data)

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

    def _update_mid_cache(self, cache_dict: dict, quant_data):
        """Update mid cache with quantized data."""
        if cache_dict["mid"] is not None:
            for i, data in enumerate(quant_data):
                cache_dict["mid"][i] = torch.cat([cache_dict["mid"][i], data], dim=2)
        else:
            cache_dict["mid"] = list(quant_data)

    def _calculate_local_lengths(self, mid_length: int) -> Tuple[int, int]:
        """Calculate adjusted local lengths based on group sizes and mid_length."""
        base_local_length = self.kvcache_settings.local_residual_length
        
        key_local_length = base_local_length
        value_local_length = base_local_length
        
        # Adjust for remainder when using group sizes
        if self.key_group_size > 0:
            key_local_length += (mid_length % self.key_group_size)
        if self.value_group_size > 0:
            value_local_length += (mid_length % self.value_group_size)
            
        return key_local_length, value_local_length

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and reconstruct full key-value cache for a layer."""
        key_cache = self._reconstruct_cache(layer_idx, is_key=True)
        value_cache = self._reconstruct_cache(layer_idx, is_key=False)
        return key_cache, value_cache

    def _reconstruct_cache(self, layer_idx: int, is_key: bool) -> torch.Tensor:
        """Reconstruct full cache tensor from global, mid, and local parts."""
        cache_dict = self.key_cache[layer_idx] if is_key else self.value_cache[layer_idx]
        dequant_func = self.key_dequant_func if is_key else self.value_dequant_func
        group_size = self.key_group_size if is_key else self.value_group_size

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

        # Calculate cache lengths
        global_length = self.kvcache_settings.global_residual_length
        base_local_length = self.kvcache_settings.local_residual_length
        mid_length = self._seen_tokens - global_length - base_local_length
        
        key_local_length, value_local_length = self._calculate_local_lengths(mid_length)

        # Initialize layer cache if needed
        self._initialize_layer_cache(layer_idx)

        # Get current cache references
        current_key_cache = self.key_cache[layer_idx]
        current_value_cache = self.value_cache[layer_idx]

        # Setup global cache (only on first update)
        key_states, value_states = self._setup_global_cache(
            key_states, value_states, current_key_cache, current_value_cache, global_length
        )

        # Update local cache
        self._update_local_cache(key_states, value_states, current_key_cache, current_value_cache)

        # Process excess in local caches (move to mid with quantization)
        self._process_excess_cache(
            current_key_cache, "key", key_local_length, 
            self.key_group_size, self.key_quant_func
        )
        self._process_excess_cache(
            current_value_cache, "value", value_local_length,
            self.value_group_size, self.value_quant_func
        )

        # Update cache references (though they're already updated by reference)
        self.key_cache[layer_idx] = current_key_cache
        self.value_cache[layer_idx] = current_value_cache

        return self[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the sequence length of the cached states for a given layer."""
        # Check if cache is empty
        if (len(self.key_cache) == 0 or len(self.key_cache) <= layer_idx):
            return 0

        key_cache = self.key_cache[layer_idx]
        
        # Calculate lengths of each cache component
        global_length = key_cache['global'].shape[-2] if key_cache['global'] is not None else 0
        local_length = key_cache['local'].shape[-2] if key_cache['local'] is not None else 0
        
        # Calculate mid length (considering group size)
        if key_cache['mid'] is not None:
            if self.key_group_size > 0:
                mid_length = key_cache['mid'][0].shape[-3] * self.key_group_size
            else:
                mid_length = key_cache['mid'].shape[-2]
        else:
            mid_length = 0

        return global_length + mid_length + local_length