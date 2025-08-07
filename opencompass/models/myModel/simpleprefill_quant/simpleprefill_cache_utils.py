from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import repeat_kv

class SimplePrefill_Cache(DynamicCache):
    """简化的KV缓存实现，直接拼接新KV到旧缓存后面"""
    
    def __init__(self, config) -> None:
        super().__init__()
        self.key_cache = []  # 存储每层的key缓存
        self.value_cache = []  # 存储每层的value缓存
        self._seen_tokens = 0

        self.kvcache_settings = config.kvcache_settings

        print("-------------------- SimplePrefill_Cache init --------------------")
        self.debug = True

    def _initialize_layer_cache(self, layer_idx: int):
        """初始化指定层的缓存空间"""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层的完整KV缓存"""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx], self.value_cache[layer_idx]
        return None, None

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存：直接将新KV拼接到旧缓存后面
        返回更新后的完整缓存
        """
        # 更新已处理的token计数（仅在首层更新）
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        print(f"{layer_idx} Input shape = {key_states.shape}")
        
        # 处理GQA情况（分组查询注意力）
        bsz, num_heads, seq_len, head_dim = query_states.shape
        _, num_kv_heads, _, _ = key_states.shape
        # if num_heads != num_kv_heads:
        #     num_groups = num_heads // num_kv_heads
        #     key_states = repeat_kv(key_states, num_groups)
        #     value_states = repeat_kv(value_states, num_groups)
        
        # 确保该层缓存已初始化
        self._initialize_layer_cache(layer_idx)
        
        # 获取当前层缓存
        current_key_cache = self.key_cache[layer_idx]
        current_value_cache = self.value_cache[layer_idx]
        
        # 更新缓存：直接拼接新KV
        if current_key_cache is None:
            ret_key_cache = key_states
            ret_value_cache = value_states
        else:
            ret_key_cache = torch.cat([current_key_cache, key_states], dim=-2)
            ret_value_cache = torch.cat([current_value_cache, value_states], dim=-2)
        
        # 存储更新后的缓存
        if current_key_cache is None:
            print(f"Prefill Stage in {layer_idx=}")
            print(f"Compress shape = {ret_key_cache.shape}")
            from .quant_utils import quantize_tensor, dequantize_tensor
            self.key_cache[layer_idx] = dequantize_tensor(*quantize_tensor(ret_key_cache, 
                                                            self.kvcache_settings['k_bits'], 
                                                            self.kvcache_settings['k_quant_dim']))
            self.value_cache[layer_idx] = dequantize_tensor(*quantize_tensor(ret_value_cache, 
                                                            self.kvcache_settings['v_bits'], 
                                                            self.kvcache_settings['v_quant_dim']))
        else:
            print(f"Decode Stage in {layer_idx=}")
            self.key_cache[layer_idx] = ret_key_cache
            self.value_cache[layer_idx] = ret_value_cache
        
        print(f"{layer_idx} Return shape = {ret_key_cache.shape}")
        
        return ret_key_cache, ret_value_cache

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """获取指定层的缓存序列长度"""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx].shape[-2]
        return 0