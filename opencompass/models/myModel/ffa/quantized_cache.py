from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _select_quant_dtype(bits: int) -> torch.dtype:
    """Choose an integer dtype based on quantization bit width."""
    if bits <= 8:
        return torch.int8
    if bits <= 16:
        return torch.int16
    return torch.int32


def _quantize_tensor(x: torch.Tensor, nbit: int, dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    scale = (x_max - x_min).clamp_min(torch.finfo(x.dtype).eps) / (2**nbit - 1)
    zero_point = x_min
    x_quantized = ((x - zero_point) / scale).round().clamp(0, 2**nbit - 1)
    return x_quantized.to(dtype=_select_quant_dtype(nbit)), scale, zero_point


def _dequantize_tensor(x_quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    return x_quantized.to(dtype=scale.dtype) * scale + zero_point


class QuantizedDynamicLayer(CacheLayerMixin):
    """
    Cache layer that quantizes incoming key/value tensors before appending them.
    Dequantized tensors are reconstructed on-the-fly to keep the downstream attention code unchanged.
    """

    is_sliding = False

    def __init__(
        self,
        key_bits: int,
        value_bits: int,
        key_quant_dim: int = -1,
        value_quant_dim: int = -1,
    ):
        super().__init__()
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_quant_dim = key_quant_dim
        self.value_quant_dim = value_quant_dim
        self.seq_dim = -2

        self.key_quantized: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_zero_point: Optional[torch.Tensor] = None
        self.value_quantized: Optional[torch.Tensor] = None
        self.value_scale: Optional[torch.Tensor] = None
        self.value_zero_point: Optional[torch.Tensor] = None

        self.key_quant_dtype = _select_quant_dtype(self.key_bits)
        self.value_quant_dtype = _select_quant_dtype(self.value_bits)

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def _append(self, stored: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if stored is None:
            return new
        return torch.cat([stored, new], dim=self.seq_dim)

    def _refresh_fp_cache(self):
        if self.key_quantized is None or self.value_quantized is None:
            self.keys, self.values = None, None
            return

        self.keys = _dequantize_tensor(self.key_quantized, self.key_scale, self.key_zero_point)
        self.values = _dequantize_tensor(self.value_quantized, self.value_scale, self.value_zero_point)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # 反量化已有缓存，用于返回值拼接（不对当前分片做量化-反量化）
        prev_keys = None
        prev_values = None
        if self.key_quantized is not None and self.value_quantized is not None:
            prev_keys = _dequantize_tensor(self.key_quantized, self.key_scale, self.key_zero_point)
            prev_values = _dequantize_tensor(self.value_quantized, self.value_scale, self.value_zero_point)

        k_quant, k_scale, k_zero_point = _quantize_tensor(key_states, self.key_bits, self.key_quant_dim)
        v_quant, v_scale, v_zero_point = _quantize_tensor(value_states, self.value_bits, self.value_quant_dim)

        self.key_quantized = self._append(self.key_quantized, k_quant.to(self.key_quant_dtype))
        self.key_scale = self._append(self.key_scale, k_scale)
        self.key_zero_point = self._append(self.key_zero_point, k_zero_point)

        self.value_quantized = self._append(self.value_quantized, v_quant.to(self.value_quant_dtype))
        self.value_scale = self._append(self.value_scale, v_scale)
        self.value_zero_point = self._append(self.value_zero_point, v_zero_point)

        # 返回：旧缓存的反量化结果拼接当前未量化的分片
        if prev_keys is None:
            ret_keys = key_states
            ret_values = value_states
        else:
            ret_keys = torch.cat([prev_keys, key_states], dim=self.seq_dim)
            ret_values = torch.cat([prev_values, value_states], dim=self.seq_dim)

        self.keys, self.values = ret_keys, ret_values
        return ret_keys, ret_values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if self.key_quantized is None or self.key_quantized.numel() == 0:
            return 0
        return self.key_quantized.shape[self.seq_dim]

    def get_max_cache_shape(self) -> int:
        return -1

    def _slice_along_seq(self, tensor: Optional[torch.Tensor], max_length: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.narrow(self.seq_dim, 0, max_length)

    def crop(self, max_length: int) -> None:
        current_len = self.get_seq_length()
        if max_length < 0:
            max_length = current_len - abs(max_length)
        if current_len == 0 or current_len <= max_length:
            return

        self.key_quantized = self._slice_along_seq(self.key_quantized, max_length)
        self.key_scale = self._slice_along_seq(self.key_scale, max_length)
        self.key_zero_point = self._slice_along_seq(self.key_zero_point, max_length)

        self.value_quantized = self._slice_along_seq(self.value_quantized, max_length)
        self.value_scale = self._slice_along_seq(self.value_scale, max_length)
        self.value_zero_point = self._slice_along_seq(self.value_zero_point, max_length)
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.key_quantized = self.key_quantized.repeat_interleave(repeats, dim=0)
        self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        self.key_zero_point = self.key_zero_point.repeat_interleave(repeats, dim=0)

        self.value_quantized = self.value_quantized.repeat_interleave(repeats, dim=0)
        self.value_scale = self.value_scale.repeat_interleave(repeats, dim=0)
        self.value_zero_point = self.value_zero_point.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.key_quantized.device)
        self.key_quantized = self.key_quantized.index_select(0, indices)
        self.key_scale = self.key_scale.index_select(0, indices)
        self.key_zero_point = self.key_zero_point.index_select(0, indices)

        self.value_quantized = self.value_quantized.index_select(0, indices)
        self.value_scale = self.value_scale.index_select(0, indices)
        self.value_zero_point = self.value_zero_point.index_select(0, indices)
        self._refresh_fp_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.key_quantized.device)
        self.key_quantized = self.key_quantized.index_select(0, beam_idx)
        self.key_scale = self.key_scale.index_select(0, beam_idx)
        self.key_zero_point = self.key_zero_point.index_select(0, beam_idx)

        self.value_quantized = self.value_quantized.index_select(0, beam_idx)
        self.value_scale = self.value_scale.index_select(0, beam_idx)
        self.value_zero_point = self.value_zero_point.index_select(0, beam_idx)
        self._refresh_fp_cache()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        self.key_quantized = self.key_quantized[..., :0, :].contiguous() if self.key_quantized is not None else None
        self.key_scale = self.key_scale[..., :0, :].contiguous() if self.key_scale is not None else None
        self.key_zero_point = self.key_zero_point[..., :0, :].contiguous() if self.key_zero_point is not None else None

        self.value_quantized = self.value_quantized[..., :0, :].contiguous() if self.value_quantized is not None else None
        self.value_scale = self.value_scale[..., :0, :].contiguous() if self.value_scale is not None else None
        self.value_zero_point = (
            self.value_zero_point[..., :0, :].contiguous() if self.value_zero_point is not None else None
        )
        self._refresh_fp_cache()

    def offload(self):
        if self.is_initialized:
            if self.key_quantized is not None:
                self.key_quantized = self.key_quantized.to("cpu", non_blocking=True)
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
                self.key_zero_point = self.key_zero_point.to("cpu", non_blocking=True)
            if self.value_quantized is not None:
                self.value_quantized = self.value_quantized.to("cpu", non_blocking=True)
                self.value_scale = self.value_scale.to("cpu", non_blocking=True)
                self.value_zero_point = self.value_zero_point.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.keys is not None and self.keys.device != self.device:
            if self.key_quantized is not None:
                self.key_quantized = self.key_quantized.to(self.device, non_blocking=True)
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
                self.key_zero_point = self.key_zero_point.to(self.device, non_blocking=True)
            if self.value_quantized is not None:
                self.value_quantized = self.value_quantized.to(self.device, non_blocking=True)
                self.value_scale = self.value_scale.to(self.device, non_blocking=True)
                self.value_zero_point = self.value_zero_point.to(self.device, non_blocking=True)
        super().prefetch()


class QuantizedCache(Cache):
    """
    Drop-in Cache subclass that quantizes every append operation.

    This keeps the public Cache API intact, so it can be passed directly to
    `modeling_llama.LlamaAttention` as `past_key_values`.
    """

    def __init__(
        self,
        key_bits: int = 8,
        value_bits: int = 8,
        key_quant_dim: int = -1,
        value_quant_dim: int = -1,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.key_quant_dim = key_quant_dim
        self.value_quant_dim = value_quant_dim

    def _ensure_layer(self, layer_idx: int):
        while len(self.layers) <= layer_idx:
            self.layers.append(
                QuantizedDynamicLayer(
                    key_bits=self.key_bits,
                    value_bits=self.value_bits,
                    key_quant_dim=self.key_quant_dim,
                    value_quant_dim=self.value_quant_dim,
                )
            )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)

        if self.offloading:
            torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        keys, values = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return keys, values
