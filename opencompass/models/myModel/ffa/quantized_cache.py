from __future__ import annotations

import math
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


def _select_quant_dtype(bits: int) -> torch.dtype:
    """Choose an integer dtype based on quantization bit width."""
    if bits <= 8:
        return torch.uint8
    if bits <= 16:
        return torch.int16
    return torch.int32


def _compute_quant_params(x: torch.Tensor, nbit: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel scale/zero-point along a given dimension."""
    x_min = x.amin(dim=dim)
    x_max = x.amax(dim=dim)
    scale = (x_max - x_min).clamp_min(torch.finfo(x.dtype).eps) / (2**nbit - 1)
    zero_point = x_min
    return scale, zero_point


def _quantize_with_params(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, nbit: int, dim: int
) -> torch.Tensor:
    """Quantize using fixed per-channel parameters (clamps out-of-range values)."""
    qmax = 2**nbit - 1
    scale = scale.to(dtype=x.dtype, device=x.device)
    zero_point = zero_point.to(dtype=x.dtype, device=x.device)
    # Insert the quantization dimension for broadcasting if missing.
    if scale.dim() == x.dim() - 1:
        scale = scale.unsqueeze(dim)
        zero_point = zero_point.unsqueeze(dim)
    x_quantized = ((x - zero_point) / scale).round().clamp(0, qmax)
    return x_quantized.to(dtype=_select_quant_dtype(nbit))


def _pack_bits(x: torch.Tensor, nbit: int, vals_per_byte: Optional[int] = None) -> torch.Tensor:
    """
    Pack quantized values along the last dimension to reduce memory.

    For example, 2-bit values are packed as 4 values per byte, so [B, T, H, K]
    becomes [B, T, H, ceil(K / 4)].
    """
    vals_per_byte = vals_per_byte or (8 // nbit)
    if vals_per_byte <= 1:
        return x.to(dtype=_select_quant_dtype(nbit))

    x = x.contiguous()
    k = x.shape[-1]
    k_pad = math.ceil(k / vals_per_byte) * vals_per_byte
    if k_pad != k:
        pad = torch.zeros(*x.shape[:-1], k_pad - k, device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=-1)

    x_grouped = x.view(*x.shape[:-1], k_pad // vals_per_byte, vals_per_byte).to(torch.int32)
    packed = torch.zeros(*x_grouped.shape[:-1], device=x.device, dtype=torch.int32)
    for i in range(vals_per_byte):
        packed |= x_grouped[..., i] << (i * nbit)
    return packed.to(dtype=_select_quant_dtype(nbit))


class QuantizedDynamicLayer(CacheLayerMixin):
    """
    Cache layer that quantizes incoming key/value tensors before appending them.
    Only key tensors are quantized (2-bit); quantized keys are bit-packed to shrink memory
    while the original key/value tensors are retained alongside the packed keys.
    """

    is_sliding = False

    def __init__(
        self,
        key_bits: int = 2,
        key_quant_dim: int = 1,
    ):
        super().__init__()
        self.key_bits = key_bits
        self.key_quant_dim = key_quant_dim  # token dimension for [B, T, H, D] input
        self.seq_dim = 1  # sequence dimension for [B, T, H, D]
        # Only pack when the bit-width evenly divides a byte; otherwise fall back to 1-value-per-byte.
        self.vals_per_byte = 8 // self.key_bits if self.key_bits > 0 and 8 % self.key_bits == 0 else 1

        self.key_quantized: Optional[torch.Tensor] = None
        self.key_scale: Optional[torch.Tensor] = None
        self.key_zero_point: Optional[torch.Tensor] = None
        self.key_original: Optional[torch.Tensor] = None
        self.value_original: Optional[torch.Tensor] = None
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None

        self.key_quant_dtype = _select_quant_dtype(self.key_bits)

    def lazy_initialization(self, key_states: torch.Tensor):
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def _append(self, stored: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if stored is None:
            return new
        return torch.cat([stored, new], dim=self.seq_dim)

    def _refresh_fp_cache(self):
        self.keys = self.key_original
        self.values = self.value_original

    def _quantize_keys(self, key_states: torch.Tensor) -> torch.Tensor:
        if self.key_scale is None or self.key_zero_point is None:
            self.key_scale, self.key_zero_point = _compute_quant_params(key_states, self.key_bits, self.key_quant_dim)
        elif self.key_scale.device != key_states.device:
            self.key_scale = self.key_scale.to(key_states.device, non_blocking=True)
            self.key_zero_point = self.key_zero_point.to(key_states.device, non_blocking=True)
        k_quant = _quantize_with_params(key_states, self.key_scale, self.key_zero_point, self.key_bits, self.key_quant_dim)
        if self.vals_per_byte > 1:
            k_quant = _pack_bits(k_quant, self.key_bits, self.vals_per_byte)
        return k_quant.to(self.key_quant_dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        k_quant = self._quantize_keys(key_states)

        self.key_quantized = self._append(self.key_quantized, k_quant)
        self.key_original = self._append(self.key_original, key_states)
        self.value_original = self._append(self.value_original, value_states)
        self._refresh_fp_cache()

        return self.keys, self.values

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
        self.key_original = self._slice_along_seq(self.key_original, max_length)
        self.value_original = self._slice_along_seq(self.value_original, max_length)
        self._refresh_fp_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.get_seq_length() == 0:
            return
        self.key_quantized = self.key_quantized.repeat_interleave(repeats, dim=0)
        self.key_original = self.key_original.repeat_interleave(repeats, dim=0)
        self.value_original = self.value_original.repeat_interleave(repeats, dim=0)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.repeat_interleave(repeats, dim=0)
        if self.key_zero_point is not None:
            self.key_zero_point = self.key_zero_point.repeat_interleave(repeats, dim=0)
        self._refresh_fp_cache()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.get_seq_length() == 0:
            return
        indices = indices.to(self.key_quantized.device)
        self.key_quantized = self.key_quantized.index_select(0, indices)
        self.key_original = self.key_original.index_select(0, indices)
        self.value_original = self.value_original.index_select(0, indices)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, indices)
        if self.key_zero_point is not None:
            self.key_zero_point = self.key_zero_point.index_select(0, indices)
        self._refresh_fp_cache()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.get_seq_length() == 0:
            return
        beam_idx = beam_idx.to(self.key_quantized.device)
        self.key_quantized = self.key_quantized.index_select(0, beam_idx)
        self.key_original = self.key_original.index_select(0, beam_idx)
        self.value_original = self.value_original.index_select(0, beam_idx)
        if self.key_scale is not None:
            self.key_scale = self.key_scale.index_select(0, beam_idx)
        if self.key_zero_point is not None:
            self.key_zero_point = self.key_zero_point.index_select(0, beam_idx)
        self._refresh_fp_cache()

    def reset(self) -> None:
        if not self.is_initialized:
            return
        if self.key_quantized is not None:
            self.key_quantized = self.key_quantized.narrow(self.seq_dim, 0, 0).contiguous()
        if self.key_original is not None:
            self.key_original = self.key_original.narrow(self.seq_dim, 0, 0).contiguous()
        if self.value_original is not None:
            self.value_original = self.value_original.narrow(self.seq_dim, 0, 0).contiguous()
        self.key_scale = None
        self.key_zero_point = None
        self._refresh_fp_cache()

    def offload(self):
        if self.is_initialized:
            if self.key_quantized is not None:
                self.key_quantized = self.key_quantized.to("cpu", non_blocking=True)
                self.key_scale = self.key_scale.to("cpu", non_blocking=True)
                self.key_zero_point = self.key_zero_point.to("cpu", non_blocking=True)
            if self.key_original is not None:
                self.key_original = self.key_original.to("cpu", non_blocking=True)
            if self.value_original is not None:
                self.value_original = self.value_original.to("cpu", non_blocking=True)
        super().offload()

    def prefetch(self):
        if self.is_initialized and self.keys is not None and self.keys.device != self.device:
            if self.key_quantized is not None:
                self.key_quantized = self.key_quantized.to(self.device, non_blocking=True)
                self.key_scale = self.key_scale.to(self.device, non_blocking=True)
                self.key_zero_point = self.key_zero_point.to(self.device, non_blocking=True)
            if self.key_original is not None:
                self.key_original = self.key_original.to(self.device, non_blocking=True)
            if self.value_original is not None:
                self.value_original = self.value_original.to(self.device, non_blocking=True)
        super().prefetch()


class QuantizedCache(Cache):
    """
    Drop-in Cache subclass that quantizes key tensors on every append operation.
    Keys are stored both in 2-bit (packed) quantized form and in full-precision; values are stored in full-precision only.

    This keeps the public Cache API intact, so it can be passed directly to
    `modeling_llama.LlamaAttention` as `past_key_values`.
    """

    def __init__(
        self,
        key_bits: int = 2,
        key_quant_dim: int = 1,
        offloading: bool = False,
        offload_only_non_sliding: bool = True,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self.key_bits = key_bits
        self.key_quant_dim = key_quant_dim

    def _ensure_layer(self, layer_idx: int):
        while len(self.layers) <= layer_idx:
            self.layers.append(
                QuantizedDynamicLayer(
                    key_bits=self.key_bits,
                    key_quant_dim=self.key_quant_dim,
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
