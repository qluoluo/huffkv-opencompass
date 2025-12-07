#!/usr/bin/env python3
"""
Minimal example showing how to call Quest's attention kernels directly.

The script exposes two helpers:
  - quest_prefill_attention: run prefill (q_len > 1) attention on a KV cache.
  - quest_decode_attention: run decode (q_len == 1) attention with optional Top-K sparsity.

It then compares Quest outputs with a PyTorch reference to verify correctness.
"""
import math
import sys
from pathlib import Path

import torch

# 允许直接在仓库外层目录运行：自动把实际的仓库根目录（包含 pyproject.toml 的 quest 子目录）加入 sys.path
FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR / "quest"
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import quest.utils


def _reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Causal self-attention reference in NHD layout (matches Hugging Face LLaMA implementation).
    q: [q_len, num_heads, head_dim]
    k/v: [kv_len, num_heads, head_dim]
    """
    assert q.dim() == k.dim() == v.dim() == 3
    assert k.size(0) == v.size(0), "k/v lengths must match"
    assert q.size(1) == k.size(1), "num_heads must match between q and k/v"

    head_dim = q.size(2)
    q_len = q.size(0)
    kv_len = k.size(0)
    if q_len > kv_len:
        raise ValueError("q_len must be <= kv_len for causal attention")

    q_t = q.transpose(0, 1)  # [num_heads, q_len, head_dim]
    k_t = k.transpose(0, 1)  # [num_heads, kv_len, head_dim]
    v_t = v.transpose(0, 1)

    scores = torch.matmul(q_t, k_t.transpose(1, 2)) / math.sqrt(head_dim)
    causal_mask = torch.ones_like(scores, dtype=torch.bool).tril(diagonal=kv_len - q_len)
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

    return torch.matmul(probs, v_t).transpose(0, 1)


def quest_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    page_size: int = 16,
    page_budget: int | None = None,
) -> torch.Tensor:
    """
    Run Quest prefill attention (q_len > 1) for a single layer.
    Inputs must be on CUDA and in NHD layout.
    """
    if page_budget is None:
        max_seq_len = k.size(0)
        page_budget = (max_seq_len + page_size - 1) // page_size

    controller = quest.utils.InferenceController(
        num_layers=1,
        num_heads=q.size(1),
        head_dim=q.size(2),
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=k.size(0),
        dtype=q.dtype,
        device=q.device,
    )

    controller.prepare_metadata(k.size(0))
    controller.begin_forward(k.size(0))
    quest.utils.append_kv(k, v, controller, layer_idx=0)
    return quest.utils.prefill_forward(q, controller, layer_idx=0)


def quest_decode_attention(
    q: torch.Tensor,
    past_k: torch.Tensor,
    past_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    page_size: int = 16,
    page_budget: int | None = None,
) -> torch.Tensor:
    """
    Run Quest decode attention (q_len == 1) for a single layer.
    If page_budget covers all pages, it behaves like full attention;
    otherwise it will estimate Top-K pages automatically.
    """
    if q.size(0) != 1:
        raise ValueError("Decode path expects q_len == 1.")

    total_len = past_k.size(0) + new_k.size(0)
    if page_budget is None:
        page_budget = (total_len + page_size - 1) // page_size

    controller = quest.utils.InferenceController(
        num_layers=1,
        num_heads=q.size(1),
        head_dim=q.size(2),
        page_size=page_size,
        page_budget=page_budget,
        max_seq_len=total_len,
        dtype=q.dtype,
        device=q.device,
    )

    # Stage 1: prefill the cache.
    controller.prepare_metadata(past_k.size(0))
    controller.begin_forward(past_k.size(0))
    quest.utils.append_kv(past_k, past_v, controller, layer_idx=0)
    controller.end_forward()

    # Stage 2: append the decoding token and run attention.
    controller.prepare_metadata(new_k.size(0))
    controller.begin_forward(new_k.size(0))
    quest.utils.append_kv(new_k, new_v, controller, layer_idx=0)

    if controller.need_estimate():
        estimated = quest.utils.decode_estimate(q, controller, layer_idx=0)
        quest.utils.decode_topk(estimated, controller)
        topk_indices = controller.topk_dindices_buffer
    else:
        topk_indices = controller.kv_indices_without_last

    out = quest.utils.decode_sparse_attn(
        q,
        controller,
        layer_idx=0,
        topk_indices=topk_indices,
    )
    controller.end_forward()
    return out


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise SystemExit("Quest kernels require a CUDA-capable GPU.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    page_size = 16

    # Prefill example (q_len > 1)
    q_len, kv_len = 4, 24
    num_heads, head_dim = 8, 64
    q = torch.randn(q_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(kv_len, num_heads, head_dim, device=device, dtype=dtype)

    quest_prefill = quest_prefill_attention(q, k, v, page_size=page_size)
    ref_prefill = _reference_attention(q, k, v)
    torch.testing.assert_close(quest_prefill, ref_prefill, rtol=5e-3, atol=5e-3)
    print(f"Prefill attention matched reference (max error: {(quest_prefill - ref_prefill).abs().max().item():.3e})")

    # Decode example (q_len == 1)
    past_len = 63  # ensures multiple pages with page_size=16
    past_k = torch.randn(past_len, num_heads, head_dim, device=device, dtype=dtype)
    past_v = torch.randn(past_len, num_heads, head_dim, device=device, dtype=dtype)
    new_k = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    new_v = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)
    q_decode = torch.randn(1, num_heads, head_dim, device=device, dtype=dtype)

    quest_decode = quest_decode_attention(
        q_decode,
        past_k,
        past_v,
        new_k,
        new_v,
        page_size=page_size,
    )
    ref_decode = _reference_attention(q_decode, torch.cat([past_k, new_k], dim=0), torch.cat([past_v, new_v], dim=0))
    torch.testing.assert_close(quest_decode, ref_decode, rtol=5e-3, atol=5e-3)
    print(f"Decode attention matched reference (max error: {(quest_decode - ref_decode).abs().max().item():.3e})")


if __name__ == "__main__":
    main()
