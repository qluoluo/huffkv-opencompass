# from .ffa_triton_decode.attn_kernel.attn_kernel_v1208_fused_bsz_fp16_refine import attn_forward_decode
# from .ffa_triton_decode.attn_kernel.attn_kernel_v1208_fused_bsz_fp8 import attn_forward_decode
from .ffa_triton_decode.attn_kernel.attn_kernel_v1210_fused_bsz_q2 import (
    attn_forward_decode_quantized as _attn_forward_decode_q2,
)
from .ffa_triton_decode.attn_kernel.attn_kernel_v1210_fused_bsz_q2fp8 import (
    attn_forward_decode_quantized as _attn_forward_decode_q2fp8,
)


def _normalize_kernel_name(kernel: str | None) -> str:
    if kernel is None:
        return "q2"
    return str(kernel).strip().lower()


def attn_forward_decode(
    *,
    q,
    k_q,
    k_scale,
    k_zero,
    v,
    k=None,
    k_residual=None,
    ffa_decode_kernel: str | None = None,
    use_fp8_residual: bool = True,
    **kwargs,
):
    kernel_name = _normalize_kernel_name(ffa_decode_kernel)
    if kernel_name in ("q2fp8", "q2_fp8", "fp8"):
        kwargs.pop("use_fp_k", None)
        return _attn_forward_decode_q2fp8(
            q=q,
            k_q=k_q,
            k_scale=k_scale,
            k_zero=k_zero,
            v=v,
            k_residual=k_residual,
            use_fp8_residual=use_fp8_residual,
            **kwargs,
        )
    kwargs.pop("use_fp8_residual", None)
    return _attn_forward_decode_q2(
        q=q,
        k_q=k_q,
        k_scale=k_scale,
        k_zero=k_zero,
        v=v,
        k=k,
        **kwargs,
    )
