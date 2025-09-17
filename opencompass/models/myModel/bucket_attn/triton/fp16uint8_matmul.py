import torch
import triton
import triton.language as tl


# A: fp16 [M, K]
# B8: uint8 [K, N]  —— 每个元素是对应 fp16 的高 8 位
# C: fp32 [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_warps=4, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fp16_u8hi_matmul_kernel(
    A_ptr, B8_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,   # A[m, k] = A_ptr + m*stride_am + k*stride_ak
    stride_bk, stride_bn,   # B8[k, n] = B8_ptr + k*stride_bk + n*stride_bn
    stride_cm, stride_cn,   # C[m, n] = C_ptr + m*stride_cm + n*stride_cn
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # persistent tiling: group along M to reuse A tiles across many N tiles
    group_size = GROUP_M * num_pid_n
    group_id = pid // group_size
    first_pid_m = group_id * GROUP_M
    pid_in_group = pid % group_size
    pid_m = first_pid_m + (pid_in_group % GROUP_M)
    pid_n = pid_in_group // GROUP_M

    # guard: some programs may fall outside
    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptrs = B8_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # hint for better codegen / tensor cores
    tl.multiple_of(rk, 16)
    tl.multiple_of(rm, 16)
    tl.multiple_of(rn, 16)

    for k0 in range(0, K, BLOCK_K):
        k_mask_row = (k0 + rk) < K
        # load A tile (fp16)
        a = tl.load(
            A_ptrs,
            mask=(rm[:, None] < M) & k_mask_row[None, :],
            other=0.0,
            eviction_policy='evict_last'  # try to keep A around
        )

        # load B8 tile (uint8), then widen/shift/bitcast to fp16 on the fly
        b8 = tl.load(
            B_ptrs,
            mask=k_mask_row[:, None] & (rn[None, :] < N),
            other=0,
            eviction_policy='evict_first'  # B 通常更“流”
        )
        b16 = b8.to(tl.uint16) << 8
        b = tl.cast(b16, tl.float16)  # reinterpret as fp16, no numeric convert

        # tensor-core friendly dot; acc in fp32
        acc += tl.dot(a, b, out_dtype=tl.float32)

        # advance
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # store
    C_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def fp16_u8hi_matmul(A_fp16: torch.Tensor, B_hi_u8: torch.Tensor) -> torch.Tensor:
    # A: [M, K] fp16, B_hi_u8: [K, N] uint8, return C: [M, N] fp32
    assert A_fp16.dtype == torch.float16 and B_hi_u8.dtype == torch.uint8
    assert A_fp16.is_cuda and B_hi_u8.is_cuda and A_fp16.device == B_hi_u8.device
    assert A_fp16.shape[1] == B_hi_u8.shape[0]
    M, K = A_fp16.shape
    K2, N = B_hi_u8.shape
    C = torch.empty((M, N), device=A_fp16.device, dtype=torch.float32)

    # strides in elements
    stride_am, stride_ak = A_fp16.stride()
    stride_bk, stride_bn = B_hi_u8.stride()
    stride_cm, stride_cn = C.stride()

    # launch grid: number of program instances
    grid = (triton.cdiv(M, 128) * triton.cdiv(N, 256),)  # will be overridden by autotune configs anyway

    fp16_u8hi_matmul_kernel[grid](
        A_fp16, B_hi_u8, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    return C

if __name__ == "__main__":
    # 例子：单个 A 向量对很多列 B 取内积
    # 把 A 视为 [1, K]，把很多个 B 列拼成 [K, N] 的 uint8 矩阵
    K, N = 4096, 8192
    A = torch.randn((1, K), device='cuda', dtype=torch.float16)
    B8 = torch.randint(0, 256, (K, N), device='cuda', dtype=torch.uint8)
    C = fp16_u8hi_matmul(A, B8)  # C shape [1, N], fp32