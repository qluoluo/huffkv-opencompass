import torch
import triton

@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr):
    pass

def solve_add(
    a: torch.Tensor, 
    b: torch.Tensor,
    c: torch.Tensor,
    N: int,
):
    grid = (1,)
    add_kernel[grid](a, b, c)

if __name__ == "__main__":
    N = 16
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    c = a + b
    print(a, b, c, sep="\n")

