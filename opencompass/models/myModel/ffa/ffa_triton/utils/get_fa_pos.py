import inspect
import flash_attn
from flash_attn import flash_attn_func
import torch

# _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward

print("flash_attn 包路径:", flash_attn.__file__)
print("flash_attn 版本:", getattr(flash_attn, "__version__", None))
print("flash_attn_func 定义于:", inspect.getfile(flash_attn_func))
print("flash_attn_func 源代码片段:\n", inspect.getsource(flash_attn_func))
