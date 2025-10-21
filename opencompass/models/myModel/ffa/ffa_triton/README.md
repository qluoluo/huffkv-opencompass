本项目用于在 GPU 上对自定义 Triton 注意力内核进行基准测试，并与 FlashAttention 对比。主脚本集中管理超参数和路径，通用工具与内核拆分在独立模块中。支持按“内核来源（文件名+函数名）”自动分桶输出。

目录结构
run_attn_bench.py：主入口，集中超参数与流程控制
kernels/attn_kernel.py：默认 Triton 内核与 Python 封装（attn_forward_all_in_one、attn_forward_fused）
utils/：
layout.py：convert_to_triton_layout、pack_k_hi_lo
bench.py：benchmark
flash.py：flash_attn_compute
cache.py：缓存命名、保存、加载
plot.py：绘图