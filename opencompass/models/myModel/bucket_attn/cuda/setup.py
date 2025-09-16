from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attn_cuda",
    ext_modules=[
        CUDAExtension(
            name="attn_cuda",
            sources=["attn_cuda.cu"],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)