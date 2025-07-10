from setuptools import setup
from torch.utils import cpp_extension
import os

debug = True


setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            "gemm_kernels._C",
            [
                os.path.join("gemm_kernels","__init__.cpp"),
                os.path.join("gemm_kernels","sm90a", "debug_mbarrier.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-Wall", "-Wextra", "-pedantic", "-std=c++17"],
                "nvcc": (
                    ["--resource-usage", "--restrict","-arch=sm_90a"] + ["--keep", "-lineinfo"]
                    if debug
                    else []
                ),
            },
            extra_link_args=["-Wl,--no-as-needed"],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
