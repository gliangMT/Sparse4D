import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

if (hasattr(torch, 'musa') and hasattr(torch.musa, 'is_available') and torch.musa.is_available()) \
        or os.getenv('FORCE_MUSA', '0') == '1':
    from torch_musa.utils.musa_extension import BuildExtension
    from torch_musa.utils.musa_extension import MUSAExtension

def make_musa_ext(
    name,
    module,
    sources,
    sources_musa=[],
    extra_args=[],
    extra_include_path=[],
):
    print('Compiling {} with MUSA'.format(name))
    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if (hasattr(torch, 'musa') and hasattr(torch.musa, 'is_available') and torch.musa.is_available()) \
            or os.getenv("FORCE_MUSA", "0") == "1":
        define_macros += [("WITH_MUSA", None)]
        extension = MUSAExtension
        # extra_compile_args["mcc"] = extra_args + [
        #     "-D__CUDA_NO_HALF_OPERATORS__",
        #     "-D__CUDA_NO_HALF_CONVERSIONS__",
        #     "-D__CUDA_NO_HALF2_OPERATORS__",
        # ]
        sources += sources_musa
    else:
        print("Compiling {} without MUSA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )

def make_cuda_ext(
    name,
    module,
    sources,
    sources_cuda=[],
    extra_args=[],
    extra_include_path=[],
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    if (hasattr(torch, 'musa') and hasattr(torch.musa, 'is_available') and torch.musa.is_available()) \
            or os.getenv('FORCE_MUSA', '0') == '1':
        setup(
            name="deformable_aggregation_ext",
            ext_modules=[
                make_musa_ext(
                    "deformable_aggregation_ext",
                    module=".",
                    sources=[
                        f"src_musa/deformable_aggregation.cpp",
                        f"src_musa/deformable_aggregation_musa.mu",
                    ],
                ),
            ],
            cmdclass={"build_ext": BuildExtension},
        )
    else:
        setup(
            name="deformable_aggregation_ext",
            ext_modules=[
                make_cuda_ext(
                    "deformable_aggregation_ext",
                    module=".",
                    sources=[
                        f"src/deformable_aggregation.cpp",
                        f"src/deformable_aggregation_cuda.cu",
                    ],
                ),
            ],
            cmdclass={"build_ext": BuildExtension},
        )
