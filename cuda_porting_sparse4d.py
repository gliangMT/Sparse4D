import os
from torch_musa.utils.simple_porting import SimplePorting

golden_path = './projects/mmdet3d_plugin/ops/src'
if os.path.exists(golden_path):
    SimplePorting(
        cuda_dir_path=golden_path,
        mapping_rule={
            "_CU_H_": "_MU_H_",
            "_CUH": "_MUH",
            "__NVCC__": "__MUSACC__",
            "MMCV_WITH_CUDA": "MMCV_WITH_MUSA",
            "AT_DISPATCH_FLOATING_TYPES_AND_HALF": "AT_DISPATCH_FLOATING_TYPES",
            "#include <ATen/cuda/CUDAContext.h>": "#include \"torch_musa/csrc/aten/musa/MUSAContext.h\"",
            "#include <c10/cuda/CUDAGuard.h>": "#include \"torch_musa/csrc/core/MUSAGuard.h\"",
            "::cuda::": "::musa::",
            "/cuda/": "/musa/",
            ", CUDA,": ", PrivateUse1,",
            ".cuh": ".muh",
            ".is_cuda()": ".is_privateuseone()",
        }
    ).run()