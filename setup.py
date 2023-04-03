import os
import ctypes
import subprocess

from setuptools import setup, Extension
from Cython.Build import cythonize

import numpy
import torch.cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


version_dependent_macros = [
    "-DVERSION_GE_1_1",
    "-DVERSION_GE_1_3",
    "-DVERSION_GE_1_5",
]

extra_cuda_flags = [
    "-std=c++14",
    "-maxrregcount=50",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

openfold = CUDAExtension(
    name="attn_core_inplace_cuda",
    sources=[
        "src/kmol/vendor/openfold/utils/kernel/csrc/softmax_cuda.cpp",
        "src/kmol/vendor/openfold/utils/kernel/csrc/softmax_cuda_kernel.cu",
    ],
    include_dirs=[
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "src/kmol/vendor/openfold/utils/kernel/csrc/")
    ],
    extra_compile_args={
        "cxx": ["-O3"] + version_dependent_macros,
        "nvcc": (["-O3", "--use_fast_math"] + version_dependent_macros + extra_cuda_flags),
    },
)

graphormer = Extension("algos", ["src/kmol/vendor/graphormer/algos.pyx"], include_dirs=[numpy.get_include()])
graphormer.define_macros += [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

modules = []
modules += [openfold]
modules += cythonize([graphormer], compiler_directives={'language_level' : "3"})

setup(ext_modules=modules, cmdclass={"build_ext": BuildExtension})
