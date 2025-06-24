from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='minimal_cuda_example',
    ext_modules=[
        CUDAExtension(
            name='minimal_cuda_example',
            sources=[
                'torch_bindings/bindings.cpp',
                'src/cuda_kernels.cu',
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 
