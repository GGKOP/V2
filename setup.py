from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_extension',
    ext_modules=[
        CUDAExtension('flash_attention', [
            'main.cpp',
            'flash_attention.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)