from setuptools import setup, Extension
import pybind11
import sys

cpp_args = ['-std=c++11']
if sys.platform == 'darwin':
    cpp_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.7'])

ext_modules = [
    Extension(
        'forecast_cpp',
        ['forecast_cpp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='forecast_cpp',
    version='1.0',
    description='C++ forecasting extension',
    ext_modules=ext_modules,
)
