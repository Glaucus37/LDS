from distutils.core import setup
from Cython.Build import cythonize
import numpy
# cython: language_level = 3

setup(
    ext_modules = cythonize('functions.pyx'),
    include_dirs = [numpy.get_include()],
    )
