from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
module1 = Extension( "categorize", sources=["C/categorize.c"], include_dirs=["C"] )
cythonMod = Extension( "cytParallel", ["pyREC/cythonOpenMP.pyx"],
libraries=["m"], extra_compile_args=["-O3", "-ffast-math", "-march=native","-fopenmp"],
extra_link_args=["-fopenmp"])

setup(
    name = "pypace",
    cmdclass={"build_ext":build_ext},
    ext_modules=[module1,cythonMod]
)
