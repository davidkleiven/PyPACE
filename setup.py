from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
module1 = Extension( "categorize", sources=["C/categorize.c"], include_dirs=["C"] )

module2 = Extension( "pypaceCython", ["pypace/pypaceCythonMP.pyx"], libraries=["m"], extra_compile_args=["-O3", "-ffast-math", "-march=native","-fopenmp"],
extra_link_args=["-fopenmp"])

module3 = Extension("missingdatac", sources=["C/missingData.c"], include_dirs=["C"],
extra_compile_args=["-O3", "-ffast-math", "-march=native","-fopenmp"],
extra_link_args=["-fopenmp", "-std=c++11"])

setup(
    name = "pypace",
    cmdclass={"build_ext":build_ext},
    ext_modules=[module1,module2,module3]
)
