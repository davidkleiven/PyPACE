from distutils.core import setup, Extension
module1 = Extension( "categorize", sources=["C/categorize.c"], include_dirs=["C"] )

setup(
    name = "pypace",
    ext_modules=[module1]
)
