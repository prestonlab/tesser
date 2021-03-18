import setuptools
import glob
from Cython.Build import cythonize

scripts = glob.glob('bin/*')
setuptools.setup(scripts=scripts, ext_modules=cythonize('src/tesser/*.pyx'))
